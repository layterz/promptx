import os
import random
import json
import inspect
from enum import Enum
import base64
import uuid
import textwrap
from typing import Any, Dict, List, Tuple, Type, Union, get_origin, get_args
from abc import abstractmethod
from pydantic import BaseModel, ValidationError
from IPython.display import display, Image
import openai
from openai.error import RateLimitError
from torch import nn
from jinja2 import Template

from .collection import Collection, Entity
from .logging import *
from .models import ChatLog, Response, LLM, MockLLM
from .tool import Tool, ToolList


class PromptDetails(BaseModel):
    name: str
    instructions: str = None


class ImageResponse(Response):
    
    def __repr__(self) -> str:
        image_bytes = base64.b64decode(self.raw)
        display(Image(data=image_bytes))


class MaxRetriesExceeded(Exception):
    pass


class Prompt(nn.Module):
    '''
    Follow the pattern shown in the examples below and
    generate a new output using the same format.
    '''

    template = """
    INSTRUCTIONS
    ---
    {{instructions}}
    {{format}}
    {{examples}}
    {{input}}
    {{output}}
    """

    input_template = """
    INPUT
    ---
    {{input}}
    END_INPUT
    """

    output_template = """
    OUTPUT
    ---
    {{output}}
    """

    example_template = f"""
    EXAMPLES
    ---
    {input_template}
    {output_template}
    END_EXAMPLES
    """

    format_template = """
    FORMAT INSTRUCTIONS
    ---
    {% if list_output %}
    Return a list of valid JSON objects with the fields described below.
    {% else %}
    Return the output as a valid JSON object with the fields described below. 
    {% endif %}
    {% for field in fields %}
    - {{field.name}} (type: {{field.type_}}, required: {{field.required}}){% if field.instructions != None %}: {{field.instructions}}{% endif %}
    {% endfor %}

    Make sure to use double quotes and avoid trailing commas!
    Ensure any required fields are set, but you can use the default value 
    if it's defined and you are unsure what to use. 
    If you are unsure about any optional fields use `null` or the default value,
    but try your best to fill them out.
    END_FORMAT_INSTRUCTIONS
    """

    id: str 
    name: str = None
    instructions: str = None
    context: str = None
    history: List[ChatLog] = []
    examples: List[Tuple[(str|BaseModel), (str|BaseModel)]] = []
    num_examples: int = 1
    output: Type[BaseModel] = None
    llm: LLM = MockLLM()

    def __init__(self, instructions=None, output=None, context=None, template=None, examples=None, id=None, num_examples=None, history=None, llm=None, logger=None, debug=False, silent=False, tools: ToolList = None, name=None):
        super().__init__()

        self.id = id or str(uuid.uuid4())
        self.name = name
        self.logger = logger
        self.name = self.__class__.__name__
        self.llm = llm or self.llm
        self.context = context or self.context
        self.history = history or self.history
        self.output = output or self.output
        instructions = instructions or self.__doc__
        if instructions is None:
            for base in inspect.getmro(self.__class__):
                if base.__doc__ is not None and issubclass(base, Prompt):
                    instructions = base.__doc__
                    break
        self.instructions = textwrap.dedent(instructions) if instructions is not None else None

        self.num_examples = num_examples or self.num_examples
        self.examples = examples or self.examples
        self.input_template = Template(self.input_template)
        self.output_template = Template(self.output_template)
        self.example_template = Template(self.example_template)
        self.format_template = Template(self.format_template)
        self.tools = [Tool.parse(t) for t in tools] if tools is not None else []

        template = template or self.template
        if template is not None:
            self.template = Template(template)
    
    def parse(self, x):
        if isinstance(x, BaseModel):
            return x.dict()
        elif isinstance(x, Entity):
            return x.object.dict()
        elif isinstance(x, Collection):
            return [y.dict() for y in x.objects]
        elif isinstance(x, str):
            return x
        elif isinstance(x, dict):
            return {k: self.parse(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self.parse(y) for y in x]
        elif x is None:
            return {}
        else:
            return x
    
    def render(self, x, **kwargs):
        input = self.input_template.render(**x) if len(x) > 0 else ''
        output = self.output_template.render(**x) if len(x) > 0 else ''
        vars = {
            **x,
            'instructions': self.instructions,
            'examples': self.render_examples(),
            'format': self.render_format(x),
            'input': input,
            'output': output,
        }
        output = self.template.render(**vars)
        return output
    
    def format_field(self, field):
        instructions = field.field_info.description or ''
        options = ''
        outer_type_ = field.outer_type_.__name__
        if issubclass(field.type_, BaseModel):
            return None

        if get_origin(field.outer_type_) is list:
            item_type = get_args(field.outer_type_)[0]
            type_ = f'{item_type.__name__}[]'
            if isinstance(item_type, type(Enum)):
                type_ = 'str[]'
                options += f'''Select any relevant options from: {", ".join([
                    member.value for member in item_type
                ])}'''
        elif isinstance(field.type_, type(Enum)):
            type_ = 'str'
            options += f'''Select only one option: {", ".join([
                member.value for member in field.type_
            ])}'''
        else:
            type_ = field.type_.__name__

        if len(options) > 0:
            instructions += ' ' + options

        return {
            'name': field.name,
            'type_': type_,
            'required': field.required,
            'default': field.default,
            'instructions': instructions.strip(),
        }
    
    def render_format(self, x, **kwargs):
        if self.output is None:
            return ''
        # check if self.output is a List
        list_output = False
        cls = self.output
        if getattr(self.output, '__origin__', None) is list:
            # if so, get the type of the list
            list_output = True
            item_type = self.output.__args__[0]
            if item_type is str:
                return 'Return an array of strings wrapped in double quotes.'
            else:
                cls = item_type
        fields = [self.format_field(f) for f in cls.__fields__.values()]
        return self.format_template.render({
            'fields': [field for field in fields if field is not None], 
            'list_output': list_output,
        })
    
    def render_examples(self, **kwargs):
        if len(self.examples) == 0:
            return ''
        examples = [
            {
                'input': json.dumps(self.parse(i)),
                'output': json.dumps(self.parse(o)),
            }
            for i, o in random.sample(self.examples, self.num_examples)
        ]
        return '\n'.join([
            self.example_template.render(**e) for e in examples
        ])
    
    def process(self, x, output, **kwargs):
        if self.output is not None:
            rows = []
            if getattr(self.output, '__origin__', None) is list:
                cls = self.output.__args__[0]
                type_ = cls.__name__.lower()
                data = json.loads(output)
                if cls is str:
                    rows += [{'type': type_, 'output': d} for d in data]
                elif issubclass(cls, BaseModel):
                    rows += [{'type': type_, **cls(**d).dict()} for d in data]
            else:
                type_ = self.output.__name__.lower()
                d = json.loads(output)
                d = {'type': type_, **d}
                return self.output(**d)
            return Collection(rows)
        else:
            return output
    
    def dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'instructions': self.instructions,
        }
    
    def forward(self, x, retries=3, dryrun=False, **kwargs):
        if retries and retries <= 0:
            e = MaxRetriesExceeded(f'{self.name} failed to forward {x}')
            self.logger.error(e)
            raise e
        
        if dryrun:
            self.logger.debug(f'Dryrun: {self.output}')
            llm = MockLLM(output=self.output)
        else:
            llm = self.llm
        
        px = self.parse(x)
            
        if self.template is not None and self.template != '':
            prompt_input = self.render({'input': px})
            if self.context: self.logger.context(self.context)
            self.logger.log(HISTORY, '\n\n\n'.join([f'''
            {log.input}
            {log.output}
            ''' for log in self.history]))
            self.logger.log(INSTRUCTIONS, self.instructions)
            if len(self.examples): self.logger.log(EXAMPLES, self.render_examples())
            if len(px): self.logger.log(INPUT, px)
            tools = [t.info for t in self.tools]
            self.logger.debug(f'FULL INPUT: {prompt_input}')
            response = llm.generate(prompt_input, context=self.context, history=self.history, tools=tools)
            if response.callback is not None:
                function_name = response.callback.name
                tool = next(t for t in self.tools if t.name == function_name)
                params = {p['name']: response.callback.params.get(p['name']) 
                          for p in tool.parameters}
                rsp = tool(**params)
                return None
            if self.output: self.logger.log(FORMAT_INSTRUCTIONS, self.render_format(px))
            try:
                self.logger.log(OUTPUT, response.raw)
                response.content = self.process(px, response.raw, **kwargs)
            except ValidationError as e:
                self.logger.warn(f'Output validation failed: {e} {response.content}')
                return self.forward(x, retries=retries-1, **kwargs)
            except json.JSONDecodeError as e:
                self.logger.warn(f'Failed to decode JSON from {response.content}: {e}')
                return self.forward(x, retries=retries-1, **kwargs)
            except RateLimitError as e:
                self.logger.warn(f'Hit rate limit for {self}: {e}')
                return self.forward(x, retries=retries-1, **kwargs)
            self.logger.log(METRICS, response.metrics)
            return response


class ChatPrompt(Prompt):
    '''
    You are a helpful assistant.
    '''

    template = '''
    {{input}}
    {{output}}
    '''