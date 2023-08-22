import random
import json
from enum import Enum
import uuid
import textwrap
from typing import Any, Dict, List, Tuple, Type, Union 
import jsonschema
from pydantic import BaseModel, Field, ValidationError, create_model
from openai.error import RateLimitError
from jinja2 import Template as JinjaTemplate

from .collection import Collection, Entity
from .logging import *
from .models import ChatLog, LLM, MockLLM
from .tool import Tool, ToolList


class TemplateDetails(BaseModel):
    name: str
    instructions: str = None


class MaxRetriesExceeded(Exception):
    pass


JSON_TYPE_MAP: Dict[str, Type[Union[str, int, float, bool, Any]]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
}

class Template:
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
    - {{field.name}} (type: {{field.type_}}, required: {{field.required}}){% if field.description != None %}: {{field.description}}{% endif %}
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
    input: Type[BaseModel] = None
    output: Type[BaseModel] = None
    llm: LLM = MockLLM()

    def __init__(self, instructions=None, output=None, context=None, template=None, examples=None, input=None, id=None, num_examples=None, history=None, llm=None, logger=None, debug=False, silent=False, tools: ToolList = None, name=None):
        super().__init__()

        self.id = id or self.id or str(uuid.uuid4())
        self.name = name or self.name
        self.logger = logger
        self.llm = llm or self.llm
        self.context = context or self.context
        self.history = history or self.history
        self.input = input or self.input
        self.output = output or self.output
        self.instructions = textwrap.dedent(instructions or self.instructions or '')

        self.num_examples = num_examples or self.num_examples
        self.examples = examples or self.examples
        self.input_template = JinjaTemplate(self.input_template)
        self.output_template = JinjaTemplate(self.output_template)
        self.example_template = JinjaTemplate(self.example_template)
        self.format_template = JinjaTemplate(self.format_template)
        self.tools = [Tool.parse(t) for t in tools] if tools is not None else []

        template = template or self.template
        if template is not None:
            self.template = JinjaTemplate(template)
    
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
    
    def format_field(self, name, field):
        description = field.get('description', '')
        options = ''

        if field.get('type') == 'array':
            item_type = field.get('items', {}).get('type', None)
            type_ = f'{item_type}[]'
            if isinstance(item_type, type(Enum)):
                type_ = 'str[]'
                options += f'''Select any relevant options from: {", ".join([
                    member.value for member in item_type
                ])}'''
        elif field.get('type') == 'enum':
            type_ = 'str'
            options += f'''Select only one option: {", ".join([
                member.value for member in field.type_
            ])}'''
        else:
            type_ = field.get('type', 'str')

        if len(options) > 0:
            instructions += ' ' + options

        return {
            'name': name,
            'title': field.get('title', None),
            'type_': type_,
            'default': field.get('default', None),
            'description': description.strip(),
        }
    
    def render_format(self, x, **kwargs):
        if self.output is None:
            return ''
        
        list_output = False

        fields = []
        for name, property in self.output.get('properties', {}).items():
            f = self.format_field(name, property)
            f['required'] = name in self.output.get('required', [])
            fields += [f]
        
        list_output = False
        return self.format_template.render({
            'fields': [field for field in fields if field is not None], 
            'list_output': list_output,
        })
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
                return Collection(rows)
            else:
                out = json.loads(output)
                jsonschema.validate(out, self.output)
                fields = {
                    name: (JSON_TYPE_MAP[field_info["type"]], ... if "default" not in field_info else field_info["default"])
                    for name, field_info in self.output["properties"].items()
                }
                m = create_model(self.output.get('title', 'Entity'), **fields)
                r = m(**out)
                return r
        else:
            return output
    
    def __dict__(self):
        return {
            'id': self.id,
            'type': 'template',
            'name': self.name or None,
            'instructions': self.instructions,
            'input': self.input.__name__ if self.input is not None else None,
            'output': self.output.__name__ if self.output is not None else None,
        }

    def __iter__(self):
        for key, value in self.__dict__().items():
            yield key, value
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
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
            print(f'FULL INPUT: {prompt_input}')
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
            except jsonschema.exceptions.ValidationError as e:
                self.logger.warn(f'Output validation failedcls: {e} {response.content}')
                return self.forward(x, retries=retries-1, **kwargs)
            except json.JSONDecodeError as e:
                self.logger.warn(f'Failed to decode JSON from {response.content}: {e}')
                return self.forward(x, retries=retries-1, **kwargs)
            except RateLimitError as e:
                self.logger.warn(f'Hit rate limit for {self}: {e}')
                return self.forward(x, retries=retries-1, **kwargs)
            self.logger.log(METRICS, response.metrics)
            return response


class ChatPrompt(Template):
    '''
    You are a helpful assistant.
    '''

    template = '''
    {{input}}
    {{output}}
    '''