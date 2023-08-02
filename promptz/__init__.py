import os
import glob
import importlib
import threading
import inspect
import logging
import random
import uuid
import textwrap
import base64
import json
from json import JSONDecodeError
from enum import Enum
from typing import Type, Callable, List, Tuple, Dict, Union, Any, get_origin, get_args
from abc import abstractmethod
from IPython.display import display, HTML, Image
import requests
import torch
from torch import nn
import numpy as np
import pandas as pd
from jinja2 import Template
import nbformat
from nbconvert import HTMLExporter
from dash import Dash, html, dcc, dash_table, page_container, page_registry, register_page, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Output, Input, State
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML
import dash_bootstrap_components as dbc
from pydantic import BaseModel, ValidationError 
from datetime import datetime
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaForCausalLM, LlamaTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
import openai
from openai.error import RateLimitError
import chromadb
import colorama

colorama.just_fix_windows_console()

API_URL = os.getenv('API_URL', 'http://localhost:8000')


FORMAT_INSTRUCTIONS = logging.DEBUG + 1
CONTEXT = logging.DEBUG + 2
EXAMPLES = logging.DEBUG + 3
METRICS = logging.DEBUG + 4
HISTORY = logging.DEBUG + 5
INSTRUCTIONS = logging.INFO + 1
INPUT = logging.INFO + 2
OUTPUT = logging.INFO + 3

logging.addLevelName(FORMAT_INSTRUCTIONS, "FORMAT INSTRUCTIONS")
logging.addLevelName(CONTEXT, "CONTEXT")
logging.addLevelName(EXAMPLES, "EXAMPLES")
logging.addLevelName(INSTRUCTIONS, "INSTRUCTIONS")
logging.addLevelName(INPUT, "INPUT")
logging.addLevelName(OUTPUT, "OUTPUT")
logging.addLevelName(METRICS, "METRICS")
logging.addLevelName(HISTORY, "HISTORY")

class NotebookFormatter(logging.Formatter):

    colors = {
        FORMAT_INSTRUCTIONS: colorama.Fore.CYAN,
        CONTEXT: colorama.Fore.BLACK + colorama.Back.YELLOW,
        EXAMPLES: colorama.Fore.MAGENTA,
        METRICS: colorama.Fore.WHITE + colorama.Style.DIM,
        HISTORY: colorama.Back.WHITE + colorama.Fore.BLACK,
        INSTRUCTIONS: colorama.Fore.YELLOW,
        INPUT: colorama.Fore.BLUE,
        OUTPUT: colorama.Fore.GREEN,
        logging.ERROR: colorama.Fore.RED,
        logging.INFO: colorama.Fore.WHITE,
        logging.DEBUG: colorama.Style.DIM,
    }

    def __init__(self, width=80, **kwargs):
        super().__init__(**kwargs)
        self.width = width

    def format(self, record) -> str:
        color = self.colors.get(record.levelno, None)
        msg = f'{color}{record.getMessage()}{colorama.Style.RESET_ALL}'
        return '\n'.join([
            textwrap.fill(line.strip(), width=self.width).ljust(self.width, ' ')
            for line in msg.split('\n')
        ])


class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "message": msg,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
        }
        return json.dumps(log_entry)


class Callback(BaseModel):
    name: str
    params: Dict[str, Any] = None


class Metrics(BaseModel):
    model: str = None
    input_tokens: int = None
    output_tokens: int = None

    @property
    def total_tokens(self):
        return self.input_length + self.output_length


class Response(BaseModel):
    raw: str = None
    content: Any = None
    metrics: Metrics = None
    callback: Callback = None
    cached: bool = False


class LLM:

    @abstractmethod
    def generate(self, x) -> Response:
        """Returns the generated output from the model"""


class MockLLM(LLM):
    response_length: int 
    output = None

    def __init__(self, response_length=1000, output=None):
        self.response_length = response_length
        self.output = output

    def generate(self, x, tools=None, **kwargs):
        if self.output is None:
            response = 'This is a mock response.'
        elif isinstance(self.output, BaseModel):
            response = ''
        return Response(
            raw=response,
            metrics=Metrics(
                model='mock',
                input_tokens=len(x),
                output_tokens=self.response_length,
            ),
        )


class GPT(LLM):
    model = 'text-davinci-003'

    def __init__(self, model=None):
        self.model = model or self.model

    def generate(self, x, **kwargs) -> Response:
        output = openai.Completion.create(
            model=self.model,
            prompt=x
        )
        text = output.choices[0].text
        return Response(
            raw=text,
            metrics=Metrics(
                model=f'{self.__class__.__name__}.{self.model}',
                input_tokens=len(x),
                output_tokens=len(text),
            )
        )


class ChatLog(BaseModel):
    prompt: str = None
    input: str = None
    output: str = None


class ChatGPT(LLM):
    model = 'gpt-3.5-turbo'
    context = '''
    You are a helpful chat assistant.
    '''

    def __init__(self, model=None, context=None, api_key=None, org_id=None):
        self.model = model or self.model
        openai.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        openai.organization = org_id or os.environ.get('OPENAI_ORG_ID')

    def generate(self, x, context=None, history: List[ChatLog]=None, tools=None, **kwargs):
        context = { 'role': 'system', 'content': context or self.context}
        history = history or []
        messages = [context]
        for log in history:
            messages.append({ 'role': 'user', 'content': log.input })
            messages.append({ 'role': 'system', 'content': log.output })
        messages.append({ 'role': 'user', 'content': x })
        
        # OpenAI API has an annoying "feature" where it will throw 
        # an error if you set functions to an empty list or None.
        if tools is None or len(tools) == 0:
            output = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
            )
        else:
            output = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                functions=tools,
            )
        
        message = output.choices[0].message
        function_call = message.get('function_call')
        callback = None
        if function_call is not None:
            callback = Callback(
                name=function_call.get('name'),
                params=message,
            )
        return Response(
            raw=message.get('content'),
            callback=callback,
            metrics=Metrics(
                model=f'{self.__class__.__name__}.{self.model}',
                input_tokens=output.usage.get('prompt_tokens'),
                output_tokens=output.usage.get('completion_tokens')
            ),
        )


class ImageResponse(Response):
    
    def __repr__(self) -> str:
        image_bytes = base64.b64decode(self.raw)
        display(Image(data=image_bytes))


class DALL_E(LLM):
    
    def __init__(self, api_key=None, org_id=None):
        openai.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        openai.organization = org_id or os.environ.get('OPENAI_ORG_ID')

    def generate(self, x, **kwargs) -> Response:
        output = openai.Image.create(
            prompt=x,
            n=1,
            size='512x512',
            response_format='b64_json',
        )
        return ImageResponse(
            raw=output['data'][0]['b64_json'],
            metrics=Metrics(
                input_tokens=len(x),
            )
        )


class HuggingfaceTransformer(LLM):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    max_length: int

    def __init__(self, model, tokenizer, max_length=50, **kwargs):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, x, context=None, **kwargs):
        input_tokens = self.tokenizer.encode(x, return_tensors='pt')
        generated_tokens = input_tokens
        with torch.no_grad():
            for _ in range(self.max_length):
                predictions = self.model(generated_tokens)[0]
                predicted_token = torch.argmax(predictions[0, -1, :]).unsqueeze(0)
                generated_tokens = torch.cat(
                    (generated_tokens, predicted_token.unsqueeze(0)), dim=1)
        
        generated_text = self.tokenizer.decode(
            generated_tokens[0][input_tokens.shape[1]:])

        return Response(
            raw=generated_text,
            metrics=Metrics(
                model=f'{self.__class__.__name__}',
                input_tokens=len(x),
                output_tokens=len(generated_text),
            ),
        )


class Llama(HuggingfaceTransformer):
    
    def __init__(self, model_path):
        model = LlamaForCausalLM.from_pretrained(model_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        super().__init__(model, tokenizer)


def get_function_info(func):
    signature = inspect.signature(func)
    parameters = signature.parameters

    func_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    for param_name, param in parameters.items():
        param_info = {
            "type": str(param.annotation),
            "description": param.default.__doc__ if param.default else None
        }
        func_info["parameters"][param_name] = param_info

    return func_info


class Tool:
    name: str
    description: str = None
    params = []
    function: callable = None

    def __init__(self, name=None, description=None, params=None, function=None):
        self.name = name or self.name
        self.description = description or self.description
        self.params = params or self.params
        self.function = function or self.function
    
    @classmethod
    def parse(cls, tool):
        if isinstance(tool, Tool):
            return tool
        elif callable(tool):
            tool_info = get_function_info(tool)
            tool = Tool(**{'function': tool, **tool_info})
            return tool
    
    @property
    def info(self):
        func_info = {
            "name": self.name,
            "description": self.description,
            "params": self.params,
        }
        return func_info
    
    def exec(self, *args, **kwargs):
        return self.function(**kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.exec(*args, **kwargs)


ToolList = Union[List[Tool], List[callable]]

class MaxRetriesExceeded(Exception):
    pass


class StringOutput(BaseModel):
    type_ = 'str'
    output: str


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

    def __init__(self, instructions=None, output=None, context=None, template=None, examples=None, num_examples=None, history=None, llm=None, logger=None, debug=False, silent=False, tools: ToolList = None, name=None):
        super().__init__()

        self.id = str(uuid.uuid4())
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
            except JSONDecodeError as e:
                self.logger.warn(f'Failed to decode JSON from {response.content}: {e}')
                return self.forward(x, retries=retries-1, **kwargs)
            except RateLimitError as e:
                self.logger.warn(f'Hit rate limit for {self}: {e}')
                return self.forward(x, retries=retries-1, **kwargs)
            self.logger.log(METRICS, response.metrics)
            return response


class Query(BaseModel):
    query: str
    where: Dict[str, Any] = None
    collection: str = None


class ChatPrompt(Prompt):
    '''
    You are a helpful assistant.
    '''

    template = '''
    {{input}}
    {{output}}
    '''


class VectorDB:

    @abstractmethod
    def query(self, texts, where=None, **kwargs):
        '''
        Query embeddings using a list of texts and optional where clause.
        '''

    @abstractmethod
    def get_or_create_collection(self, name, **kwargs):
        '''
        Return a collection or create a new one if it doesn't exist.
        '''


class ChromaVectorDB(VectorDB):

    def __init__(self, endpoint=None, api_key=None, **kwargs):
        self.client = chromadb.Client()

    def query(self, texts, where=None, **kwargs):
        return self.client.query(texts, where=where, **kwargs)
    
    def get_or_create_collection(self, name, **kwargs):
        return self.client.get_or_create_collection(name, **kwargs)


class Entity(BaseModel):

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def __repr__(self):
        return self.json()
    
    def display(self):
        # Check if we're in an IPython environment
        try:
            get_ipython
        except NameError:
            # If we're not in an IPython environment, fall back to json
            return self.json()

        # Convert the dictionary to a HTML table
        html = '<table>'
        for field, value in self.dict().items():
            html += f'<tr><td>{field}</td><td>{value}</td></tr>'
        html += '</table>'

        # Display the table
        display(HTML(html))
        return self.json()


class EntitySeries(pd.Series):

    @property
    def _constructor(self):
        return EntitySeries

    @property
    def _constructor_expanddim(self):
        return Collection
    
    @property
    def object(self):
        d = self.to_dict()
        return Entity(**d)


class Collection(pd.DataFrame):
    _metadata = ['collection']

    @property
    def _constructor(self, *args, **kwargs):
        return Collection
    
    @property
    def _constructor_sliced(self):
        return EntitySeries
    
    @classmethod
    def load(cls, collection):
        records = collection.get(where={'item': 1})
        docs = [
            {'id': id, **json.loads(r)} 
            for id, r in zip(records['ids'], records['documents'])
        ]
        c = Collection(docs)
        c.collection = collection
        return c
    
    def embedding_query(self, *texts, where=None, threshold=0.69, **kwargs):
        texts = [t for t in texts if t is not None]
        
        scores = {}
        if len(texts) == 0:
            results = self.collection.get(where=where, **kwargs)
            for id, m in zip(results['ids'], results['metadatas']):
                if m.get('item') != 1:
                    id = m.get('item_id')
                if id not in scores:
                    scores[id] = 1
                else:
                    scores[id] += 1
        else:
            results = self.collection.query(query_texts=texts, where=where, **kwargs)
            for i in range(len(results['ids'])):
                for id, d, m in zip(results['ids'][i], results['distances'][i], results['metadatas'][i]):
                    if m.get('item') != 1:
                        id = m.get('item_id')
                    if id not in scores:
                        scores[id] = 1 - d
                    else:
                        scores[id] += 1 - d
        
        try:
            df = self.copy()
            df['score'] = df['id'].map(scores)
            df = df[df['score'].notna()]
            df = df.sort_values('score', ascending=False)
            df = df.drop(columns=['score'])
            return df
        except KeyError as e:
            return None
    
    def __call__(self, *texts, where=None, **kwargs) -> Any:
        return self.embedding_query(*texts, where=where, **kwargs)
    
    @property
    def name(self):
        return self.collection.name
    
    @property
    def objects(self):
        return [r.object for _, r in self.iterrows()]
    
    @property
    def first(self):
        return self.objects[0]
    
    def embed(self, *items, **kwargs):
        records = []
        new_items = []
        unique_columns = {}
        for item in items:
            new_item = False
            try:
                id = item.id
            except AttributeError:
                id = str(uuid.uuid4())
                new_item = True

            now = datetime.now().isoformat()

            for name, field in item.__fields__.items():
                try:
                    if field.field_info.default[0].extra.get('unique'):
                        unique_columns[name] = True
                except Exception as e:
                    pass

                f = { name: getattr(item, name) }
                # TODO: Handle nested fields
                field_record = {
                    'id': f'{id}_{name}',
                    'document': json.dumps(f),
                    'metadata': {
                        'field': name,
                        'collection': self.name,
                        'item': 0,
                        'item_id': id,
                        'created_at': now,
                    },
                }
                records.append(field_record)
                if new_item: 
                    new_items.append(field_record)

            type_ = getattr(item, 'type', item.__class__.__name__.lower())
            doc_record = {
                'id': id,
                'document': item.json(),
                'metadata': {
                    'collection': self.name,
                    'type': type_,
                    'item': 1,
                    'created_at': now,
                },
            }

            records.append(doc_record)
            if new_item:
                new_items.append(doc_record)
            
        self.collection.upsert(
            ids=[r['id'] for r in records],
            documents=[r['document'] for r in records],
            metadatas=[r['metadata'] for r in records],
        )

        docs = [json.loads(r['document']) for r in new_items]
        _collection = self.collection
        self = pd.concat([self, Collection(docs)], ignore_index=True)
        if len(unique_columns) > 0:
            self.drop_duplicates(subset=list(unique_columns.keys()), inplace=True)
        self.collection = _collection
        return self


class Session:
    _history = []
    _collections = {}
    use_cache = False
    cache_threshold = 85.0

    def __init__(self, name, db, llm, ef, default_collection='default', logger=None, collections=None, use_cache=False):
        self.name = name
        self.db = db
        self.llm = llm
        self.ef = ef
        self._collection = default_collection
        if collections is not None:
            self._collections = collections 
        self.logger = logger
        self._history = []
        self.use_cache = use_cache
    
    def activate(self):
        set_default_session(self)
    
    def _run_prompt(self, p, input, dryrun=False, retries=3, **kwargs):
        e = None
        try:
            rendered = p.render({'input': p.parse(input)})
            if self.use_cache:
                rs = self.history(rendered)
                if rs is not None and len(rs) > 0:
                    scores = self.evaluate(*[
                        (rendered, r.input) for r in rs.objects
                    ])
                    scores = scores[scores['score'] >= self.cache_threshold]
                    if len(scores) > 0:
                        r = rs.first
                        return Response(
                            raw=r.output,
                            cached=True,
                            content=p.process(input, r.output, **kwargs),
                        )

            r = p(input, dryrun=dryrun, retries=retries, **kwargs)
            log = ChatLog(prompt=p.id, input=rendered, output=r.raw)
            self._history.append(log)
            collection = self.collection('history')
            collection.embed(log)
            return r
        except MaxRetriesExceeded as e:
            self.logger.error(f'Max retries exceeded: {e}')
            return None
    
    def _run_batch(self, p, inputs, dryrun=False, retries=3, **kwargs):
        o = []
        for input in inputs:
            r = self._run_prompt(p, input, dryrun=dryrun, retries=retries, **kwargs)
            if r is not None:
                o.append(r.content)
        return o

    def prompt(self, instructions=None, input=None, output=None, prompt=None, context=None, template=None, llm=None, examples=None, num_examples=1, history=None, tools=None, dryrun=False, retries=3, debug=False, silent=False, **kwargs):
        logger = self.logger.getChild('prompt')
        level = logging.INFO
        if debug: level = logging.DEBUG
        elif silent: level = logging.ERROR
        logger.setLevel(level)

        if prompt is None:
            p = Prompt(
                output=output,
                instructions=instructions,
                llm=llm or self.llm,
                context=context,
                examples=examples,
                num_examples=num_examples,
                history=history,
                tools=tools,
                template=template,
                debug=debug,
                silent=silent,
                logger=logger,
            )
        else:
            p = prompt
            p.llm = llm or self.llm
            p.logger = self.logger
            p.debug = debug
            p.silent = silent

        if isinstance(input, list):
            o = self._run_batch(p, input, dryrun=dryrun, retries=retries, **kwargs)
        else:
            r = self._run_prompt(p, input, dryrun=dryrun, retries=retries, **kwargs)
            if r is None:
                return None
            o = r.content
        return o
    
    def embed(self, item, field=None):
        if isinstance(item, str):
            return self.ef([item])[0]
        elif isinstance(item, BaseModel):
            return self.ef([item.json()])[0]
        elif isinstance(item, list):
            return [self.embed(i, field=field) for i in item]
    
    def query(self, *texts, field=None, where=None, collection=None):
        c = self.collection(collection)
        if len(texts) == 0 and field is None and where is None:
            return c
        where = where or {}
        if field is not None:
            where['field'] = field
        return c(*texts, where=where)

    def store(self, *items, collection=None):
        def flatten(lst):
            return [
                item for sublist in lst 
                for item in (
                    flatten(sublist) if isinstance(sublist, list) else [sublist]
                )
            ]
        
        items = flatten([
            item.objects if isinstance(item, Collection) else item 
            for item in items
        ])
        
        c = self.collection(collection)
        c.embed(*[item for item in items if item is not None])
        return self.collection(collection)
    
    def chain(self, *steps, llm=None, **kwargs):
        llm = llm or self.llm
        input = None
        cc = self.collection(f'chain.{uuid.uuid4()}')
        for step in steps:
            if isinstance(step, Prompt):
                p = step
                p.llm = llm
                input = self._run_prompt(p, input, **kwargs)
                self.store(*input, collection=cc.name)
            if isinstance(step, str):
                p = Prompt(step, llm=llm)
                input = self._run_prompt(p, input, **kwargs)
                if isinstance(input, list):
                    if isinstance(input[0], BaseModel):
                        self.store(*input, collection=cc.name)
                elif isinstance(input, BaseModel):
                    self.store(input, collection=cc.name)
            if isinstance(step, dict):
                input = {
                    k: self.chain(v, llm=llm, **kwargs)
                    for k, v in step.items()
                }
            if isinstance(step, list):
                input = [
                    self.chain(v, llm=llm, **kwargs)
                    for v in step
                ]
            if isinstance(step, Query):
                qc = self.collection(step.collection) if step.collection is not None else cc
                input = qc(step.query or '', where=step.where)
        return cc
        
    def collection(self, name=None):
        if name is None:
            name = self._collection
        try:
            collection = self._collections[name]
        except KeyError:
            collection = self.db.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
            self._collections[name] = collection
        
        return Collection.load(collection)
    
    def score(self, x, y, threshold=0.69):
        ex = self.embed(x)
        ey = self.embed(y)
        dot_product = np.dot(ex, ey)
        norm_1 = np.linalg.norm(ex)
        norm_2 = np.linalg.norm(ey)

        cosine_similarity = dot_product / (norm_1 * norm_2)
        if cosine_similarity < threshold:
            cosine_similarity = 0
        rescaled = (cosine_similarity - threshold) * (100 / (1 - threshold))
        return max(0, rescaled)
    
    def evaluate(self, *testcases, **kwargs):
        rows = [(self.score(x, y), x, y) for x, y in testcases]
        df = pd.DataFrame(rows, columns=['score', 'result', 'expected'])
        return df
    
    def __call__(self, collection: Collection, *args, **kwargs):
        return self.run(collection, *self._processors, **kwargs)
    
    @property
    def history(self):
        return self.collection('history')


Processor = Callable[[Collection], Collection]

class System:
    query: Query
    processor: Processor

    def __init__(self, query=None, processor=None, **kwargs):
        self.query = query or self.query
        self.processor = processor 
    
    def process(self, items, **kwargs):
        if self.processor is not None:
            return self.processor(items, **kwargs)
        else:
            raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.process(*args, **kwds)


class World:
    name: str
    sessions: List[Session]
    collections: Dict[str, Collection]
    systems: Dict[str, System]
    prompts: Dict[str, Prompt]
    notebooks: Dict[str, str]

    def __init__(self, name, systems=None, llm=None, ef=None, logger=None, db=None, prompts=None, notebooks=None):
        self.name = name
        self.sessions = []
        self.collections = {}
        self.llm = llm or MockLLM()
        self.ef = ef or (lambda x: [0] * len(x))
        self.db = db or ChromaVectorDB()
        self.logger = logger or logging.getLogger(self.name)
        self.prompts = prompts or {}
        self.systems = systems or {}
        self.notebooks = notebooks or {}

        collection = self.db.get_or_create_collection('history', metadata={"hnsw:space": "cosine"})
        self.collections['history'] = collection
    
    def create_session(self, name=None, db=None, llm=None, ef=None, logger=None, silent=False, debug=False, use_cache=False, log_format='notebook'):
        llm = llm or self.llm
        ef = ef or self.ef
        db = db or self.db
        logger = logger or self.logger.getChild(f'session.{name}')
        ch = logging.StreamHandler()
        formatter = JSONLogFormatter() if log_format == 'json' else NotebookFormatter()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        level = logging.INFO
        if debug: level = logging.DEBUG
        elif silent: level = logging.ERROR
        logger.setLevel(level)
        session = Session(name=name, db=db, llm=llm, ef=ef, logger=logger,
                          collections=self.collections, use_cache=use_cache)
        self.sessions.append(session)
        return session
    
    def __call__(self, session, *args: Any, **kwds: Any) -> Any:
        for system in self.systems.values():
            items = session.query(system.query)
            updates = system(items)
            session.store(updates)


class PromptInput(BaseModel):
    input: Any


class API:
    world: World

    def __init__(self, world):
        self.world = world
        self.fastapi_app = FastAPI()
        
        @self.fastapi_app.get("/prompts")
        async def get_prompts():
            return {"response": {k: v.dict() for k, v in self.world.prompts.items()}}

        @self.fastapi_app.get("/prompts/{id}")
        async def get_prompt(id: str):
            history = self.world.collections['history']
            c = Collection.load(history)
            if c.empty:
                results = []
            else:
                results = c[c['prompt'] == id].to_dict('records')
            prompt = self.world.prompts[id]
            print('results', results, c, id)
            return {'prompt': prompt, 'results': results}

        @self.fastapi_app.post("/prompts/{name}/run")
        async def run_prompt(name: str, input: PromptInput = None):
            session = self.world.create_session()
            if name not in self.world.prompts:
                raise HTTPException(status_code=404, detail="Prompt not found")
            prompt_config = self.world.prompts[name]
            d = {**prompt_config, 'input': input}
            response = session.prompt(**{**prompt_config, 'input': input})
            return {"response": response}
        
        @self.fastapi_app.get("/systems")
        async def get_systems():
            return {"response": self.world.systems.keys()}
        
        @self.fastapi_app.get("/systems/{name}")
        async def get_system(name: str):
            return {"response": self.world.systems[name]}
        
        @self.fastapi_app.post("/systems/run")
        async def run_systems():
            session = self.world.create_session()
            return {"response": self.world(session)}
        
        @self.fastapi_app.post("/query")
        async def query(query: Query):
            session = self.world.create_session()
            response = session.query(query.query, where=query.where, collection=query.collection)
            return {"response": response}
        
        @self.fastapi_app.get("/notebooks")
        async def get_notebooks():
            return {"response": self.world.notebooks}
        
        @self.fastapi_app.get("/notebooks/{name}")
        async def get_notebook(name: str):
            return {"response": self.world.notebooks[name]}
        
        @self.fastapi_app.get("/chats")
        async def get_chats():
            return {"response": {}}


class Admin:
    world: World

    def __init__(self, world):
        self.world = world
        self.app = Dash(
            world.name, 
            use_pages=True,
            pages_folder='admin',
            external_stylesheets=[dbc.themes.ZEPHYR],
        )

        register_page(
            'Inbox',
            layout=html.Div(children=[
            ]),
            path='/',
        )

        def prompts_list_layout():
            response = requests.get(f'{API_URL}/prompts')
            if response.status_code == 200:
                prompts = response.json()
            else:
                raise Exception(f'Error getting prompts: {response.status_code}')
            
            print('prompts', prompts)
            return html.Div(children=[
                html.H1(children='Prompts'),
                html.Ul([
                    html.Li(html.A(p['name'], href=f'/prompts/{id}')) for id, p in prompts['response'].items()
                ]),
            ])

        register_page(
            'Prompts',
            layout=prompts_list_layout,
            path='/prompts',
        )

        def prompt_layout(id: str = None):
            response = requests.get(f'{API_URL}/prompts/{id}')
            if response.status_code == 200:
                data = response.json()
                prompt = data['prompt']
                results = data['results']
            else:
                raise Exception(f'Error getting prompt: {response.status_code}')

            return html.Div(children=[
                html.H1(id),
                dbc.Button('Run', id='run-prompt', n_clicks=0, name=id, color='primary'),
                dcc.Store(id='api-call-result', storage_type='session'),
                html.Div(id, id='prompt-id', style={'display': 'none'}),
                html.Div(id='prompt-results', children=[
                    html.P(result) for result in results
                ]),
            ])

        register_page('Prompt', layout=prompt_layout, path_template='/prompts/<id>')

        @self.app.callback(
            Output('api-call-result', 'data'),
            [Input('run-prompt', 'n_clicks')],
            [State('api-call-result', 'data'),
             State('prompt-id', 'children')],
        )
        def run_prompt(n_clicks, current_data, id):
            if n_clicks is None or n_clicks == 0:
                raise PreventUpdate
            else:
                requests.post(f'{API_URL}/prompts/{id}/run')

        def systems_list_layout():
            response = requests.get(f'{API_URL}/systems')
            if response.status_code == 200:
                systems = response.json()
            else:
                raise Exception(f'Error getting systems: {response.status_code}')
            
            return html.Div(children=[
                html.H1(children='Systems'),
                html.Ul([
                    html.Li(html.A(name, href=f'/systems/{name}')) for name in systems['response'].keys()
                ]),
            ])

        register_page(
            'Systems',
            layout=systems_list_layout,
            path='/systems',
        )

        def notebook_list_layout():
            response = requests.get(f'{API_URL}/notebooks')
            if response.status_code == 200:
                notebooks = response.json()
            else:
                raise Exception(f'Error getting notebooks: {response.status_code}')
            
            return html.Div(children=[
                html.H1(children='Notebooks'),
                html.Ul([
                    html.Li(html.A(name, href=f'/notebooks/{name}')) for name in notebooks['response'].keys()
                ]),
            ])

        register_page(
            'Notebooks',
            layout=notebook_list_layout,
            path='/notebooks',
        )

        def notebook_layout(name: str = None):
            response = requests.get(f'{API_URL}/notebooks/{name}')
            notebook_html = response.json()['response']
            return html.Div(children=[
                html.H1(name),
                DangerouslySetInnerHTML(notebook_html)
            ])

        register_page('Notebook', layout=notebook_layout, path_template='/notebooks/<name>')

        def chatbots_list_layout():
            response = requests.get(f'{API_URL}/chats')
            if response.status_code == 200:
                chats = response.json()
            else:
                raise Exception(f'Error getting chats: {response.status_code}')
            
            return html.Div(children=[
                html.H1(children='Chats'),
                html.Ul([
                    html.Li(html.A(name, href=f'/chats/{name}')) for name in chats['response'].keys()
                ]),
            ])

        register_page(
            'Chats',
            layout=chatbots_list_layout,
            path='/chats',
        )

        menu = [
            'Prompts',
            'Notebooks',
        ]

        self.app.layout = dbc.Row([
            dbc.Col([
                dbc.Row(
                    html.H3('promptz', style={'color': 'white'})
                ),
                *[
                    dbc.Row(
                        html.Div(
                            dcc.Link(
                                f"{page_registry[name]['name']}", href=page_registry[name]["relative_path"],
                                style={'color': 'white', 'text-decoration': 'none'}
                            )
                        ),
                    )
                    for name in menu
                ]
            ], width=3, style={'background-color': '#663399', 'padding': '10px 20px', 'height': '100vh'}),
            dbc.Col([
                page_container
            ], width=9, style={'padding': '10px'}),
        ])


class App:
    name: str
    world: World
    prompts_dir: str = 'prompts'
    systems_dir: str = 'systems'
    notebooks_dirs: str = ['notebooks']

    def __init__(self, name, world=None, llm=None, ef=None, logger=None, db=None):
        self.name = name
        self.logger = logger or logging.getLogger(self.name)
        prompts = self._load_prompts()
        systems = self._load_systems()
        notebooks = self._load_notebooks()
        self.world = world or World(name, prompts=prompts, systems=systems, notebooks=notebooks, llm=llm, ef=ef, logger=logger, db=db)
        self.api = API(self.world)
        self.admin = Admin(self.world)
    
    def _load_prompts(self):
        prompts = {}
        for file in glob.glob(os.path.join(self.prompts_dir, '*.json')):
            with open(file, 'r') as f:
                prompt = Prompt(**json.load(f))
                prompts[prompt.id] = prompt
        self.logger.info(f'Loaded {len(prompts)} prompts: {prompts}')
        return prompts
    
    def _load_notebooks(self):
        html_notebooks = {}
        for dir in self.notebooks_dirs:
            for file in glob.glob(os.path.join(dir, '*.ipynb')):
                with open(file, 'r') as f:
                    file_name = os.path.splitext(os.path.basename(file))[0]
                    html_exporter = HTMLExporter()
                    notebook_node = nbformat.read(f, as_version=4)
                    body, resources = html_exporter.from_notebook_node(notebook_node)
                    html_notebooks[file_name] = body
        return html_notebooks
    
    def _load_systems(self):
        systems = {}
        #for file in glob.glob(os.path.join(self.systems_dir, '*.py')):
        #    file_name = os.path.splitext(os.path.basename(file))[0]
        #    module = importlib.import_module(f'{self.systems_dir}.{file_name}')
        #    systems[file_name] = module
        return systems
    
    def _serve_api(self, host='0.0.0.0', port=8000):
        import uvicorn
        uvicorn.run(self.api.fastapi_app, host=host, port=port)
    
    def _serve_admin(self, host='0.0.0.0', port=8001):
        from waitress import serve
        serve(self.admin.app.server, host=host, port=port)
    
    def serve(self, host='0.0.0.0', port=8000, admin_port=8001):
        api = threading.Thread(target=lambda: self._serve_api(host=host, port=port))
        admin = threading.Thread(target=lambda: self._serve_admin(host=host, port=admin_port))
        api.start()
        admin.start()


def prompt(instructions=None, input=None, output=None, prompt=None, context=None, template=None, llm=None, examples=None, num_examples=1, history=None, tools=None, dryrun=False, retries=3, debug=False, silent=False, **kwargs):
    kwargs = dict(
        instructions=instructions,
        input=input,
        output=output,
        prompt=prompt,
        llm=llm,
        context=context,
        examples=examples,
        num_examples=num_examples,
        history=history,
        tools=tools,
        template=template,
        debug=debug,
        dryrun=dryrun,
        silent=silent,
        retries=retries,
        store=store,
    )
    return DEFAULT_SESSION.prompt(**kwargs)


def chain(*steps, **kwargs):
    return DEFAULT_SESSION.chain(*steps, **kwargs)


def store(*items, collection=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.store(*items, collection=collection, **kwargs)


def query(*texts, field=None, where=None, collection=None, **kwargs) -> Collection:
    return DEFAULT_SESSION.query(
        *texts, field=field, where=where, collection=collection, **kwargs)


def collection(name=None) -> Collection:
    return DEFAULT_SESSION.collection(name)


def evaluate(*testcases, **kwargs) -> Collection:
    return DEFAULT_SESSION.evaluate(*testcases, **kwargs)


def history(query=None, **kwargs):
    if query is not None:
        return DEFAULT_SESSION.history(query, **kwargs)
    else:
        return DEFAULT_SESSION.history


def session() -> Session:
    return DEFAULT_SESSION


def run(world=None, **kwargs):
    if world is None:
        world = DEFAULT_WORLD
    return world(**kwargs)


DEFAULT_WORLD = None
def set_default_world(world):
    global DEFAULT_WORLD
    DEFAULT_WORLD = world


DEFAULT_SESSION = None
def set_default_session(session):
    global DEFAULT_SESSION
    DEFAULT_SESSION = session


Embedding = List[float]
EmbedFunction = Callable[[List[str]], List[Embedding]]


def init(llm=None, ef=None, logger=None, use_cache=False, log_format='notebook', **kwargs):
    w = World('test', llm=llm, ef=ef, logger=logger, **kwargs)
    session = w.create_session(use_cache=use_cache, log_format=log_format)
    set_default_world(w)
    set_default_session(session)
