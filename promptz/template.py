import time
import random
import json
import urllib3
from typing import * 
import jsonschema
from pydantic import BaseModel
import openai
from jinja2 import Template as JinjaTemplate

from .collection import Collection, Query
from .logging import *
from .models import LLM, MockLLM
from .utils import Entity, model_to_json_schema, create_entity_from_schema


class MaxRetriesExceeded(Exception):
    pass


E = TypeVar('E', bound=BaseModel)

class Template(Entity):
    
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
    {% if string_list_output %}
    Return a JSON array of strings.
    {% elif list_output %}
    Return a list of valid JSON objects with the fields described below.
    {% else %}
    Return the output as a valid JSON object with the fields described below. 
    {% endif %}
    {% for field in fields %}
    - {{field.name}} (type: {{field.type_}}, required: {{field.required}}, default: {{field.default}}){% if field.description != None %}: {{field.description}}{% endif %}
    {% endfor %}

    Make sure to use double quotes and avoid trailing commas!
    Ensure any required fields are set, but you can use the default value 
    if it's defined and you are unsure what to use. 
    If you are unsure about any optional fields use `null` or the default value,
    but try your best to fill them out.
    END_FORMAT_INSTRUCTIONS
    """

    type: str = 'template'
    name: str = None
    instructions: str = None
    num_examples: int = 1
    examples: List = None
    input: str = None
    output: str = None
    context: str = None
    data: Query = None


class TemplateRunner:
    template: Template
    llm: LLM = MockLLM()

    def __init__(self, llm=None, logger=None, debug=False, silent=False, name=None):
        self.logger = logger or logging.getLogger(name or self.__class__.__name__)
        self.llm = llm or self.llm
    
    def parse(self, x):
        if isinstance(x, BaseModel):
            return {k: v for k, v in x.dict().items() if k not in ['id', 'type']}
        elif isinstance(x, Entity):
            return {k: v for k, v in x.object.dict().items() if k not in ['id', 'type']}
        elif isinstance(x, Collection):
            return [
                {k: v for k, v in y.dict().items() if k not in ['id', 'type']}
                for y in x.objects
            ]
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
    
    def render(self, t, x, **kwargs):
        input_template = JinjaTemplate(t.input_template)
        input = input_template.render(**x) if len(x) > 0 else ''
        output_template = JinjaTemplate(t.output_template)
        output = output_template.render(**x) if len(x) > 0 else ''
        vars = {
            **x,
            'instructions': t.instructions,
            'examples': self.render_examples(t),
            'format': self.render_format(t, x),
            'input': input,
            'output': output,
        }
        template = JinjaTemplate(t.template)
        output = template.render(**vars)
        return output
    
    def format_field(self, name, field, definitions, required):
        # TODO: hack, source should be stored on the schema
        if name in ['id', 'type', 'source']:
            return None
        description = field.get('description', '')
        options = ''

        list_field = False
        if field.get('type') == 'array':
            print('array', name, field)
            list_field = True
            item_type = field.get('items', {}).get('type', None)
            if item_type is None:
                ref = field.get('items', {}).get('$ref', None)
                ref = ref.split('/')[-1]
                definition = definitions.get(ref, {})
                type_ = f'{definition.get("title", ref)}[]'
            else:
                type_ = f'{item_type}[]'
            field = field.get('items', {})
        elif len(field.get('allOf', [])) > 0:
            print('allOf', name, field)
            ref = field.get('allOf')[0].get('$ref')
            ref = ref.split('/')[-1]
            definition = definitions.get(ref, {})
            type_ = f'{definition.get("title", ref)}'
        elif field.get('$ref'):
            print('ref', name, field)
            ref = field.get('$ref')
            ref = ref.split('/')[-1]
            definition = definitions.get(ref, {})
            type_ = f'{definition.get("title", ref)}'
        else:
            print('default', name, field)
            type_ = field.get('type', 'str')
        
        if 'enum' in field:
            if list_field:
                options += f'''
                Select any relevant options from: {", ".join(field["enum"])}
                '''
            else:
                options += f'''
                Select one option from: {", ".join(field["enum"])}
                '''

        if len(options) > 0:
            description += ' ' + options

        return {
            'name': name,
            'title': field.get('title', None),
            'type_': type_,
            'default': field.get('default', None),
            'description': description.strip(),
            'required': name in required,
        }
    
    def render_format(self, t, x, **kwargs):
        if t.output is None or t.output == str:
            return ''
        
        output = json.loads(t.output)
        format_template = JinjaTemplate(t.format_template)
        if output.get('type', None) == 'array' and output.get('items', {}).get('type', None) == 'string':
            return format_template.render({
                'string_list_output': True,
            })

        list_output = False
        fields = []
        properties = {}
        if output.get('type', None) == 'array':
            properties = output.get('items', {}).get('properties', {})
            definitions = output.get('items', {}).get('definitions', {})
            required = output.get('items', {}).get('required', [])
            list_output = True
        elif output.get('type', None) == 'object':
            properties = output.get('properties', {})
            definitions = output.get('definitions', {})
            required = output.get('required', [])
        
        for name, property in properties.items():
            f = self.format_field(name, property, definitions, required)
            fields += [f]
        
        return format_template.render({
            'fields': [field for field in fields if field is not None], 
            'list_output': list_output,
        })
    
    def render_examples(self, t, **kwargs):
        if t.examples is None or len(t.examples) == 0:
            return ''
        examples = [
            {
                'input': json.dumps(self.parse(i)),
                'output': json.dumps(self.parse(o)),
            }
            for i, o in random.sample(t.examples, t.num_examples)
        ]
        example_template = JinjaTemplate(t.example_template)
        return '\n'.join([
            example_template.render(**e) for e in examples
        ])
    
    def process(self, t, x, output, **kwargs):
        if t.output is None:
            return output
        out = json.loads(output)
        schema = model_to_json_schema(json.loads(t.output))
        if schema.get('type', None) == 'string' or (schema.get('type', None) == 'array' and schema.get('items', {}).get('type', None) == 'string'):
            return out
        entities = create_entity_from_schema(schema, out)
        return entities
    
    def dict(self):
        return {
            'id': self.id,
            'type': 'template',
            'name': self.name or None,
            'instructions': self.instructions,
            'input': self.input,
            'output': self.output,
        }
    
    def __call__(self, t, x, **kwargs):
        return self.forward(t, x, **kwargs)
    
    def forward(self, t, x, context=None, history=None, retries=3, dryrun=False, **kwargs):
        if retries and retries <= 0:
            e = MaxRetriesExceeded(f'{t.name} failed to forward {x}')
            self.logger.error(e)
            raise e
        
        if dryrun:
            self.logger.debug(f'Dryrun: {t.output}')
            llm = MockLLM(output=t.output)
        else:
            llm = self.llm
        
        px = self.parse(x)
            
        prompt_input = self.render(t, {'input': px})
        self.logger.log(INSTRUCTIONS, t.instructions)
        if t.examples: self.logger.log(EXAMPLES, self.render_examples(t))
        if len(px): self.logger.log(INPUT, px)

        try:
            response = llm.generate(prompt_input, context=context or t.context, history=history)
        except (urllib3.exceptions.ReadTimeoutError,
                urllib3.exceptions.ConnectTimeoutError,
                urllib3.exceptions.NewConnectionError) as e:
            self.logger.warning(f'LLM generation failed: {e}')
            time.sleep(2)
            return self.forward(t, x, retries=retries, **kwargs)
        except (openai.error.APIError,
                openai.error.Timeout,
                openai.error.ServiceUnavailableError) as e:
            self.logger.warning(f'LLM generation failed: {e}')
            time.sleep(2)
            return self.forward(t, x, retries=retries, **kwargs)
        except openai.error.RateLimitError as e:
            self.logger.warning(f'Hit rate limit for {self}: {e}')
            time.sleep(10)
            return self.forward(t, x, retries=retries, **kwargs)

        if t.output: self.logger.log(FORMAT_INSTRUCTIONS, self.render_format(t, px))
        try:
            self.logger.log(OUTPUT, response.raw)
            response.content = self.process(t, px, response.raw, **kwargs)
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f'Output validation failed: {e}')
            return self.forward(t, x, retries=retries-1, **kwargs)
        except json.JSONDecodeError as e:
            self.logger.warning(f'Failed to decode JSON from {e}')
            return self.forward(t, x, retries=retries-1, **kwargs)
        except Exception as e:
            self.logger.error(f'Failed to forward {x}: {e}')
            return self.forward(t, x, retries=retries-1, **kwargs)
        self.logger.log(METRICS, response.metrics)
        return response
