import pytest
import openai
import json
from enum import Enum
from pydantic import Field

from promptz.template import *
from promptz.models import LLM, Response
from promptz.utils import Entity


class Role(Enum):
    admin = 'admin'
    user = 'user'

class Trait(Enum):
    nice = 'nice'
    mean = 'mean'
    funny = 'funny'
    smart = 'smart'

class User(Entity):
    name: str = Field(..., min_length=3, max_length=20)
    age: int = Field(..., ge=18, lt=100)
    role: Role = Role.admin
    traits: List[Trait] = Field(..., description='What kind of personality describes the user?', min_items=1, max_items=3)
    banned: bool = Field(None, generate=False)
    vigor: float = Field(0, max=1, min=0)

class Account(Entity):
    user: User


def test_basic_response(mocker):
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='Test response',
    )
    llm.generate.return_value = response
    
    t = Template()
    runner = TemplateRunner(llm=llm)
    o = runner(t, 'test')

    assert o.content == 'Test response'

def test_json_valid_output(mocker):
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='{ "name": "test", "age": 20, "traits": ["nice"] }',
    )
    llm.generate.return_value = response
    
    t = Template(output=User.schema_json())
    runner = TemplateRunner(llm=llm)
    o = runner(t, None)

    assert o.content.type == 'User'
    assert o.content.name == 'test'
    assert o.content.age == 20
    
def test_json_valid_output__extra_field(mocker):
    llm = mocker.Mock(spec=LLM)
    t = Template(output=User.schema_json())
    response = Response(
        raw='{ "name": "test", "age": 20, "location": "london", "traits": ["nice"] }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)

    o = runner(t, None)
    assert o.content.name == 'test'
    assert o.content.age == 20
    # TODO: should this fail?
    assert o.content.location == 'london'
    
def test_json_invalid_output__missing_required_field(mocker):
    llm = mocker.Mock(spec=LLM)
    t = Template(output=User.schema_json())
    response = Response(
        raw='{ "age": 20 }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        o = runner(t, None)
    
def test_json_invalid_output__formatting(mocker):
    llm = mocker.Mock(spec=LLM)
    t = Template(output=User.schema_json())
    response = Response(
        raw='"name": "test", "age": 20, "traits": ["nice"] }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        o = runner(t, None)
    
def test_invalild_json_output__validation(mocker):
    llm = mocker.Mock(spec=LLM)
    t = Template(output=User.schema_json())
    response = Response(
        raw='{ "name": "test", "age": "young" }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        o = runner(t, None)

# TODO: this should probably have some kind of separate retry budget
def test_exception_handling(mocker):
    llm = mocker.Mock(spec=LLM)
    t = Template(output=User.schema_json())
    llm.generate.side_effect = [openai.error.Timeout, Response(raw='Test response')]
    
    t = Template()
    runner = TemplateRunner(llm=llm)
    o = runner(t, 'test')
    assert o.content == 'Test response'

def test_parse_exception_handling(mocker):
    llm = mocker.Mock(spec=LLM)
    t = Template()
    mocker.patch.object(TemplateRunner, 'process', side_effect=[*[json.JSONDecodeError('test', 'test', 0)] * 4, 'test'])
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        runner(t, None)
    
    mocker.patch.object(TemplateRunner, 'process', side_effect=[*[json.JSONDecodeError('test', 'test', 0)] * 3, 'test'])
    runner = TemplateRunner(llm=llm)
    o = runner(t, None)
    
    assert o.content == 'test'

def test_invalid_input_raises_error():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    with pytest.raises(MaxRetriesExceeded):
        runner(t, {'age': 'young'})

def test_output_parsing(mocker):
    llm = mocker.Mock(spec=LLM)
    llm.generate.return_value = Response(raw='{ "name": "test", "age": 20, "traits": ["nice"] }')
    t = Template(output=User.schema_json())
    runner = TemplateRunner(llm=llm)

    o = runner(t, None)
    assert o.content.type == 'User'
    assert o.content.name == 'test'
    assert o.content.age == 20

def test_format_rendering():
    t = Template(instructions='Some example instructions')
    runner = TemplateRunner()
    p = runner.render(t, {})
    assert t.instructions in p

def test_format_rendering_with_input():
    t = Template(instructions='Some example instructions')
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'Some test input' in p

def test_format_rendering_with_output():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'name (type: string, required: True, default: None' in p

def test_format_rendering_object():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'Return the output as a valid JSON object with the fields described below' in p

def test_format_rendering_list():
    schema = json.dumps({
        'type': 'array',
        'items': {}
    })
    t = Template(instructions='Some example instructions', output=schema)
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'Return a list of valid JSON objects with the fields described below' in p

def test_format_rendering_with_basic_types():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'name (type: string, required: True, default: None' in p
    assert 'age (type: integer, required: True, default: None' in p

def test_format_rendering_with_enum():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'role (type: string, required: False, default: admin' in p
    assert 'Select one option from: admin, user' in p

def test_format_rendering_with_enum_list():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'traits (type: string[], required: True, default: None' in p
    assert 'Select any relevant options from: nice, mean, funny, smart' in p

def test_format_rendering_with_excluded_fields():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'banned (type: bool, required: False, default: False' not in p

def test_format_rendering_with_field_description():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'What kind of personality describes the user?' in p

def test_format_rendering_with_field_min_max():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'ge: 18' in p
    assert 'lt: 100' in p

def test_format_rendering_with_field_min_max_items():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'min_items: 1' in p
    assert 'max_items: 3' in p

def test_format_rendering_with_field_min_max_length():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'min_length: 3' in p
    assert 'max_length: 20' in p

def test_example_rendering(mocker):
    user = User(name="John Wayne", age=64, traits=[Trait.mean])
    t = Template(instructions='Some example instructions', output=User.schema_json(), examples=[(None, user)])
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})

    assert 'EXAMPLES' in p
    assert 'John Wayne' in p
    assert '64' in p
    assert 'mean' in p
    assert 'banned' not in p

def test_example_rendering_multiple(mocker):
    user = User(name="John Wayne", age=64, traits=[Trait.mean])
    t = Template(instructions='Some example instructions', output=User.schema_json(), examples=[(None, user)] * 5, num_examples=3)
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})

    assert p.count('John Wayne') == 3