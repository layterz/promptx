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
    name: str
    age: int
    role: Role = Role.admin
    traits: List[Trait] = None
    friends: list['User'] = None
    status: str = Field(None, generate=False)

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
        raw='{ "name": "test", "age": 20 }',
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
        raw='{ "name": "test", "age": 20, "location": "london" }',
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
        raw='"name": "test", "age": 20 }',
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
    llm.generate.return_value = Response(raw='{ "name": "test", "age": 20 }')
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
    assert 'name (type: string, required: True, default: None)' in p

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

def test_format_rendering():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'name (type: string, required: True, default: None)' in p
    assert 'age (type: integer, required: True, default: None)' in p

def test_format_rendering_with_enum():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'role (type: string, required: False, default: admin)' in p
    assert 'Select one option from: admin, user' in p

def test_format_rendering_with_enum_list():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'traits (type: string[], required: False, default: None)' in p
    assert 'Select any relevant options from: nice, mean, funny, smart' in p

def test_format_rendering_with_excluded_fields():
    t = Template(instructions='Some example instructions', output=User.schema_json())
    runner = TemplateRunner()
    p = runner.render(t, {'input': 'Some test input'})
    assert 'status (type: string, required: False, default: None)' not in p

def test_example_rendering(mocker):
    assert True == False