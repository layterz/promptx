import pytest
import openai
import json
from enum import Enum
from pydantic import Field

from promptx.template import *
from promptx.models import LLM, Response
from promptx.utils import Entity


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
    traits: List[Trait] = Field(..., description='What kind of personality describes the user?', min_length=1, max_length=3)
    banned: bool = Field(None, json_schema_extra={'generate': False})
    vigor: float = Field(0, ge=0, le=1)

class Account(Entity):
    user: User

@pytest.fixture
def template():
    t = Template(instructions='Some example instructions', output=json.dumps(User.model_json_schema()))
    return t


def test_basic_response(mocker):
    template = Template(instructions='Some example instructions')
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='Test response',
    )
    llm.generate.return_value = response
    runner = TemplateRunner(llm=llm)
    o = runner(template, None)

    assert o.content == 'Test response'

def test_json_valid_output(mocker, template):
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='{ "name": "test", "age": 20, "traits": ["nice"] }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)
    o = runner(template, None)

    assert o.content.type == 'User'
    assert o.content.name == 'test'
    assert o.content.age == 20
    
def test_json_valid_output__extra_field(mocker, template):
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='{ "name": "test", "age": 20, "location": "london", "traits": ["nice"] }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)

    o = runner(template, None)
    assert o.content.name == 'test'
    assert o.content.age == 20
    with pytest.raises(AttributeError):
        assert o.content.location == 'london'
    
def test_json_invalid_output__missing_required_field(mocker, template):
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='{ "age": 20 }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        runner(template, None)
    
def test_json_invalid_output__formatting(mocker, template):
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='"name": "test", "age": 20, "traits": ["nice"] }',
    )
    llm.generate.return_value = response
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        runner(template, None)
    
def test_invalild_json_output__validation(mocker, template):
    llm = mocker.Mock(spec=LLM)
    response = Response(
        raw='{ "name": "test", "age": "young" }',
    )
    llm.generate.return_value = response
    
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        runner(template, None)

# TODO: this should probably have some kind of separate retry budget
def test_exception_handling(mocker, template):
    llm = mocker.Mock(spec=LLM)
    llm.generate.side_effect = [openai.error.Timeout, Response(raw='Test response')]
    template = Template(instructions='Some example instructions')
    
    runner = TemplateRunner(llm=llm)
    o = runner(template, None)
    assert o.content == 'Test response'

def test_parse_exception_handling(mocker, template):
    llm = mocker.Mock(spec=LLM)
    mocker.patch.object(TemplateRunner, 'process', side_effect=[*[json.JSONDecodeError('test', 'test', 0)] * 4, 'test'])
    runner = TemplateRunner(llm=llm)

    with pytest.raises(MaxRetriesExceeded):
        runner(template, None)
    
    mocker.patch.object(TemplateRunner, 'process', side_effect=[*[json.JSONDecodeError('test', 'test', 0)] * 3, 'test'])
    runner = TemplateRunner(llm=llm)
    o = runner(template, None)
    
    assert o.content == 'test'

def test_invalid_input_raises_error(template):
    runner = TemplateRunner()
    with pytest.raises(MaxRetriesExceeded):
        runner(template, {'age': 'young'})

def test_output_parsing(mocker, template):
    llm = mocker.Mock(spec=LLM)
    llm.generate.return_value = Response(raw='{ "name": "test", "age": 20, "traits": ["nice"] }')
    runner = TemplateRunner(llm=llm)

    o = runner(template, None)
    assert o.content.type == 'User'
    assert o.content.name == 'test'
    assert o.content.age == 20

def test_format_rendering(template):
    runner = TemplateRunner()
    p = runner.render(template, {})
    assert template.instructions in p

def test_format_rendering_with_input(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'Some test input' in p

def test_format_rendering_with_output(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'name (type: string, required: True, default: None' in p

def test_format_rendering_object(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
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

def test_format_rendering_with_basic_types(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'name (type: string, required: True, default: None' in p
    assert 'age (type: integer, required: True, default: None' in p

def test_format_rendering_with_enum(template):
    t = Template(instructions='Some example instructions', output=json.dumps(User.model_json_schema()))
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'role (type: string, required: False, default: admin' in p
    assert 'Select one option from: admin, user' in p

def test_format_rendering_with_enum_list(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'traits (type: string[], required: True, default: None' in p
    assert 'Select any relevant options from: nice, mean, funny, smart' in p

def test_format_rendering_with_excluded_fields(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'banned (type: bool, required: False, default: False' not in p

def test_format_rendering_with_field_description(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'What kind of personality describes the user?' in p

def test_format_rendering_with_field_min_max(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'ge: 18' in p
    assert 'lt: 100' in p

def test_format_rendering_with_field_min_max_items(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'min_length: 1' in p
    assert 'max_length: 3' in p

def test_format_rendering_with_field_min_max_length(template):
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})
    assert 'min_length: 3' in p
    assert 'max_length: 20' in p

def test_example_rendering(template):
    user = User(name="John Wayne", age=64, traits=[Trait.mean])
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})

    assert 'EXAMPLES' in p
    assert 'John Wayne' in p
    assert '64' in p
    assert 'mean' in p
    assert 'banned' not in p

def test_example_rendering_multiple(template):
    user = User(name="John Wayne", age=64, traits=[Trait.mean])
    runner = TemplateRunner()
    p = runner.render(template, {'input': 'Some test input'})

    assert p.count('John Wayne') == 3