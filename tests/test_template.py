import pytest
import openai
import json

from promptz.template import *
from promptz.models import LLM, Response
from promptz.utils import Entity


class User(Entity):
    name: str
    age: int


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

def test_json_output(mocker):
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

# TODO: this should probably have some kind of separate retry budget
def test_exception_handling(mocker):
    llm = mocker.Mock(spec=LLM)
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
