import pytest

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

def test_exception_handling(mocker):
    llm = mocker.Mock(spec=LLM)
    llm.generate.side_effect = json.JSONDecodeError(msg='Invalid JSON', doc='', pos=0)
    
    t = Template()
    runner = TemplateRunner(llm=llm)
    
    with pytest.raises(json.JSONDecodeError):
        runner(t, 'test')