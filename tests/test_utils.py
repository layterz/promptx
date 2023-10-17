import pytest
from pydantic import BaseModel
from promptz.utils import model_to_json_schema  # Replace 'your_module' with the actual module where the function is defined


def test_convert_pydantic_model_to_json_schema():
    class Person(BaseModel):
        name: str
        age: int
    schema = model_to_json_schema(Person)
    expected_schema = {
        'title': 'Person',
        'type': 'object',
        'properties': {
            'name': {'title': 'Name', 'type': 'string'},
            'age': {'title': 'Age', 'type': 'integer'}
        },
        'required': ['name', 'age']
    }
    assert schema == expected_schema


def test_convert_list_of_pydantic_models_to_json_schema():
    class Item(BaseModel):
        name: str
        price: float
    schema = model_to_json_schema([Item])
    expected_schema = {
        'type': 'array',
        'items': {
            'title': 'Item',
            'type': 'object',
            'properties': {
                'name': {'title': 'Name', 'type': 'string'},
                'price': {'title': 'Price', 'type': 'number'}
            },
            'required': ['name', 'price']
        },
        'definitions': {},
    }
    assert schema == expected_schema


def test_passthrough_dictionary_as_json_schema():
    schema = {'type': 'object', 'properties': {'key': {'type': 'string'}}}
    result_schema = model_to_json_schema(schema)
    assert result_schema == schema