import pytest
from enum import Enum
from pydantic import BaseModel
from promptz.utils import model_to_json_schema  # Replace 'your_module' with the actual module where the function is defined


class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    age: int
    address: Address

class Item(BaseModel):
    name: str
    color: Color

class Order(BaseModel):
    order_id: str
    person: Person
    items: list[Item]


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


def test_convert_pydantic_model_with_nested_models_to_json_schema():
    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        age: int
        address: Address

    schema = model_to_json_schema(Person)
    expected_schema = {
        'type': 'object',
        'title': 'Person',
        'properties': {
            'name': {'title': 'Name', 'type': 'string'},
            'age': {'title': 'Age', 'type': 'integer'},
            'address': {
                '$ref': '#/definitions/Address'
            }
        },
        'required': ['name', 'age', 'address'],
        'definitions': {
            'Address': {
                'type': 'object',
                'title': 'Address',
                'properties': {
                    'street': {'title': 'Street', 'type': 'string'},
                    'city': {'title': 'City', 'type': 'string'}
                },
                'required': ['street', 'city']
            }
        }
    }
    assert schema == expected_schema


def test_convert_pydantic_enum_to_json_schema():

    schema = model_to_json_schema(Item)
    expected_schema = {
        'type': 'object',
        'title': 'Item',
        'properties': {
            'name': {
                'title': 'Name',
                'type': 'string'
            },
            'color': {
                '$ref': '#/definitions/Color'
            }
        },
        'required': ['name', 'color'],
        'definitions': {
            'Color': {
                'title': 'Color',
                'description': 'An enumeration.',
                'enum': ['red', 'green', 'blue']
            }
        }
    }
    assert schema == expected_schema


def test_passthrough_dictionary_with_nested_schema_as_json_schema():
    schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'address': {
                'type': 'object',
                'properties': {
                    'street': {'type': 'string'},
                    'city': {'type': 'string'}
                },
                'required': ['street', 'city']
            }
        },
        'required': ['name'],
    }
    result_schema = model_to_json_schema(schema)
    assert result_schema == schema


def test_convert_pydantic_model_with_nested_list_of_enums_to_json_schema():
    class Item(BaseModel):
        name: str
        colors: list[Color]
    
    schema = model_to_json_schema(Item)
    expected_schema = {
        'type': 'object',
        'title': 'Item',
        'properties': {
            'name': {
                'title': 'Name',
                'type': 'string'
            },
            'colors': {
                'type': 'array',
                'items': {
                    '$ref': '#/definitions/Color'
                }
            }
        },
        'required': ['name', 'colors'],
        'definitions': {
            'Color': {
                'title': 'Color',
                'description': 'An enumeration.',
                'enum': ['red', 'green', 'blue']
            }
        }
    }
    assert schema == expected_schema


def test_convert_pydantic_model_with_nested_base_model_to_json_schema():
    schema = model_to_json_schema(Person)
    expected_schema = {
        'title': 'Person',
        'type': 'object',
        'properties': {
            'name': {'title': 'Name', 'type': 'string'},
            'age': {'title': 'Age', 'type': 'integer'},
            'address': {
                '$ref': '#/definitions/Address'
            }
        },
        'required': ['name', 'age', 'address'],
        'definitions': {
            'Address': {
                'title': 'Address',
                'type': 'object',
                'properties': {
                    'street': {'title': 'Street', 'type': 'string'},
                    'city': {'title': 'City', 'type': 'string'}
                },
                'required': ['street', 'city']
            }
        }
    }
    assert schema == expected_schema

def test_convert_pydantic_model_with_nested_list_of_base_models_to_json_schema():
    schema = model_to_json_schema(Order)
    expected_schema = {
        'title': 'Order',
        'type': 'object',
        'properties': {
            'order_id': {'title': 'Order Id', 'type': 'string'},
            'person': {
                '$ref': '#/definitions/Person'
            },
            'items': {
                'title': 'Items',
                'type': 'array',
                'items': {
                    '$ref': '#/definitions/Item'
                }
            }
        },
        'required': ['order_id', 'person', 'items'],
        'definitions': {
            'Person': {
                'title': 'Person',
                'type': 'object',
                'properties': {
                    'name': {'title': 'Name', 'type': 'string'},
                    'age': {'title': 'Age', 'type': 'integer'},
                    'address': {
                        '$ref': '#/definitions/Address'
                    },
                },
                'required': ['name', 'age', 'address']
            },
            'Address': {
                'title': 'Address',
                'type': 'object',
                'properties': {
                    'street': {'title': 'Street', 'type': 'string'},
                    'city': {'title': 'City', 'type': 'string'}
                },
                'required': ['street', 'city']
            },
            'Color': {
                'title': 'Color',
                'description': 'An enumeration.',
                'enum': ['red', 'green', 'blue']
            },
            'Item': {
                'title': 'Item',
                'type': 'object',
                'properties': {
                    'name': {'title': 'Name', 'type': 'string'},
                    'color': {
                        '$ref': '#/definitions/Color'
                    }
                },
                'required': ['name', 'color']
            }
        }
    }
    assert schema == expected_schema