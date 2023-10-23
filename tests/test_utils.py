import pytest
from enum import Enum
from pydantic import BaseModel
from promptx.utils import *


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


def test_create_model_from_schema_with_required_fields():
    schema = {
        'title': 'Person',
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'}
        },
        'required': ['name']
    }
    Person = create_model_from_schema(schema)
    person = Person(name='Alice', age=30)
    assert isinstance(person, BaseModel)
    assert person.name == 'Alice'
    assert person.age == 30


def test_create_model_from_schema_with_missing_optional_fields():
    # Create a schema without the 'age' property, which is optional
    schema = {
        'title': 'Person',
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'}
        },
        'required': ['name']
    }
    Person = create_model_from_schema(schema)
    person = Person(name='Alice')
    assert isinstance(person, BaseModel)
    assert person.name == 'Alice'
    assert person.age is None


def test_create_model_from_schema_with_custom_type():
    # Create a schema with a custom 'type' field
    schema = {
        'title': 'Item',
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'price': {'type': 'number'}
        }
    }
    Item = create_model_from_schema(schema)
    item = Item(name='Product', price=10.99)
    assert isinstance(item, BaseModel)
    assert item.name == 'Product'
    assert item.price == 10.99


def test_create_model_from_schema_with_enum():
    # Example JSON schema with complex properties
    complex_schema = {
        'title': 'Person',
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'},
            'gender': {
                'type': 'string',
                'enum': ['Male', 'Female', 'Other']
            },
            'address': {
                'type': 'object',
                'properties': {
                    'street': {'type': 'string'},
                    'city': {'type': 'string'}
                }
            },
            'hobbies': {
                'type': 'array',
                'items': {'type': 'string'}
            }
        },
        'required': ['name', 'age']
    }
    
    Person = create_model_from_schema(complex_schema)
    person = Person(
        name='Alice',
        age=30,
        gender='female',
        address={'street': '123 Main St', 'city': 'City'},
        hobbies=['Reading', 'Cooking']
    )
    assert person.name == 'Alice'
    assert person.age == 30
    assert person.gender == 'female' 
    assert person.address['street'] == '123 Main St'
    assert person.address['city'] == 'City'
    assert person.hobbies == ['Reading', 'Cooking']


# Example JSON schema for testing
person_schema = {
    'title': 'Person',
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'integer'}
    },
    'required': ['name']
}

def test_create_entity_from_schema_with_single_entity():
    data = {'name': 'Alice', 'age': 30}
    person = create_entity_from_schema(person_schema, data)
    assert isinstance(person, BaseModel)
    assert person.name == 'Alice'
    assert person.age == 30

def test_create_entity_from_schema_with_single_entity_missing_required_field():
    # Missing 'name' field, which is required
    data = {'age': 30}
    with pytest.raises(jsonschema.ValidationError):
        create_entity_from_schema(person_schema, data)

def test_create_entity_from_schema_with_single_entity_and_generated_id():
    data = {'name': 'Alice', 'age': 30}
    person = create_entity_from_schema(person_schema, data)
    assert hasattr(person, 'id')  # 'id' should be generated

def test_create_entity_from_schema_with_list_of_entities():
    data_list = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    people_schema = {
        'type': 'array',
        'items': person_schema
    }
    people = create_entity_from_schema(people_schema, data_list)
    assert isinstance(people, list)
    assert len(people) == 2
    assert all(isinstance(person, BaseModel) for person in people)
    assert people[0].name == 'Alice'
    assert people[1].age == 25

def test_create_entity_from_schema_with_list_of_entities_and_generated_ids():
    data_list = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    people_schema = {
        'type': 'array',
        'items': person_schema
    }
    people = create_entity_from_schema(people_schema, data_list)
    assert all(hasattr(person, 'id') for person in people)  # 'id' should be generated for each entity
