import pytest
from enum import Enum
from pydantic import BaseModel

from promptx.collection import *
from . import User, Account, _user, user, session


def test_objects_are_returned_with_correct_schema(session, mocker):
    user = User(name="test", age=20, traits=['nice', 'mean'])
    db = mocker.Mock(spec=VectorDB)
    db.name = 'test'
    db.get.return_value = {'ids': [user.id], 'documents': [user.model_dump_json()], 'metadatas': [{'schema': json.dumps(user.model_json_schema())}]}
    collection = Collection.load(session, db)
    collection.embed(user)

    assert len(collection.objects) == 1
    assert collection.first.name == "test"

def test_embedding_an_entity(session, mocker):
    user = User(name="test", age=20, traits=['nice', 'mean'])
    db = mocker.Mock(spec=VectorDB)
    db.name = 'test'
    db.get.return_value = {'ids': [user.id], 'documents': [user.model_dump_json()], 'metadatas': [{'schema': json.dumps(user.model_json_schema())}]}
    collection = Collection.load(session, db)
    collection.embed(user)

    assert len(collection.objects) == 1
    assert collection.first.name == "test"



def test_convert_pydantic_model_to_json_schema():
    schema = model_to_json_schema(User)
    expected_schema = {
        'title': 'User',
        'type': 'object',
        'properties': {
            'id': {'title': 'Id', 'type': 'string', 'default': None},
            'type': {'title': 'Type', 'type': 'string', 'default': None},
            'name': {'maxLength': 20, 'minLength': 3, 'title': 'Name', 'type': 'string'},
            'age': {'exclusiveMaximum': 100, 'minimum': 18, 'title': 'Age', 'type': 'integer'},
            'role': {'allOf': [{'$ref': '#/$defs/Role'}], 'default': 'admin'},
            'banned': {'default': False, 'generate': False, 'title': 'Banned', 'type': 'boolean'}, 
            'vigor': {'default': 0, 'maximum': 1.0, 'minimum': 0.0, 'title': 'Vigor', 'type': 'number'}, 
            'traits': {
                'title': 'Traits', 
                'type': 'array',
                'items': {'$ref': '#/$defs/Trait'}, 
                'maxItems': 3, 'minItems': 1, 
            },
        },
        '$defs': {
            'Role': {'enum': ['admin', 'user'], 'title': 'Role', 'type': 'string'}, 
            'Trait': {'enum': ['nice', 'mean', 'funny', 'smart'], 'title': 'Trait', 'type': 'string'},
        },
        'required': ['name', 'age', 'traits']
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
        '$defs': {},
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
                '$ref': '#/$defs/Address'
            }
        },
        'required': ['name', 'age', 'address'],
        '$defs': {
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


def test_convert_pydantic_model_with_nested_entities_to_json_schema():
    schema = model_to_json_schema(Account)
    expected_schema = {
        'title': 'Account',
        'type': 'object',
        'properties': { 'id': {'title': 'Id', 'type': 'string', 'default': None},
            'type': {'title': 'Type', 'type': 'string', 'default': None},
            'user': { '$ref': '#/$defs/Query' },
            'payees': { '$ref': '#/$defs/Query' },
        },
        '$defs': {
            'Query': {
                'type': 'object',
                'properties': {
                    'ids': { 'type': 'array', 'items': { 'type': 'string' } },
                    'query': { 'type': 'string' },
                    'collection': { 'type': 'string' },
                    'limit': { 'type': 'integer' },
                },
                'required': []
            }
        },
        'required': ['user'],
    }
    assert schema['$defs']['Query'] == expected_schema['$defs']['Query']


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
    Person = create_model_from_schema(schema, base=Entity)
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
    Person = create_model_from_schema(schema, base=Entity)
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
    Item = create_model_from_schema(schema, base=Entity)
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
    
    Person = create_model_from_schema(complex_schema, base=Entity)
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

def test_create_entity_from_schema_with_single_entity(session):
    data = {'name': 'Alice', 'age': 30}
    person = create_entity_from_schema(person_schema, data, session=session, base=Entity)
    assert isinstance(person, BaseModel)
    assert person.name == 'Alice'
    assert person.age == 30

def test_create_entity_from_schema_with_single_entity_missing_required_field(session):
    # Missing 'name' field, which is required
    data = {'age': 30}
    with pytest.raises(jsonschema.ValidationError):
        create_entity_from_schema(person_schema, data, session=session, base=Entity)

def test_create_entity_from_schema_with_single_entity_and_generated_id_and_type(session):
    data = {'name': 'Alice', 'age': 30}
    person = create_entity_from_schema(person_schema, data, session=session, base=Entity)
    assert hasattr(person, 'id')
    assert hasattr(person, 'type')
    assert person.type == 'person'

def test_create_entity_from_schema_with_list_of_entities(session):
    data_list = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    people_schema = {
        'type': 'array',
        'items': person_schema
    }
    people = create_entity_from_schema(people_schema, data_list, session=session, base=Entity)
    assert isinstance(people, list)
    assert len(people) == 2
    assert all(isinstance(person, BaseModel) for person in people)
    assert people[0].name == 'Alice'
    assert people[1].age == 25

def test_create_entity_from_schema_with_list_of_entities_and_generated_ids(session):
    data_list = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    people_schema = {
        'type': 'array',
        'items': person_schema
    }
    people = create_entity_from_schema(people_schema, data_list, session=session, base=Entity)
    assert all(hasattr(person, 'id') for person in people)  # 'id' should be generated for each entity
