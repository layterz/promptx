import uuid
from enum import Enum
from typing import *
import jsonschema
from pydantic import BaseModel, Field, ConfigDict, create_model
from IPython.display import display, HTML


JSON_TYPE_MAP: Dict[str, Type[Union[str, int, float, bool, Any]]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

PYTYPE_TO_JSONTYPE: Dict[Type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _is_list(schema):
    return schema.get('type') == 'array'


def _is_list_type(type_hint):
    origin = get_origin(type_hint)
    return origin is list or (origin is List and len(get_args(type_hint)) == 1)


def _get_title(schema):
    return schema.get('title', schema.get('items', {}).get('title', 'Entity'))


def _get_properties(schema):
    properties = schema.get('properties', {}) if not _is_list(schema) else schema.get('items', {}).get('properties', {})
    return properties


def _get_field_type(field_info, definitions):
    field_type = field_info.get('type')
    if field_type is None:
        ref = field_info.get('$ref')
        if ref is None:
            ref = field_info.get('allOf', [{}])[0].get('$ref')
        if ref is None:
            return str
        ref_name = ref.split('/')[-1]
        field_type = ref_name
        definition = definitions.get(ref_name, {})
        if 'enum' in definition:
            members = {v: v for v in definition['enum']}
            E = Enum(definition.get('title', ref_name), members)
            return E
        else:
            M = create_model_from_schema(definition)
            return M

    if field_type == 'array':
        info = field_info.get('items', {})
        return List[_get_field_type(info, definitions)]
    return JSON_TYPE_MAP[field_type]


def _create_field(field_info, definitions, required=False):
    field_type = _get_field_type(field_info, definitions)
    field_default = field_info.get('default', ... if required else None)
    return (field_type, field_default)


def model_to_json_schema(model):
    """
    Convert a Pydantic BaseModel or Python data type to a JSON schema.

    Args:
        model: A Pydantic BaseModel, a Python data type, a list of BaseModel instances, or a dictionary.

    Returns:
        dict: A JSON schema representation of the input model.

    This function takes various types of input and converts them into a JSON schema representation:

    - If `model` is a Pydantic BaseModel, it extracts its schema using `model.schema()`.

    - If `model` is a Python data type (e.g., str, int, float), it maps it to the corresponding JSON type.

    - If `model` is a list of Pydantic BaseModels, it generates a JSON schema for an array of the BaseModel's schema.

    - If `model` is a dictionary, it is returned as is.

    Example:
    >>> from pydantic import BaseModel
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    ...
    >>> schema = model_to_json_schema(Person)
    >>> print(schema)
    {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'}
        },
        'required': ['name']
    }
    """
    output = None
    if isinstance(model, list):
        inner = model[0]
        if issubclass(inner, BaseModel):
            schema = inner.model_json_schema()
            output = {
                'type': 'array',
                'items': schema,
                '$defs': schema.get('$defs', {})
            }
        else:
            output = {
                'type': 'array',
                'items': {
                    'type': PYTYPE_TO_JSONTYPE[inner]
                }
            }
    elif isinstance(model, dict):
        output = model
    elif isinstance(model, BaseModel):
        output = model.model_json_schema()
    elif isinstance(model, type):
        if issubclass(model, BaseModel):
            output = model.model_json_schema()
    
    return output


def create_model_from_schema(schema, base=None):
    """
    Create a Pydantic BaseModel from a JSON schema.

    Args:
        schema (dict): The JSON schema to create the Pydantic model from.

    Returns:
        pydantic.BaseModel: A Pydantic data model class generated from the schema.

    This function takes a JSON schema and generates a Pydantic BaseModel class
    with fields corresponding to the properties defined in the schema. It
    also handles definitions and required fields.

    If the schema doesn't specify a 'type' field, it defaults to 'Entity'.

    Example:
    >>> schema = {
    ...     'title': 'Person',
    ...     'type': 'object',
    ...     'properties': {
    ...         'name': {'type': 'string'},
    ...         'age': {'type': 'integer'}
    ...     },
    ...     'required': ['name']
    ... }
    >>> Person = create_model_from_schema(schema)
    >>> person = Person(name='Alice', age=30)
    >>> person.name
    'Alice'
    >>> person.age
    30
    """
    properties = _get_properties(schema)
    definitions = schema.get('$defs', {})
    required = schema.get('required', [])
    fields = {
        name: _create_field(field_info, definitions, name in required)
        for name, field_info in properties.items()
    }
    if 'id' not in fields:
        fields['id'] = (str, None)
    if 'type' not in fields:
        fields['type'] = (str, schema.get('title', 'Entity').lower())
    return create_model(schema.get('title', 'Entity').capitalize(), **fields, __base__=base)


def create_entity_from_schema(schema, data, base=None):
    """
    Create a Pydantic data entity from a JSON schema and input data.

    Args:
        schema (dict): The JSON schema that defines the structure of the entity.
        data (dict or list): The input data to populate the entity. For a single entity, provide a dictionary.
                             For a list of entities, provide a list of dictionaries.

    Returns:
        pydantic.BaseModel or List[pydantic.BaseModel]: A Pydantic data entity or a list of entities generated
                                                      from the schema and input data.

    This function takes a JSON schema and input data and creates a Pydantic data entity or a list of entities
    based on the schema and data provided. It handles properties, definitions, and optional fields defined
    in the schema.

    If the schema defines an entity as a list, the input data should be a list of dictionaries. Each dictionary
    represents an entity. If 'id' is not provided for each entity, it will be generated using a random UUID.

    If the schema defines an entity as an object (not a list), the input data should be a dictionary representing
    a single entity. If 'id' is not provided, it will be generated using a random UUID.

    Example:
    >>> schema = {
    ...     'title': 'Person',
    ...     'type': 'object',
    ...     'properties': {
    ...         'name': {'type': 'string'},
    ...         'age': {'type': 'integer'}
    ...     },
    ...     'required': ['name']
    ... }
    >>> data = {'name': 'Alice', 'age': 30}
    >>> person = create_entity_from_schema(schema, data)
    >>> person.name
    'Alice'
    >>> person.age
    30

    >>> schema_list = {
    ...     'title': 'People',
    ...     'type': 'array',
    ...     'items': {
    ...         'type': 'object',
    ...         'properties': {
    ...             'name': {'type': 'string'},
    ...             'age': {'type': 'integer'}
    ...         },
    ...         'required': ['name']
    ...     }
    ... }
    >>> data_list = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    >>> people = create_entity_from_schema(schema_list, data_list)
    >>> len(people)
    2
    >>> people[0].name
    'Alice'
    >>> people[1].age
    25
    """

    if _is_list(schema):
        data = [
            {**o, 'id': str(uuid.uuid4()) if o.get('id') is None else o['id']}
            for o in data
        ]
    else:
        if data.get('id') is None:
            data['id'] = str(uuid.uuid4())
    
    definitions = schema.get('definitions', {})
    for name, field in schema.get('properties', {}).items():
        _type = _get_field_type(field, definitions)
        if isinstance(_type, type) and issubclass(_type, Enum):
            if data.get(name):
                data[name] = data[name].name.lower()
        elif getattr(_type, '__origin__', None) == list and isinstance(_type.__args__[0], type) and issubclass(_type.__args__[0], Enum):
            if data.get(name):
                data[name] = [d.name.lower() for d in data[name]]
    
    # TODO: need to somehow handle nested entities which should be stored as IDs
    # currently the schema is correct in that it defines the desired type as a string
    # however, the entity needs to be loaded when the parent entity is loaded
    
    jsonschema.validate(data, schema)
    m = create_model_from_schema(schema, base=base)
    defaults = {
        'type': _get_title(schema).lower(),
    }
    if _is_list(schema):
        return [m.load(**{**defaults, **o}) for o in data]
    else:
        return m.load(**{**defaults, **data})


def serializer(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, BaseModel):
        return obj.model_json_schema()
    raise TypeError(f"Type {type(obj)} not serializable")