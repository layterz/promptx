import uuid
from enum import Enum
from typing import Any, Dict, List, Type, Union, get_origin, get_args
import jsonschema
from pydantic import BaseModel, create_model
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


class Entity(BaseModel):
    id: str = None
    type: str = None

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
    
    def __init__(self, id=None, **data):
        if 'type' not in data:
            data['type'] = self.__class__.__name__.lower()
        super().__init__(**{'id': id or str(uuid.uuid4()), **data})
    
    def generate_schema_for_field(self, name, field_type: Any, default=False):
        return_list = False
        definitions = {}
        
        if isinstance(field_type, list):
            field_type = field_type[0]
            return_list = True
            
        # Handle basic types
        if isinstance(field_type, type) and issubclass(field_type, (int, float, str, bool)):
            type_ = PYTYPE_TO_JSONTYPE[field_type]
            schema = {"type": type_}

        # Handle Pydantic model types (reference schema)
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # If the schema for this model hasn't been generated before
            for name, info in field_type.__fields__.items():
                if name not in definitions:
                    field, defs = self.generate_schema_for_field(name, info.type_, info.default)
                    definitions[name] = {
                        "type": "object",
                        "properties": field,
                    }
                    definitions = {**definitions, **defs}
            schema = {
                "type": "object",
                "$ref": f"#/definitions/{field_type.__name__}",
            }
        
        # Handle default case by getting the cls field and calling schema
        else:
            if isinstance(field_type, list):
                field_type = field_type[0]
            if field_type.__name__ not in definitions:
                definitions[field_type.__name__] = {
                    "type": "object",
                    "properties": default.schema() if default is not None else {}
                }
            schema = {"$ref": f"#/definitions/{field_type.__name__}"} 

        if return_list:
            schema = {
                "type": "array",
                "items": schema
            } 

        if default:
            schema['default'] = default
        return schema, definitions
    
    def schema(self, by_alias: bool = True, **kwargs):
        properties = {}
        required = []
        definitions = {}

        for field_name, field_info in self.__fields__.items():
            try:
                field, defs = self.generate_schema_for_field(field_name, field_info.type_, field_info.default)
                properties[field_name] = field
                definitions = {**definitions, **defs}
            except Exception as e:
                print('schema field failed', field_name, e)
            
            if field_info.required:
                required.append(field_name)

        # Construct the base schema
        base_schema = {
            "title": self.type or self.__class__.__name__,
            "type": "object",
            "properties": properties,
            "definitions": definitions,  # Include definitions for references
            "required": required,
        }

        return base_schema
    
    def display(self):
        # Check if we're in an IPython environment
        try:
            get_ipython
        except NameError:
            # If we're not in an IPython environment, fall back to json
            return self.json()

        # Convert the dictionary to a HTML table
        html = '<table>'
        for field, value in self.dict().items():
            html += f'<tr><td>{field}</td><td>{value}</td></tr>'
        html += '</table>'

        # Display the table
        display(HTML(html))
        return self.json()


def model_to_json_schema(model):
    output = None
    if isinstance(model, list):
        inner = model[0]
        if issubclass(inner, BaseModel):
            schema = inner.schema()
            output = {
                'type': 'array',
                'items': schema,
                'definitions': schema.get('definitions', {})
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
        output = model.schema()
    elif isinstance(model, type):
        if issubclass(model, BaseModel):
            output = model.schema()
    
    return output


def _is_list(schema):
    return schema.get('type') == 'array'


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
        definition = definitions.get(ref_name)
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


def create_model_from_schema(schema):
    properties = _get_properties(schema)
    definitions = schema.get('definitions', {})
    required = schema.get('required', [])
    fields = {
        name: _create_field(field_info, definitions, name in required)
        for name, field_info in properties.items()
    }
    if 'type' not in fields:
        fields['type'] = (str, ...)
    return create_model(schema.get('title', 'Entity'), **fields, __base__=Entity)


def create_entity_from_schema(schema, data):
    if _is_list(schema):
        data = [
            {**o, 'id': str(uuid.uuid4()) if o.get('id') is None else o['id']}
            for o in data
        ]
    else:
        if data.get('id') is None:
            data['id'] = str(uuid.uuid4())
    
    definitions = schema.get('definitions', {})
    for name, field in schema.get('raw_inpuiproperties', {}).items():
        _type = _get_field_type(field, definitions)
        if isinstance(_type, type) and issubclass(_type, Enum):
            if data.get(name):
                data[name] = data[name].lower()
        elif getattr(_type, '__origin__', None) == list and isinstance(_type.__args__[0], type) and issubclass(_type.__args__[0], Enum):
            if data.get(name):
                data[name] = [d.lower() for d in data[name]]
    
    jsonschema.validate(data, schema)
    m = create_model_from_schema(schema)
    defaults = {
        'type': _get_title(schema),
    }
    if _is_list(schema):
        return [m(**{**defaults, **o}) for o in data]
    else:
        return m(**{**defaults, **data})
