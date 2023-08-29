import uuid
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

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
    
    @classmethod
    def generate_schema_for_field(cls, field_type: Any):
        definitions = {}
        # Handle basic types
        print('field_type', field_type)
        if isinstance(field_type, type) and issubclass(field_type, (int, float, str, bool)):
            return {"type": PYTYPE_TO_JSONTYPE[field_type]}, definitions
        
        # Handle case for List[Type]
        elif get_origin(field_type) is List:
            type_inside = get_args(field_type)[0]
            return {
                "type": "array",
                "items": cls.generate_schema_for_field(type_inside)
            }, definitions

        # Handle Pydantic model types (reference schema)
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # If the schema for this model hasn't been generated before
            if field_type.__name__ not in definitions:
                definitions[field_type.__name__] = {
                    "type": "object",
                    "properties": {name: cls.generate_schema_for_field(info.type_) for name, info in field_type.__fields__.items()}
                }
            return {"$ref": f"#/definitions/{field_type.__name__}"}, definitions
        
        # Default case (for unhandled types)
        return {}, definitions
    
    @classmethod
    def schema(cls, by_alias: bool = True):
        properties = {}
        required = []
        definitions = {}

        for field_name, field_info in cls.__fields__.items():
            try:
                print('field', field_name)
                properties[field_name], defs = cls.generate_schema_for_field(field_info.type_)
                definitions = {**definitions, **defs}
            except Exception as e:
                print('schema field failed', field_name, e)
            
            # Check if the field is required
            if field_info.default == ...:
                required.append(field_name)

        # Construct the base schema
        base_schema = {
            "type": "object",
            "properties": properties,
            "definitions": definitions  # Include definitions for references
        }

        if required:
            base_schema["required"] = required

        return base_schema
    
    def __repr__(self):
        return self.json()
    
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
        schema = inner.schema()
        output = {
            'type': 'array',
            'items': schema,
            'definitions': schema.get('definitions', {})
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
        ref = field_info.get('allOf', [{}])[0].get('$ref')
        ref_name = ref.split('/')[-1]
        field_type = ref_name
        definition = definitions.get(ref_name)
        m = create_model_from_schema(definition)
        return m

    if field_type == 'array':
        field_type = field_info.get('items', {}).get('type')
    return JSON_TYPE_MAP[field_type]


def _create_field(field_info, definitions):
    field_type = _get_field_type(field_info, definitions)
    field_default = field_info.get('default', ...)
    return (field_type, field_default)


def create_model_from_schema(schema):
    properties = _get_properties(schema)
    definitions = schema.get('definitions', {})
    fields = {
        name: _create_field(field_info, definitions)
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
    jsonschema.validate(data, schema)
    m = create_model_from_schema(schema)
    defaults = {
        'type': _get_title(schema),
    }
    if _is_list(schema):
        return [m(**{**defaults, **o}) for o in data]
    else:
        return m(**{**defaults, **data})


def create_entity_from_data(data):
    schema = ''
    return create_entity_from_schema(schema, data)