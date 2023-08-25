import uuid
from typing import Any, Dict, Type, Union
import jsonschema
from pydantic import BaseModel, create_model
from IPython.display import display, HTML

class Entity(BaseModel):
    id: str = None

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        for field, value in data.items():
            # if value is NaN, set it to None
            if isinstance(value, float) and pd.isna(value):
                data[field] = None

        super().__init__(**data)
    
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
    if model is None:
        return None
    if isinstance(model, dict):
        return model
    if getattr(model, '_name', None) == 'List':
        inner = model.__args__[0]
        schema = inner.schema()
        output = {
            'type': 'array',
            'items': schema,
            'definitions': schema.get('definitions', {})
        }
    else:
        output = model.schema()
    
    return output


JSON_TYPE_MAP: Dict[str, Type[Union[str, int, float, bool, Any]]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


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
        print('field_info', field_info)
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
    print('create_model_from_schema', properties, schema)
    # TODO handle nested objects - should look up def and recurse
    fields = {
        name: _create_field(field_info, definitions)
        for name, field_info in properties.items()
    }
    if 'type' not in fields:
        fields['type'] = (str, ...)
    
    return create_model(schema.get('title', 'Entity'), **fields, __base__=Entity)


def create_entity_from_schema(schema, data):
    print('create_entity_from_schema', schema, data)
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
    print('create_entity_from_data', data)
    schema = ''
    return create_entity_from_schema(schema, data)