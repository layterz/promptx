from typing import Any, Dict, Type, Union
import jsonschema
from pydantic import create_model


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


def _get_properties(schema):
    properties = schema.get('properties', {}) if not _is_list(schema) else schema.get('items', {}).get('properties', {})
    return properties


def create_model_from_schema(schema):
    properties = _get_properties(schema)
    fields = {
        name: (JSON_TYPE_MAP[field_info["type"]], ... if "default" not in field_info else field_info["default"])
        for name, field_info in properties.items()
    }
    if 'type' not in fields:
        fields['type'] = (str, ...)
    
    return create_model(schema.get('title', 'Entity'), **fields)


def create_entity_from_schema(schema, data):
    jsonschema.validate(data, schema)
    properties = _get_properties(schema)
    m = create_model_from_schema(schema)
    type_ = properties.get('title', 'Entity').lower()
    if _is_list(schema):
        return [m(**{'type': type_, **o}) for o in data]
    else:
        return m(**{'type': type_, **data})