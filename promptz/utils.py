
def model_to_json_schema(model):
    if model is None:
        return None
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