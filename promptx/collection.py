import uuid
import json
from enum import Enum
from datetime import datetime
from abc import abstractmethod
from typing import *
from loguru import logger
import jsonschema
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, create_model
from pydantic_core._pydantic_core import PydanticUndefinedType
from IPython.display import display, HTML
import chromadb


REGISTERED_ENTITIES = {}

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
        if issubclass(model, Entity):
            output = model.model_json_schema()

            for name, field in model.__annotations__.items():
                if isinstance(field, type) and issubclass(field, Entity):
                    output['properties'][name] = {'$ref': '#/$defs/Query'}
                    output['$defs']['Query'] = {
                        'type': 'object',
                        'properties': {
                            'ids': {'type': 'array', 'items': {'type': 'string'}},
                            'query': { 'type': 'string' },
                            'collection': { 'type': 'string' },
                            'limit': { 'type': 'integer' },
                        },
                        'required': [],
                    }
                elif getattr(field, '__origin__', None) == list and isinstance(field.__args__[0], type) and issubclass(field.__args__[0], Entity):
                    output['properties'][name] = {'$ref': '#/$defs/Query'}
                    output['$defs']['Query'] = {
                        'type': 'object',
                        'properties': {
                            'ids': {'type': 'array', 'items': {'type': 'string'}},
                            'query': { 'type': 'string' },
                            'collection': { 'type': 'string' },
                            'limit': { 'type': 'integer' },
                        },
                        'required': [],
                    }
        elif issubclass(model, BaseModel):
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


def create_entity_from_schema(schema, data, session=None, base=None):
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
    >>> person = create_entity_from_schema(session, schema, data)
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
    >>> people = create_entity_from_schema(session, schema_list, data_list)
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
    
    def _field_serializer(obj):
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, 'model_dump'):
            return {k: v for k, v in obj.model_dump().items() if v is not None}
        elif isinstance(obj, list):
            if all(hasattr(o, 'model_dump') for o in obj):
                return [{k: v for k, v in e.model_dump().items() if v is not None} for e in obj]
        return obj
    
    if _is_list(schema):
        data = [{k: _field_serializer(v) for k, v in item.items()} for item in data]
    else:
        data = {k: _field_serializer(v) for k, v in data.items()}

    jsonschema.validate(data, schema)
    m = create_model_from_schema(schema, base=base)
    defaults = {
        'type': _get_title(schema).lower(),
    }
    if _is_list(schema):
        return [m.load(session=session, **{**defaults, **o}) for o in data]
    else:
        return m.load(session=session, **{**defaults, **data})


def serializer(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, BaseModel):
        return obj.model_json_schema()
    raise TypeError(f"Type {type(obj)} not serializable")


class Entity(BaseModel):
    id: str = None
    type: str = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        if 'type' not in data:
            data['type'] = self.__class__.__name__.lower()
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())

        for k, v in data.items():
            if isinstance(v, Enum):
                data[k] = v.value
            elif isinstance(v, list) and all(isinstance(e, Enum) for e in v):
                data[k] = [e.value for e in v]
        
        super().__init__(**data)
    
    @classmethod
    def load(cls, session=None, **kwargs):
        for name, field in cls.__annotations__.items():
            if field.__name__ == 'Entity':
                def loader(self, cls=cls, session=session, name=name, field=field, data=kwargs):
                    # Lazy-loading logic here
                    logger.info(f'Loading {name}')
                    field_data = data.get(name)
                    if field_data is None:
                        return None
                    limit = field_data.get('limit')
                    collection = field_data.get('collection')
                    response = session.query(ids=field_data.get('ids'), limit=limit, collection=collection)
                    if response is None:
                        return None
                    if limit is None or limit > 1:
                        return response.objects
                    elif limit == 1:
                        return response.first
                    return None
                
                setattr(cls, name, property(loader))
            
        return cls(**kwargs)
    
    @classmethod
    def generate_schema_for_field(cls, name, field_type: Any, field: Field):
        return_list = False
        definitions = {}

        if _is_list_type(field_type):
            field_type = get_args(field_type)[0]
            return_list = True
        
        # Handle enums
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            schema = {
                "type": "string",
                "enum": [e.name.lower() for e in field_type],
            }

        # Handle basic types
        elif isinstance(field_type, type) and issubclass(field_type, (int, float, str, bool)):
            type_ = PYTYPE_TO_JSONTYPE[field_type]
            schema = {"type": type_}

        # Handle Pydantic model types (reference schema)
        elif isinstance(field_type, type) and issubclass(field_type, Entity):
            schema = {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                },
            }
        
        # Handle default case by getting the cls field and calling schema
        else:
            if isinstance(field_type, list):
                field_type = field_type[0]
            if field_type.__name__ not in definitions:
                definitions[field_type.__name__] = {
                    "type": "object",
                    "properties": field.default.schema() if field.default is not None else {}
                }
            schema = {"$ref": f"#/$defs/{field_type.__name__}"} 

        if return_list:
            schema = {
                "type": "array",
                "items": schema,
            }
        
        # TODO: need to test/fix this
        info = None
        if info is not None:
            if info.description:
                schema['description'] = info.description
            if info.ge:
                schema['ge'] = info.ge
            if info.gt:
                schema['gt'] = info.gt
            if info.le:
                schema['le'] = info.le
            if info.lt:
                schema['lt'] = info.lt
            if info.min_length:
                schema['min_length'] = info.min_length
            if info.max_length:
                schema['max_length'] = info.max_length
            
            extra = info.extra
            if extra is not None:
                if 'generate' in extra:
                    schema['generate'] = extra['generate']

        if field.default:
            schema['default'] = field.default
        return schema, definitions, []
    
    @classmethod
    def __schema(cls, by_alias: bool = True, **kwargs):
        properties = {}
        required = []
        definitions = {}

        for field_name, field in cls.model_fields.items():
            try:
                type_ = cls.__annotations__.get(field_name, field.annotation)
                field_schema, defs, reqs = cls.generate_schema_for_field(field_name, type_, field)
                properties[field_name] = field_schema
                definitions = {**definitions, **defs}
            except Exception as e:
                logger.error('schema field failed', field_name, e, field)
                continue
            
            if field.is_required():
                required.append(field_name)
            required += reqs

        # Construct the base schema
        base_schema = {
            "title": cls.__name__,
            "type": "object",
            "properties": properties,
            "$defs": definitions,  # Include definitions for references
            "required": required,
        }

        return base_schema
    
    def display(self):
        return self.model_dump_json()
    

class Query(BaseModel):
    type: str = 'query'
    query: str = None
    where: Dict[str, (int|str|bool)] = None
    collection: str = None

    def __init__(self, query=None, where=None, collection=None, **kwargs):
        super().__init__(query=query, where=where, collection=collection, **kwargs)


class Subscription(Entity):
    type: str = 'subscription'
    query: Query = None

    def __init__(self, query=None, **kwargs):
        super().__init__(query=query, **kwargs)


class VectorDB:
    name: str

    @abstractmethod
    def get(self, ids=None, where=None, **kwargs):
        '''
        Get embeddings by ids or where clause.
        '''

    @abstractmethod
    def query(self, texts, where=None, ids=None, **kwargs):
        '''
        Query embeddings using a list of texts and optional where clause.
        '''

    @abstractmethod
    def get_collection(self, name, **kwargs):
        '''
        Return a collection if it exists.
        '''

    @abstractmethod
    def get_or_create_collection(self, name, **kwargs):
        '''
        Return a collection or create a new one if it doesn't exist.
        '''

    @abstractmethod
    def delete_collection(self, name, **kwargs):
        '''
        Return a collection or create a new one if it doesn't exist.
        '''
    
    @abstractmethod
    def collections():
        '''
        Return a list of collections.
        '''
    
    @abstractmethod
    def upsert(self, ids, documents, metadatas, **kwargs):
        '''
        Upsert embeddings.
        '''


class ChromaVectorDB(VectorDB):

    def __init__(self, endpoint=None, api_key=None, path=None, **kwargs):
        self.client = chromadb.PersistentClient(path=f'{path}/.px/db' if path else "./.px/db")

    def query(self, texts, where=None, ids=None, **kwargs):
        return self.client.query(texts, where=where, **kwargs)
    
    def get_or_create_collection(self, name, **kwargs):
        return self.client.get_or_create_collection(name, **kwargs)
    
    def create_collection(self, name, **kwargs):
        return self.client.create_collection(name, **kwargs)
    
    def get_collection(self, name, **kwargs):
        try:
            return self.client.get_collection(name, **kwargs)
        except ValueError:
            return None
    
    def delete_collection(self, name, **kwargs):
        return self.client.delete_collection(name, **kwargs)
    
    def collections(self):
        return self.client.list_collections()
    
    def upsert(self, ids, documents, metadatas, **kwargs):
        return self.client.upsert(ids, documents, metadatas, **kwargs)
    
    def get(self, ids=None, where=None, **kwargs):
        return self.client.get(ids=ids, where=where, **kwargs)


class EntitySeries(pd.Series):

    @property
    def _constructor(self):
        return EntitySeries

    @property
    def _constructor_expanddim(self):
        return Collection
    
    @property
    def object(self):
        d = self.to_dict()
        return Entity(**d)


class Collection(pd.DataFrame):
    _metadata = ['db', 'schema', 'session']

    @property
    def _constructor(self, *args, **kwargs):
        return Collection
    
    @property
    def _constructor_sliced(self):
        return EntitySeries
    
    @classmethod
    def load(cls, session, db):
        records = db.get(where={'item': 1})
        docs = [
            {
                'id': id, 
                **json.loads(r), 
            } 
            for id, r, m in zip(records['ids'], records['documents'], records['metadatas'])
        ]
        c = Collection(docs)
        c.db = db
        c.session = session
        return c
    
    def embedding_query(self, *texts, ids=None, where=None, threshold=0.1, limit=None, **kwargs):
        texts = [t for t in texts if t is not None]
        
        scores = {}
        if len(texts) == 0:
            results = self.db.get(ids=ids, where=where, **kwargs)
            for id, m in zip(results['ids'], results['metadatas']):
                if m.get('item') != 1:
                    id = m.get('item_id')
                if id not in scores:
                    scores[id] = 1
                else:
                    scores[id] += 1
        else:
            results = self.db.query(query_texts=texts, where=where, **kwargs)
            for i in range(len(results['ids'])):
                for id, d, m in zip(results['ids'][i], results['distances'][i], results['metadatas'][i]):
                    if m.get('item') != 1:
                        id = m.get('item_id')
                    if id not in scores:
                        scores[id] = 1 - d
                    else:
                        scores[id] += 1 - d
        
        try:
            filtered_scores = {k: v for k, v in scores.items() if v >= threshold}
            sorted_ids = sorted(filtered_scores, key=filtered_scores.get, reverse=True)
            results = self[self['id'].isin(sorted_ids)].set_index('id').loc[sorted_ids].reset_index()
            logger.info(f'Found {len(results)} results for query: {texts}')
            if limit is not None:
                return results.head(limit)
            else:
                return results
        except KeyError as e:
            logger.error(f'Failed to parse query results: {e}')
            return None
    
    def __call__(self, *texts, where=None, **kwargs) -> Any:
        return self.embedding_query(*texts, where=where, **kwargs)
    
    @property
    def name(self):
        return self.db.name
    
    @property
    def objects(self):
        if self.empty:
            return []
        if hasattr(self, 'db'):
            ids = self['id'].values.tolist()
            d = self.db.get(ids=ids)
            m = {id: metadata for id, metadata in zip(d['ids'], d['metadatas'])}
            schemas = {
                id: json.loads(metadata['schema']) for id, metadata in m.items()
                    if 'schema' in metadata and metadata['schema'] is not None
            }
            return [
                create_entity_from_schema(
                    schemas.get(r['id']),
                    {
                        k: v for k, v in r.items() if (len(v) if isinstance(v, list) else pd.notnull(v))
                    },
                    session=self.session,
                    base=REGISTERED_ENTITIES.get(r['type'], Entity),
                ) 
                for r in self.to_dict('records')
            ]
        else:
            return [
                create_entity_from_schema(
                    self.schema or {},
                    {
                        k: v for k, v in r.items() if (len(v) if isinstance(v, list) else pd.notnull(v))
                    },
                    base=REGISTERED_ENTITIES.get(r['type'], Entity),
                ) 
                for r in self.to_dict('records')
            ]
    
    @property
    def first(self):
        objects = self.objects
        if len(objects) == 0:
            return None
        else:
            return objects[0]
    
    def delete(self, *items):
        self.db.delete(ids=[i.id.replace(' ', '') for i in items])
        for item in items:
            self.db.delete(where={'item_id': item.id})
        self.drop(self[self['id'].isin([i.id for i in items])].index, inplace=True)
        logger.info(f'Deleted {len(items)} items from {self.name}')

    def embed(self, *items, **kwargs):
        records = self._create_records(*items, **kwargs)
        if len(records) == 0:
            raise ValueError('No items to embed')
        
        # dedupe the records based on id
        records = {r['id']: r for r in records}.values()

        ids = [r['id'] for r in records]
        self.db.upsert(
            ids=ids,
            documents=[r['document'] for r in records],
            metadatas=[r['metadata'] for r in records],
        )

        if self.empty:
            new_items = [r for r in records if r['metadata']['item'] == 1]
        else:
            new_items = [
                r for r in records 
                if r['id'] not in self['id'].values
                and r['metadata']['item'] == 1
            ]
        docs = [{'id': r['id'], **json.loads(r['document'])} for r in new_items]
        df = pd.concat([self, Collection(docs)], ignore_index=True)
        self.drop(self.index, inplace=True)
        for column in df.columns:
            self[column] = df[column]
        logger.info(f'Embedded {len(new_items)} items into {self.name}')

        for item in items:
            if item.type not in REGISTERED_ENTITIES:
                REGISTERED_ENTITIES[item.type] = item.__class__
        return self

    def _create_records(self, *items, **kwargs):
        records = []

        def _field_serializer(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Entity):
                return {
                    'ids': [obj.id],
                    'collection': self.name,
                    'limit': 1,
                }
            elif isinstance(obj, list):
                if len(obj) and isinstance(obj[0], Entity):
                    return {
                        'ids': [o.id for o in obj],
                        'collection': self.name,
                    }
            raise TypeError(f"Type {type(obj)} not serializable")

        def _serializer(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Entity):
                record = { 'id': obj.id, 'type': obj.type }
                for name, field in obj.model_dump().items():
                    if name in ['id', 'type']:
                        continue
                    f = obj.model_fields.get(name)
                    if f is None:
                        logger.error(f'Field {name} not found in {obj.__class__}')
                        continue
                    if isinstance(f.annotation, type) and issubclass(f.annotation, Entity):
                        field = _field_serializer(getattr(obj, name))
                    if f.json_schema_extra and f.json_schema_extra.get('embed', True) == False:
                        continue
                    record[name] = field
            raise TypeError(f"Type {type(obj)} is not serializable")

        def _schema_serializer(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, PydanticUndefinedType):
                return None
            elif isinstance(obj, BaseModel):
                return obj.model_json_schema()
            raise TypeError(f"Type {type(obj)} not serializable", obj)

        for item in items:
            now = datetime.now().isoformat()
            
            if isinstance(item, str):
                item = Entity(type='string', value=item)
            
            for name, field in item.model_dump().items():
                if name in ['id', 'type']:
                    continue

                f = item.model_fields.get(name)
                if f is None:
                    continue

                document = json.dumps({name: field}, default=_serializer)
                if isinstance(f.annotation, type) and issubclass(f.annotation, Entity):
                    # TODO: Handle nested fields
                    logger.debug(f'Field {name} is an Entity')
                    continue
                if f.json_schema_extra and f.json_schema_extra.get('embed', True) == False:
                    continue
                if isinstance(field, int) or isinstance(field, float) or isinstance(field, bool):
                    continue

                field_record = {
                    'id': f'{item.id}_{name}',
                    'document': document,
                    'metadata': {
                        'field': name,
                        'collection': self.name,
                        'item': 0,
                        'item_id': item.id,
                        'created_at': now,
                    },
                }
                records.append(field_record)

            doc = { k: v for k, v in item.model_dump().items() if k not in ['id'] }
            for k, v in doc.items():
                # if v represents an Entity field then create records for it and replace it with a reference
                # use model_fields to check if the current v is an Entity
                f = item.model_fields.get(k)
                if f is None:
                    continue
                if v is None:
                    continue
                if isinstance(f.annotation, type) and issubclass(f.annotation, Entity):
                    doc[k] = _field_serializer(getattr(item, k))
                    records += self._create_records(getattr(item, k))
                elif getattr(f.annotation, '__origin__', None) == list and isinstance(f.annotation.__args__[0], type) and issubclass(f.annotation.__args__[0], Entity):
                    doc[k] = _field_serializer(getattr(item, k))
                    records += self._create_records(*getattr(item, k))

            doc_record = {
                'id': item.id,
                'document': json.dumps(doc, default=_serializer),
                'metadata': {
                    'collection': self.name,
                    'type': item.type,
                    'item': 1,
                    'schema': json.dumps(model_to_json_schema(item.__class__)),
                    'created_at': now,
                },
            }
            records.append(doc_record)
        
        return records


class CollectionEntity(Entity):
    type: str = 'collection'
    name: str = None
    description: str = None

    def __init__(self, name, description=None, records=None, **kwargs):
        super().__init__(
            name=name, description=description, records=records, **kwargs
        )