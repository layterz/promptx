import json
import uuid
from datetime import datetime
from abc import abstractmethod
from typing import Any, Dict
import pandas as pd
from pydantic import BaseModel 
from IPython.display import display, HTML
import chromadb


class Query(BaseModel):
    query: str
    where: Dict[str, Any] = None
    collection: str = None

    def __init__(self, query, where=None, collection=None, **kwargs):
        super().__init__(query=query, where=where, collection=collection, **kwargs)


class VectorDB:

    @abstractmethod
    def query(self, texts, where=None, ids=None, **kwargs):
        '''
        Query embeddings using a list of texts and optional where clause.
        '''

    @abstractmethod
    def get_or_create_collection(self, name, **kwargs):
        '''
        Return a collection or create a new one if it doesn't exist.
        '''


class ChromaVectorDB(VectorDB):

    def __init__(self, endpoint=None, api_key=None, path=None, **kwargs):
        self.client = chromadb.PersistentClient(path=f'{path}/.db' if path else "./.db")

    def query(self, texts, where=None, ids=None, **kwargs):
        return self.client.query(texts, where=where, **kwargs)
    
    def get_or_create_collection(self, name, **kwargs):
        return self.client.get_or_create_collection(name, **kwargs)


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
    _metadata = ['collection']

    @property
    def _constructor(self, *args, **kwargs):
        return Collection
    
    @property
    def _constructor_sliced(self):
        return EntitySeries
    
    @classmethod
    def load(cls, collection):
        records = collection.get(where={'item': 1})
        docs = [
            {'id': id, **json.loads(r)} 
            for id, r in zip(records['ids'], records['documents'])
        ]
        c = Collection(docs)
        c.collection = collection
        return c
    
    def embedding_query(self, *texts, ids=None, where=None, threshold=0.5, **kwargs):
        texts = [t for t in texts if t is not None]
        
        scores = {}
        if len(texts) == 0:
            results = self.collection.get(ids=ids, where=where, **kwargs)
            for id, m in zip(results['ids'], results['metadatas']):
                if m.get('item') != 1:
                    id = m.get('item_id')
                if id not in scores:
                    scores[id] = 1
                else:
                    scores[id] += 1
        else:
            results = self.collection.query(query_texts=texts, where=where, **kwargs)
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
            return results
        except KeyError as e:
            return None
    
    def __call__(self, *texts, where=None, **kwargs) -> Any:
        return self.embedding_query(*texts, where=where, **kwargs)
    
    @property
    def name(self):
        return self.collection.name
    
    @property
    def objects(self):
        return [r.object for _, r in self.iterrows()]
    
    @property
    def first(self):
        if len(self.objects) == 0:
            return None
        else:
            return self.objects[0]
    
    def embed(self, *items, **kwargs):
        records = []
        for item in items:
            id = item['id'] or str(uuid.uuid4())
            now = datetime.now().isoformat()

            for name, field in item.items():
                if name in ['id', 'type']:
                    continue

                # TODO: Handle nested fields
                field_record = {
                    'id': f'{id}_{name}',
                    'document': json.dumps({name: field}),
                    'metadata': {
                        'field': name,
                        'collection': self.name,
                        'item': 0,
                        'item_id': id,
                        'created_at': now,
                    },
                }
                records.append(field_record)

            doc = { k: v for k, v in item.items() if k not in ['id'] }
            doc_record = {
                'id': id,
                'document': json.dumps(doc),
                'metadata': {
                    'collection': self.name,
                    'type': item['type'],
                    'item': 1,
                    'created_at': now,
                },
            }
            records.append(doc_record)
            
        ids = [r['id'] for r in records]
        self.collection.upsert(
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
        return self