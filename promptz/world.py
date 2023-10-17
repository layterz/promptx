import json
import uuid
import logging
import numpy as np
from pydantic import BaseModel
from typing import Any, Dict, List
from chromadb.utils import embedding_functions

from .collection import Collection, CollectionEntity, Query, VectorDB
from .template import Template, TemplateRunner, MaxRetriesExceeded, MockLLM
from .models import PromptLog, QueryLog
from .chat import ChatBot
from .logging import JSONLogFormatter, NotebookFormatter
from .utils import model_to_json_schema


class Session:
    _chat_history: List[PromptLog] = None

    def __init__(self, world, name, db, llm, ef=None, default_collection='default', logger=None, collections=None):
        self.world = world
        self.name = name
        self.db = db
        self.llm = llm
        self.ef = ef or embedding_functions.DefaultEmbeddingFunction()
        self._collection = default_collection
        self._chat_history = []
        if collections is not None:
            self._collections = collections 
        self.logger = logger or self.world.logger.getChild(self.name)
    
    def _run_prompt(self, t, input, context=None, history=None, dryrun=False, retries=3, to_json=False, **kwargs):
        e = None
        s = self.world.template_system
        rendered = s.render(t, {'input': s.parse(input)})
        try:
            if t.data is not None:
                print('t.data', t.data)
            r = s(t, input, context=context, history=history, dryrun=dryrun, retries=retries, **kwargs)
            log = PromptLog(template=t.id, raw_input=rendered, raw_output=r.raw)
            self.store(log, collection='logs')
            if isinstance(r.content, list):
                if len(r.content) > 0 and isinstance(r.content[0], BaseModel):
                    es = [dict(e) for e in r.content]
                else:
                    es = r.content
                if to_json:
                    return es
                else:
                    c = Collection(es)
                    if t.output is not None:
                        output_schema = json.loads(t.output)
                        # if output schema is a list, get the inner type
                        if output_schema.get('type', None) == 'array':
                            output_schema = output_schema.get('items', {})
                        c.schema = output_schema
                    return c
            else:
                if to_json:
                    return dict(r.content)
                else:
                    return r.content
        except MaxRetriesExceeded as e:
            self.logger.error(f'Max retries exceeded: {e}')
            log = PromptLog(template=t.id, raw_input=rendered, error=str(e))
            self.store(log, collection='logs')
            return None
    
    def _run_batch(self, t, inputs, dryrun=False, retries=3, to_json=False, **kwargs):
        o = []
        for input in inputs:
            r = self._run_prompt(t, input, dryrun=dryrun, retries=retries, to_json=to_json, **kwargs)
            if r is not None:
                o.append(r.content)
        return o

    def prompt(self, instructions=None, input=None, output=None, id=None, context=None, template=None, llm=None, examples=None, num_examples=1, logs=None, tools=None, dryrun=False, retries=3, debug=False, silent=False, to_json=False, **kwargs):
        logger = self.logger.getChild('prompt')
        level = logging.INFO
        if debug: level = logging.DEBUG
        elif silent: level = logging.ERROR
        logger.setLevel(level)

        if output is not None:
            output = model_to_json_schema(output)
            if output is not None:
                output = json.dumps(output)

        if template is None:
           template = Template(
                id=id,
                output=output,
                instructions=instructions,
                llm=llm or self.llm,
                context=context,
                examples=examples,
                num_examples=num_examples,
                logs=logs,
                tools=tools,
                debug=debug,
                silent=silent,
                logger=logger,
            )
        elif isinstance(template, str):
            template = self.world.templates(ids=[template]).first
            if template is None:
                raise ValueError(f'No template found with id {template}')

        if isinstance(input, list):
            return self._run_batch(template, input, dryrun=dryrun, retries=retries, to_json=to_json, **kwargs)
        else:
            return self._run_prompt(template, input, dryrun=dryrun, retries=retries, to_json=to_json, **kwargs)
    
    def embed(self, item, field=None):
        if isinstance(item, str):
            return self.ef([item])[0]
        elif isinstance(item, BaseModel):
            return self.ef([item.json()])[0]
        elif isinstance(item, list):
            return [self.embed(i, field=field) for i in item]
    
    def query(self, *texts, field=None, ids=None, where=None, collection=None, limit=None):
        c = self.collection(collection)
        if c is None:
            raise ValueError(f'No collection found with name {collection}')
        if len(texts) == 0 and field is None and where is None:
            if limit is None:
                return c
            else:
                return c.head(limit)
        where = where or {}
        if field is not None:
            where['field'] = field
        r =  c(*texts, ids=ids, where=where, limit=limit)
        if r is None:
            return None

        def serialize(item):
            if isinstance(item, BaseModel):
                return item.json()
            else:
                return item
        
        if False:
            # TODO: fix this
            log = QueryLog(
                query=texts,
                where=where,
                collection=collection,
                result=json.dumps(r.objects, default=serialize),
            )
            self.store(log, collection='queries')
        return r
    
    def chat(self, message, context=None, agent=None, **kwargs):
        if agent is None:
            agent = ChatBot('default')
        
        _context = []
        
        if isinstance(context, Collection):
            try:
                for item in context(message, limit=3).objects:
                    try:
                        _context.append(item.text)
                    except AttributeError:
                        _context.append(item.value)
            except Exception as e:
                self.logger.error(f'Error getting context: {e}')

        if len(_context):
            _context = [
                '''
                The following is some additional context to the question being asked.
                You can use other information from your training data as well, but the answer should be focused on the information provided.
                ''',
                *_context,
            ]

        history = self._chat_history[-5:]
        output = self._run_prompt(
            agent.template, {'message': message }, context='\n'.join(_context), history=history,
        )
        self._chat_history.append(PromptLog(
            template=agent.template.id,
            input=message,
            output=output,
        ))
        return output

    def store(self, *items, collection=None):
        def flatten(lst):
            return [
                item for sublist in lst 
                for item in (
                    flatten(sublist) if isinstance(sublist, list) else [sublist]
                )
            ]
        
        items = flatten([
            item.objects if isinstance(item, Collection) else item 
            for item in items
        ])
        
        c = self.collection(collection)
        if c is None:
            self.world.create_collection(collection)
            c = self.collection(collection)
        c.embed(*[item for item in items if item is not None])
    
    def delete(self, *items, collection=None):
        c = self.collection(collection)
        if c is None:
            raise ValueError(f'No collection found with name {collection}')
        
        c.delete(*items)
    
    def delete_collection(self, collection):
        self.world.delete_collection(collection)
    
    def create_collection(self, collection):
        self.world.create_collection(collection)
    
    def collections(self):
        return self.world.db.collections()
    
    def chain(self, *steps, llm=None, **kwargs):
        llm = llm or self.llm
        input = None
        cc = self.collection(f'chain.{uuid.uuid4()}')
        for step in steps:
            if isinstance(step, Template):
                p = step
                p.llm = llm
                input = self._run_prompt(p, input, **kwargs)
                self.store(*input, collection=cc.name)
            if isinstance(step, str):
                p = Template(step, llm=llm)
                input = self._run_prompt(p, input, **kwargs)
                if isinstance(input, list):
                    if isinstance(input[0], BaseModel):
                        self.store(*input, collection=cc.name)
                elif isinstance(input, BaseModel):
                    self.store(input, collection=cc.name)
            if isinstance(step, dict):
                input = {
                    k: self.chain(v, llm=llm, **kwargs)
                    for k, v in step.items()
                }
            if isinstance(step, list):
                input = [
                    self.chain(v, llm=llm, **kwargs)
                    for v in step
                ]
            if isinstance(step, Query):
                qc = self.collection(step.collection) if step.collection is not None else cc
                input = qc(step.query or '', where=step.where)
        return cc
        
    def collection(self, name=None):
        if name is None:
            name = self._collection
        collection = self.world._collections.get(name)
        return collection
    
    def evaluate(self, actual, expected, **kwargs):
        threshold = 0.69
        ex = self.embed(actual)
        ey = self.embed(expected)
        dot_product = np.dot(ex, ey)
        norm_1 = np.linalg.norm(ex)
        norm_2 = np.linalg.norm(ey)

        cosine_similarity = dot_product / (norm_1 * norm_2)
        if cosine_similarity < threshold:
            cosine_similarity = 0
        rescaled = (cosine_similarity - threshold) * (100 / (1 - threshold))
        return max(0, rescaled)
    
    def __call__(self, collection: Collection, *args, **kwargs):
        return self.run(collection, *self._processors, **kwargs)
    
    @property
    def logs(self):
        return self.collection('logs')


# TODO: rename World to Space
class World:
    name: str
    sessions: List[Session]
    _collections: Dict[str, Collection]
    db: VectorDB

    def __init__(self, name, db, llm=None, ef=None, logger=None, templates=None):
        self.name = name
        self.sessions = []
        self._collections = {}
        self.llm = llm or MockLLM()
        self.ef = ef or (lambda x: [0] * len(x))
        self.db = db
        self.logger = logger or logging.getLogger(self.name)
        
        collection = self.db.get_or_create_collection('collections')
        self.collections = Collection.load(collection)
        self.create_collection('default', 'Default collection used when calling store()')
        self.create_collection('logs', 'Stores a log of all prompts and their outputs')
        self.create_collection('queries', 'Query stored objects in collections')
        self.create_collection('subscriptions', 'Subscriptions to queries')
        self.create_collection('agents', 'Configurations for interactive and autonomous AI agents')
        self.create_collection('models', 'Configurations for AI models')

        self.create_collection('templates', 'Prompt templates used to interact with AI models')
        for template in (templates or []):
            self.create_template(template)
        
        for collection in self.db.collections():
            self.create_collection(collection.name)
        
        # TODO: hack, should register as normal system
        self.template_system = TemplateRunner(llm=self.llm, logger=self.logger.getChild('template_system'))
    
    def create_session(self, user, name=None, db=None, llm=None, ef=None, logger=None, silent=False, debug=False, log_format='notebook'):
        llm = llm or self.llm
        ef = ef or self.ef
        db = db or self.db
        logger = logger or self.logger.getChild(f'session.{name}')
        ch = logging.StreamHandler()
        formatter = JSONLogFormatter() if log_format == 'json' else NotebookFormatter()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        level = logging.INFO
        if debug: level = logging.DEBUG
        elif silent: level = logging.ERROR
        logger.setLevel(level)
        session = Session(self, name=name, db=db, llm=llm, ef=ef, logger=logger,
                          collections=self.collections)
        self.sessions.append(session)
        return session
    
    def create_collection(self, name, description=None, metadata=None):
        collection = self.db.get_collection(name)
        if collection is None:
            if metadata is None:
                metadata = {"hnsw:space": "cosine"}
            collection = self.db.create_collection(name, metadata=metadata)
            r = CollectionEntity(name=name, description=description)
            self.collections.embed(r)
        c = Collection.load(collection)
        self._collections[name] = c
        return collection
    
    def delete_collection(self, name):
        self.db.delete_collection(name)
        if name in self._collections:
            del self._collections[name]

    def create_template(self, template: Template):
        return self.templates.embed(template)
    
    @property
    def templates(self):
        return self._collections['templates']
    
    @property
    def logs(self):
        return self._collections['logs']

    def __call__(self, session, *args: Any, **kwds: Any) -> Any:
        for system in self.systems().objects:
            q = Query(**system.query)
            items = session.query(q.query, where=q.where, collection=q.collection)
            try:
                _system = self._systems[system.name]
                updates = _system(items)
                if updates is not None:
                    session.store(updates)
            except Exception as e:
                raise e