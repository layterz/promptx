import os
import uuid
import logging
import numpy as np
from pydantic import BaseModel
from typing import Any, Dict, List, Callable
from chromadb.utils import embedding_functions

from .collection import Collection, Query, ChromaVectorDB
from .prompts import Prompt, PromptDetails, ChatLog, MaxRetriesExceeded, MockLLM
from .logging import JSONLogFormatter, NotebookFormatter


Processor = Callable[[Collection], Collection]

class System:
    query: Query
    processor: Processor

    def __init__(self, query=None, processor=None, **kwargs):
        self.query = query or self.query
        self.processor = processor 
    
    def process(self, items, **kwargs):
        if self.processor is not None:
            return self.processor(items, **kwargs)
        else:
            raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.process(*args, **kwds)


class Session:

    def __init__(self, world, name, db, llm, ef=None, default_collection='default', logger=None, collections=None):
        self.world = world
        self.name = name
        self.db = db
        self.llm = llm
        self.ef = ef or embedding_functions.DefaultEmbeddingFunction()
        self._collection = default_collection
        if collections is not None:
            self._collections = collections 
        self.logger = logger or self.world.logger.getChild(self.name)
    
    def _run_prompt(self, p, input, dryrun=False, retries=3, **kwargs):
        e = None
        try:
            rendered = p.render({'input': p.parse(input)})
            r = p(input, dryrun=dryrun, retries=retries, **kwargs)
            log = ChatLog(prompt=p.id, input=rendered, output=r.raw)
            self.store(log, collection='history')
            return r
        except MaxRetriesExceeded as e:
            self.logger.error(f'Max retries exceeded: {e}')
            return None
    
    def _run_batch(self, p, inputs, dryrun=False, retries=3, **kwargs):
        o = []
        for input in inputs:
            r = self._run_prompt(p, input, dryrun=dryrun, retries=retries, **kwargs)
            if r is not None:
                o.append(r.content)
        return o

    def prompt(self, instructions=None, input=None, output=None, id=None, prompt=None, context=None, template=None, llm=None, examples=None, num_examples=1, history=None, tools=None, dryrun=False, retries=3, debug=False, silent=False, **kwargs):
        logger = self.logger.getChild('prompt')
        level = logging.INFO
        if debug: level = logging.DEBUG
        elif silent: level = logging.ERROR
        logger.setLevel(level)

        if prompt is None:
            p = Prompt(
                id=id,
                output=output,
                instructions=instructions,
                llm=llm or self.llm,
                context=context,
                examples=examples,
                num_examples=num_examples,
                history=history,
                tools=tools,
                template=template,
                debug=debug,
                silent=silent,
                logger=logger,
            )
        else:
            p = prompt
            p.llm = llm or self.llm
            p.logger = self.logger
            p.debug = debug
            p.silent = silent

        if isinstance(input, list):
            o = self._run_batch(p, input, dryrun=dryrun, retries=retries, **kwargs)
        else:
            r = self._run_prompt(p, input, dryrun=dryrun, retries=retries, **kwargs)
            if r is None:
                return None
            o = r.content
        return o
    
    def embed(self, item, field=None):
        if isinstance(item, str):
            return self.ef([item])[0]
        elif isinstance(item, BaseModel):
            return self.ef([item.json()])[0]
        elif isinstance(item, list):
            return [self.embed(i, field=field) for i in item]
    
    def query(self, *texts, field=None, where=None, collection=None):
        c = self.collection(collection)
        if len(texts) == 0 and field is None and where is None:
            return c
        where = where or {}
        if field is not None:
            where['field'] = field
        return c(*texts, where=where)

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
        c.embed(*[item for item in items if item is not None])
        return self.collection(collection)
    
    def chain(self, *steps, llm=None, **kwargs):
        llm = llm or self.llm
        input = None
        cc = self.collection(f'chain.{uuid.uuid4()}')
        for step in steps:
            if isinstance(step, Prompt):
                p = step
                p.llm = llm
                input = self._run_prompt(p, input, **kwargs)
                self.store(*input, collection=cc.name)
            if isinstance(step, str):
                p = Prompt(step, llm=llm)
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
        try:
            collection = self.world.collections[name]
        except KeyError:
            collection = self.db.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
            self.world.collections[name] = collection
        
        return Collection.load(collection)
    
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
    def history(self):
        return self.collection('history')


class World:
    name: str
    sessions: List[Session]
    collections: Dict[str, Collection]
    systems: Dict[str, System]
    notebooks: Dict[str, str]

    def __init__(self, name, systems=None, llm=None, ef=None, logger=None, db=None, prompts=None, notebooks=None):
        self.name = name
        self.sessions = []
        self.collections = {}
        self.llm = llm or MockLLM()
        self.ef = ef or (lambda x: [0] * len(x))
        self.db = db or ChromaVectorDB(path=os.environ.get('PROMPTZ_PATH'))
        self.logger = logger or logging.getLogger(self.name)
        self.systems = systems or {}
        self.notebooks = notebooks or {}

        self.create_collection('prompts')
        for p in prompts:
            self.create_prompt(p)

        history = self.db.get_or_create_collection('history', metadata={"hnsw:space": "cosine"})
        self.collections['history'] = history
    
    def create_session(self, name=None, db=None, llm=None, ef=None, logger=None, silent=False, debug=False, log_format='notebook'):
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
    
    def create_collection(self, name, metadata=None):
        if metadata is None:
            metadata = {"hnsw:space": "cosine"}
        collection = self.db.get_or_create_collection(name, metadata=metadata)
        self.collections[name] = Collection.load(collection)
        return collection

    def create_prompt(self, details: PromptDetails):
        c = self.prompts.embed(details)
        return c

    @property
    def prompts(self):
        return self.collections['prompts']

    def __call__(self, session, *args: Any, **kwds: Any) -> Any:
        for system in self.systems.values():
            items = session.query(system.query)
            updates = system(items)
            session.store(updates)