from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .world import World 
from .collection import Collection, Query
from .prompts import PromptDetails


class PromptInput(BaseModel):
    input: Any = None


class API:
    world: World

    def __init__(self, world, logger=None):
        self.world = world
        self.logger = logger or world.logger.getChild('api')
        self.fastapi_app = FastAPI()
        
        @self.fastapi_app.get("/prompts")
        async def get_prompts():
            r = self.world.prompts()
            if r is None:
                prompts = []
            else:
                prompts = r.objects

            return {"response": prompts}

        @self.fastapi_app.get("/prompts/{id}")
        async def get_prompt(id: str):
            history = self.world.collections['history']
            c = Collection.load(history)()
            if c is None or c.empty:
                results = []
            else:
                results = c[c['prompt'] == id].to_dict('records')
            prompt = self.world.prompts(ids=[id]).first
            return {'prompt': prompt, 'results': results}
        
        @self.fastapi_app.post("/prompts")
        async def create_prompt(details: PromptDetails):
            p = self.world.create_prompt(details)
            return p.first.id

        @self.fastapi_app.post("/prompts/{id}/run")
        async def run_prompt(id: str, input: PromptInput = None):
            session = self.world.create_session()
            prompt_config = self.world.prompts(ids=[id]).first
            if prompt_config is None:
                self.logger.info(f'prompts: {self.world.prompts()}')
                raise HTTPException(status_code=404, detail="Prompt not found")
            response = session.prompt(**{**dict(prompt_config), 'input': input})
            return {"response": response}
        
        @self.fastapi_app.get("/history")
        async def get_history():
            history = Collection.load(self.world.collections['history'])
            if history.empty:
                return {'response': []}
            else:
                return {'response': history().objects}

        @self.fastapi_app.get("/systems")
        async def get_systems():
            return {"response": self.world.systems.keys()}
        
        @self.fastapi_app.get("/systems/{name}")
        async def get_system(name: str):
            return {"response": self.world.systems[name]}
        
        @self.fastapi_app.post("/systems/run")
        async def run_systems():
            session = self.world.create_session()
            return {"response": self.world(session)}
        
        @self.fastapi_app.post("/query")
        async def query(query: Query):
            session = self.world.create_session()
            response = session.query(query.query, where=query.where, collection=query.collection)
            return {"response": response}
        
        @self.fastapi_app.get("/notebooks")
        async def get_notebooks():
            return {"response": self.world.notebooks}
        
        @self.fastapi_app.get("/notebooks/{name}")
        async def get_notebook(name: str):
            return {"response": self.world.notebooks[name]}
        
        @self.fastapi_app.get("/chats")
        async def get_chats():
            return {"response": {}}