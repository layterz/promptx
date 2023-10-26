from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .world import World 
from .collection import Query
from .template import Template
from .auth import DefaultUser


class PromptInput(BaseModel):
    input: Any = None


class API:
    world: World

    def __init__(self, world):
        self.world = world
        self.fastapi_app = FastAPI()

        @self.fastapi_app.get("/inbox")
        async def get_inbox():
            user = DefaultUser()
            logs = self.world.logs(where={'assigned_to': user.id})
            logs = self.world.logs()

            if logs is not None:
                return {"list": logs.objects}
            else:
                return {"list": []}

        @self.fastapi_app.post("/prompt")
        async def run_prompt(details: Template):
            session = self.world.create_session()
            template = Template(**details.dict())
            response = session.prompt(**{**dict(template), 'input': {}})
            return {"response": response}
        
        @self.fastapi_app.get("/templates")
        async def get_templates():
            r = self.world.templates()
            if r is None:
                templates = []
            else:
                templates = r.objects
            
            return {"list": templates}

        @self.fastapi_app.get("/{collection}")
        async def get_index(collection: str, query: str = None):
            if collection == 'collections':
                c = self.world.collections
            else:
                c = self.world._collections[collection]
            if c is None:
                raise HTTPException(status_code=404, detail=f"{collection} collection not found")
            if query is not None:
                r = c(query).objects
            else:
                r = c.objects
            return {'list': r}

        @self.fastapi_app.get("/{collection}/{id}")
        async def get_entity(collection: str, id: str):
            if collection == 'collections':
                c = self.world.collections
            else:
                c = self.world._collections[collection]
            r = c(ids=[id]).first
            return {'details': r, 'collection': collection}

        @self.fastapi_app.get("/templates/{id}")
        async def get_template(id: str):
            template = self.world.templates(ids=[id]).first
            if template is None:
                raise HTTPException(status_code=404, detail="Template not found")
            return {'details': template}
        
        @self.fastapi_app.post("/templates")
        async def create_template(details: Template):
            t = self.world.create_template(details)
            return t.id

        @self.fastapi_app.post("/templates/{id}/run")
        async def run_template(id: str, input: PromptInput = None):
            session = self.world.create_session(DefaultUser, debug=True)
            response = session.prompt(template=id, input=input, to_json=True)
            return {"response": response}
        
        @self.fastapi_app.get("/logs")
        async def get_logs():
            if self.world.logs.empty:
                return {'list': []}
            else:
                return {'list': self.world.logs.objects}

        @self.fastapi_app.get("/conversations")
        async def get_collections():
            return {"list": []}

        @self.fastapi_app.get("/collections")
        async def get_collections():
            return {"list": [c for c in self.world.collections().objects]}

        @self.fastapi_app.get("/collections/{name}")
        async def get_collection(name: str):
            try:
                c = self.world.collections(ids=[name])
                cc = self.world._collections[name]
                if c is None:
                    return {"response": None}
                else:
                    return {"details": dict(c), "list": cc.objects}
            except KeyError as e:
                raise HTTPException(status_code=404, detail="Collection not found")

        @self.fastapi_app.get("/systems")
        async def get_systems():
            r = self.world.systems()
            if r is None:
                systems = []
            else:
                systems = r.objects
            
            return {"list": systems}
        
        @self.fastapi_app.get("/systems/{name}")
        async def get_system(name: str):
            system = self.world.systems(ids=[name]).first
            return {"details": system}
        
        @self.fastapi_app.post("/systems/run")
        async def run_systems():
            session = self.world.create_session()
            return {"response": self.world(session)}
        
        @self.fastapi_app.post("/query")
        async def query(query: Query):
            session = self.world.create_session()
            response = session.query(query.query, where=query.where, collection=query.collection)
            return {"response": response}