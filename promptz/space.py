from typing import *

from .collection import Collection, VectorDB


class Query:
    pass


class System:
    pass


# Should space just be the nested space/subspace structure?
class Space:
    name: str
    _db: VectorDB

    def __init__(self, name):
        super().__init__(name=name)
        self.create_collection('subspaces')
    
    def create_subspace(self, name, **kwargs):
        pass

    def create_collection(self, name, **kwargs):
        return self._db.get_or_create_collection(name=name, **kwargs)

    def collection(self, name):
        db = self._db.get_collection(name=name)
        return Collection.load(db)
    
    @property
    def collections(self):
        return self._db.get_collections()
    
    @property
    def subspaces(self):
        return self.collection('subspaces')


class App(Space):

    def __init__(self, name):
        super().__init__(name=name)
        self.create_collection('templates')
        self.create_collection('systems')
        self.create_collection('sessions')

    def create_session(self, name, **kwargs):
        pass

    def create_system(self, name, **kwargs):
        pass

    @property
    def templates(self):
        return self.collection('templates')
    
    @property
    def systems(self):
        return self.collection('systems')
    
    @property
    def sessions(self):
        return self.collection('sessions')
    
    def run(self):
        pass