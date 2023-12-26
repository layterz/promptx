import chromadb
from ..collection import VectorDB


class ChromaVectorDB(VectorDB):

    def __init__(self, path=None):
        self.client = chromadb.PersistentClient(path=f'{path}/.px/db' if path else "./.px/db")
    
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