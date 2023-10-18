import pytest

from promptz import *
from promptz.utils import Entity
from promptz.world import World
from promptz.collection import VectorDB


class User(Entity):
    name: str
    age: int


def test_store(mocker):
    user = User(name="test", age=20)
    db = mocker.Mock(spec=VectorDB)
    c = mocker.Mock(spec=VectorDB)
    c.get.return_value = {'ids': [user.id], 'documents': [user.json()], 'metadatas': [{'schema': user.schema_json()}]}
    c.name = 'test_collection'
    db.get_or_create_collection.return_value = c
    db.get_collection.return_value = c
    db.collections.return_value = [c]
    world = World('tests', db)
    session = world.create_session('test_store')
    session.store(user)

    results = session.query()
    user_ = results.first

    assert len(results) == 1
    assert user_.name == "test"
    assert user_.age == 20