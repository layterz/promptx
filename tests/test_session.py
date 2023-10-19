import random
import uuid
import pytest
from enum import Enum
from pydantic import Field

from promptz import *
from promptz.utils import Entity
from promptz.world import World
from promptz.collection import VectorDB, ChromaVectorDB


class Role(Enum):
    admin = 'admin'
    user = 'user'

class Trait(Enum):
    nice = 'nice'
    mean = 'mean'
    funny = 'funny'
    smart = 'smart'

class User(Entity):
    name: str = Field(..., min_length=3, max_length=20)
    age: int = Field(..., ge=18, lt=100)
    role: Role = Role.admin
    #traits: List[Trait] = Field(..., description='What kind of personality describes the user?', min_items=1, max_items=3)
    banned: bool = Field(None, generate=False)
    vigor: float = Field(0, max=1, min=0)

class Account(Entity):
    user: User


@pytest.fixture
def session():
    import shutil
    try:
        shutil.rmtree('tests/.db')
    except FileNotFoundError as e:
        pass
    db = ChromaVectorDB(path='tests')
    world = World('tests', db)
    session = world.create_session('test_store')
    yield session
    shutil.rmtree('tests/.db')

@pytest.fixture
def user():
    name = random.choice(['John', 'Jane', 'Jack', 'Jill'])
    age = random.randint(18, 100)
    #traits = random.choices(list(Trait), k=random.randint(1, 3))
    #return User(name=name, age=age, traits=traits)
    return User(name=name, age=age)


def test_store(session, user):
    session.store(user)
    x = session.query(ids=[user.id]).first

    assert x is not None
    assert x.id == user.id
    assert x.name == user.name
    assert x.age == user.age

def test_store__multiple(session):
    n = 2
    ids = [str(uuid.uuid4()) for _ in range(n)]
    session.store(
        *[
            User(id=ids[i], name="test", age=20)
            for i in range(n)
        ]
    )

    users = session.query().objects
    stored_ids = [user.id for user in users]
    assert all([id in stored_ids for id in ids])

def test_store__alt_collection(session):
    user = User(name="test", age=20)
    session.store(user, collection='alt-test')

    assert session.query(ids=[user.id]) is None

    user = session.query(ids=[user.id], collection='alt-test').first
    assert user is not None
    assert user.name == "test"
    assert user.age == 20

def test_store__relation(session):
    pass