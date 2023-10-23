import random
import uuid
import pytest
from enum import Enum
from pydantic import Field

from promptx import *
from promptx.utils import Entity
from promptx.world import World
from promptx.collection import VectorDB, ChromaVectorDB


class Role(Enum):
    admin = 'admin'
    user = 'user'

class Trait(Enum):
    nice = 'nice'
    mean = 'mean'
    funny = 'funny'
    smart = 'smart'

class Address(Entity):
    street: str
    city: str
    state: str
    zip: str

class User(Entity):
    name: str = Field(..., min_length=3, max_length=20)
    age: int = Field(..., ge=18, lt=100)
    role: Role = Role.admin
    banned: bool = Field(None, generate=False)
    vigor: float = Field(0, max=1, min=0)
    traits: List[Trait] = Field(None, description='What kind of personality describes the user?', min_items=1, max_items=3)
    friends: list['User'] = None
    address: Address = None

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


def _user():
    name = random.choice(['John', 'Jane', 'Jack', 'Jill'])
    age = random.randint(18, 99)
    role = random.choice(list(Role))
    banned = random.choice([True, False])
    vigor = random.random()
    return User(
        name=name, 
        age=age,
        role=role,
        banned=banned,
        vigor=vigor,
    )

@pytest.fixture
def user():
    return _user()


def test_store(session, user):
    session.store(user)
    x = session.query(ids=[user.id]).first

    assert x is not None
    assert x.id == user.id
    assert x.name == user.name
    assert x.age == user.age
    assert x.role == user.role.value
    assert x.banned == user.banned
    assert x.vigor == user.vigor

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

def test_store__alt_collection(session, user):
    session.store(user, collection='alt-test')

    assert session.query(ids=[user.id]) is None

    user = session.query(ids=[user.id], collection='alt-test').first
    assert user is not None

def test_store__list_field(session, user):
    traits = random.choices(list(Trait), k=3)
    user.traits = traits
    session.store(user)
    x = session.query(ids=[user.id]).first

    assert x is not None
    assert len(x.traits) == len(traits)
    assert x.traits[0] == traits[0].value
    assert x.traits[1] == traits[1].value
    assert x.traits[2] == traits[2].value

def test_store__relations(session, user):
    friends = [_user() for _ in range(3)]
    user.friends = friends
    session.store(user)
    x = session.query(ids=[user.id]).first

    assert x is not None
    assert len(x.friends) == len(friends)
    assert x.friends[0].id == friends[0].id
    assert x.friends[1].id == friends[1].id
    assert x.friends[2].id == friends[2].id
    assert x.friends[0].name == friends[0].name

def test_store__relation(session, user):
    address = Address(street='123 Main St', city='New York', state='NY', zip='10001')
    user.address = address
    session.store(user)
    x = session.query(ids=[user.id]).first

    assert x is not None
    assert x.address is not None
    assert x.address.id == address.id
    assert x.address.street == address.street
    assert x.address.city == address.city
    assert x.address.state == address.state
    assert x.address.zip == address.zip

    y = session.query(ids=[address.id]).first
    assert y is not None
    assert y.id == address.id

def test_query__ids(session):
    users = [_user() for _ in range(3)]
    session.store(*users)
    x = session.query(ids=[u.id for u in users])

    assert x is not None
    assert len(x.objects) == len(users)
    assert x[x['id'] == users[0].id].first.id == users[0].id