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
    banned: bool = Field(False, json_schema_extra={'generate': False})
    vigor: float = Field(0, ge=0, le=1)
    traits: List[Trait] = Field(None, description='What kind of personality describes the user?', min_length=1, max_length=3)

class Account(Entity):
    user: User


@pytest.fixture
def session():
    import shutil
    try:
        shutil.rmtree('tests/.px')
    except FileNotFoundError as e:
        pass
    db = ChromaVectorDB(path='tests')
    world = World('tests', db)
    session = world.create_session('test_store')
    yield session
    shutil.rmtree('tests/.px')


def _user():
    name = random.choice(['John', 'Jane', 'Jack', 'Jill'])
    age = random.randint(18, 99)
    role = random.choice(list(Role))
    banned = random.choice([True, False])
    vigor = random.random()
    traits = random.choices(list(Trait), k=3)
    return User(
        name=name, 
        age=age,
        role=role,
        banned=banned,
        vigor=vigor,
        traits=traits,
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
    traits = random.choices(list(Trait), k=3)
    session.store(
        *[
            User(id=ids[i], name="test", age=20, traits=traits)
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

def test_query__ids(session):
    users = [_user() for _ in range(3)]
    session.store(*users)
    x = session.query(ids=[u.id for u in users])

    assert x is not None
    assert len(x.objects) == len(users)
    assert x[x['id'] == users[0].id].first.id == users[0].id

def test_foriegn_key(session, user):
    account = Account(user=user)
    session.store(account)

    _user = session.query(ids=[user.id]).first
    assert _user is not None
    assert _user.id == user.id

    _account = session.query(ids=[account.id]).first
    assert _account is not None
    assert _account.id == account.id
    assert _account.user['id'] == user.id

def test_one_to_many():
    pass

def test_many_to_many():
    pass