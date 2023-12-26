import random
from typing import *
from enum import Enum
import pytest
from pydantic import Field

from promptx import App
from promptx.collection import Entity
from promptx.models import LLM, Response


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
    traits: List[Trait] = Field(..., min_length=1, max_length=3)

class Account(Entity):
    user: User
    payees: List[User] = None


@pytest.fixture
def session():
    import shutil
    try:
        shutil.rmtree('tests/.px')
    except FileNotFoundError as e:
        pass
        
    from promptx.collection import MemoryVectorDB
    db = MemoryVectorDB()
    from promptx.models import MockLLM
    llm = MockLLM()
    app = App.load('tests', db, llm, env={'PX_ENV': 'test'})
    session = app.world.create_session('test_store')
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


@pytest.fixture
def llm(mocker):
    llm = mocker.Mock(spec=LLM)
    llm.generate.return_value = Response(
        raw='This is a mock response.',
    )
    return llm