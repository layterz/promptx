import random
import uuid
import pytest

from . import User, Trait, Account, _user, user, session


def test_store(session, user):
    session.store(user)
    x = session.query(ids=[user.id]).first

    assert x is not None
    assert x.id == user.id
    assert x.name == user.name
    assert x.age == user.age
    assert x.role.value == user.role.value
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
    assert x.traits[0].value == traits[0].value
    assert x.traits[1].value == traits[1].value
    assert x.traits[2].value == traits[2].value

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
    assert _user.name == user.name

    _account = session.query(ids=[account.id]).first
    assert _account is not None
    assert _account.id == account.id
    assert _account.user.id == user.id
    assert _account.user.name == user.name

def test_one_to_many(session, user):
    payees = [_user() for _ in range(3)]
    account = Account(user=user, payees=payees)
    session.store(account)

    for user in session.query(ids=[u.id for u in payees]).objects:
        assert user is not None
        assert user.id in [u.id for u in payees]

    _account = session.query(ids=[account.id]).first
    assert _account is not None
    assert len(_account.payees) == len(payees)
    assert _account.payees[0].id == payees[0].id
    assert _account.payees[0].name == payees[0].name
    assert _account.payees[1].id == payees[1].id