from enum import Enum
from .collection import Entity


class UserRole(Enum):
    admin = 'admin'


class User(Entity):
    role: UserRole
    username: str


class DefaultUser(User):
    role: UserRole = UserRole.admin
    username: str = 'root'