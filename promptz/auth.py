from enum import Enum
from .utils import Entity


class UserRole(Enum):
    admin = 'admin'


class User(Entity):
    role: UserRole
    username: str


class DefaultUser(User):
    role = UserRole.admin
    username = 'root'