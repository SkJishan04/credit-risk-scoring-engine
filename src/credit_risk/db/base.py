"""Declarative base shared by all ORM models and Alembic autogeneration."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass