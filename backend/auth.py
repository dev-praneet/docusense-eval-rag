import hashlib
import hmac
import os
import secrets
import time
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from uuid import uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, LargeBinary, String, and_, column, create_engine, false, func, or_, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


AUTH_LIFETIME = 7 * 24 * 60 * 60


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    salt: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    token_hash: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    expires_at: Mapped[int] = mapped_column(Integer, nullable=False)


class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(or_(
            and_(column("guest_id").is_not(None), column("user_id").is_(None)),
            and_(column("guest_id").is_(None), column("user_id").is_not(None)),
        )),
        Index("documents_guest", "guest_id"),
        Index("documents_user", "user_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    guest_id: Mapped[str | None] = mapped_column(String)
    user_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    relative_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


@lru_cache(maxsize=1)
def database_engine():
    return create_engine(URL.create(
        "postgresql+psycopg",
        username=os.getenv("POSTGRES_USER", "docusense"),
        password=os.getenv("POSTGRES_PASSWORD", "local-dev-only"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "docusense"),
    ), pool_pre_ping=True)


@contextmanager
def session_scope():
    with Session(database_engine(), expire_on_commit=False) as session, session.begin():
        yield session


def initialize_database():
    Base.metadata.create_all(database_engine())


def create_user(email: str, password: str):
    email = email.strip().lower()
    salt = secrets.token_bytes(16)
    password_hash = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
    user_id = uuid4().hex
    with session_scope() as session:
        session.add(User(id=user_id, email=email, password_hash=password_hash, salt=salt))
    return user_id, email


def authenticate(email: str, password: str):
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email.strip().lower()))
    if user and hmac.compare_digest(
        user.password_hash,
        hashlib.scrypt(password.encode(), salt=user.salt, n=16384, r=8, p=1),
    ):
        return user.id, user.email
    return None


def create_auth_session(user_id: str):
    token = secrets.token_urlsafe(32)
    with session_scope() as session:
        session.add(
            AuthSession(
                token_hash=hashlib.sha256(token.encode()).hexdigest(),
                user_id=user_id,
                expires_at=int(time.time()) + AUTH_LIFETIME,
            )
        )
    return token


def current_user(token: str | None):
    if not token:
        return None
    with session_scope() as session:
        user = session.scalar(
            select(User).join(AuthSession, User.id == AuthSession.user_id).where(
                AuthSession.token_hash == hashlib.sha256(token.encode()).hexdigest(),
                AuthSession.expires_at > int(time.time()),
            )
        )
        return {"id": user.id, "email": user.email} if user else None


def revoke_session(token: str | None):
    if token:
        with session_scope() as session:
            record = session.get(AuthSession, hashlib.sha256(token.encode()).hexdigest())
            if record:
                session.delete(record)


def save_document(document_id: str, guest_id: str | None, user_id: str | None, filename: str, relative_path: str):
    with session_scope() as session:
        session.add(
            DocumentRecord(
                id=document_id, guest_id=guest_id, user_id=user_id,
                filename=filename, relative_path=relative_path,
            )
        )


def owner_filter(guest_id: str | None, user_id: str | None):
    if guest_id is not None and user_id is None:
        return and_(DocumentRecord.guest_id == guest_id, DocumentRecord.user_id.is_(None))
    if user_id is not None and guest_id is None:
        return and_(DocumentRecord.user_id == user_id, DocumentRecord.guest_id.is_(None))
    return false()


def document_values(document: DocumentRecord):
    return {
        "id": document.id,
        "guest_id": document.guest_id,
        "user_id": document.user_id,
        "filename": document.filename,
        "relative_path": document.relative_path,
    }


def document_for_owner(document_id: str, guest_id: str | None, user_id: str | None):
    with session_scope() as session:
        document = session.scalar(
            select(DocumentRecord).where(DocumentRecord.id == document_id, owner_filter(guest_id, user_id))
        )
        return document_values(document) if document else None


def document_exists(document_id: str):
    with session_scope() as session:
        return session.get(DocumentRecord, document_id) is not None


def documents_for_owner(guest_id: str | None, user_id: str | None):
    with session_scope() as session:
        documents = session.scalars(
            select(DocumentRecord).where(owner_filter(guest_id, user_id)).order_by(DocumentRecord.created_at.desc(), DocumentRecord.id.desc())
        ).all()
        return [{"id": document.id, "filename": document.filename} for document in documents]


def remove_document(document_id: str):
    with session_scope() as session:
        document = session.get(DocumentRecord, document_id)
        if document:
            session.delete(document)


def guest_documents(guest_id: str):
    with session_scope() as session:
        documents = session.scalars(select(DocumentRecord).where(owner_filter(guest_id, None))).all()
        return [document_values(document) for document in documents]


def claim_document(document_id: str, guest_id: str, user_id: str):
    with session_scope() as session:
        document = session.scalar(
            select(DocumentRecord).where(DocumentRecord.id == document_id, owner_filter(guest_id, None))
        )
        if document:
            document.guest_id = None
            document.user_id = user_id
