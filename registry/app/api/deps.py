from fastapi import Header, HTTPException, Depends
from jose import jwt, JWTError
from app.core.config import settings
from app.db.session import SessionLocal


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_actor(authorization: str | None = Header(default=None)) -> str:
    """
    если ALLOW_ANON_WRITE=true и нет Authorization, то actor="anonymous"
    если есть Bearer токен, то берём sub
    """
    if not authorization:
        if settings.allow_anon_write:
            return "anonymous"
        raise HTTPException(status_code=401, detail="Authorization required")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token (no sub)")
        return str(sub)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")