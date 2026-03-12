import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
from models.models import User

router = APIRouter(prefix='/api/auth', tags=['auth'])

SESSION_TTL_HOURS = int(os.getenv('AUTH_SESSION_HOURS', '24'))
DEFAULT_ADMIN_USERNAME = os.getenv('DEFAULT_ADMIN_USERNAME', 'admin')
DEFAULT_ADMIN_PASSWORD = os.getenv('DEFAULT_ADMIN_PASSWORD', 'admin1234!')
DEFAULT_ADMIN_NAME = os.getenv('DEFAULT_ADMIN_NAME', '구이일')
DEFAULT_ADMIN_ROLE = os.getenv('DEFAULT_ADMIN_ROLE', '\uc18c\uc7a5')

_sessions: dict[str, dict] = {}


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    name: str
    role: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    user: UserResponse


class LogoutResponse(BaseModel):
    message: str


class BootstrapResponse(BaseModel):
    username: str
    name: str
    role: str


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 600000)
    return f'{salt}${digest.hex()}'


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split('$', 1)
    except ValueError:
        return False
    return secrets.compare_digest(_hash_password(password, salt), stored_hash)


def _cleanup_sessions() -> None:
    now = datetime.now(timezone.utc)
    expired_tokens = [token for token, payload in _sessions.items() if payload['expires_at'] <= now]
    for token in expired_tokens:
        _sessions.pop(token, None)


def _build_user_response(user: User) -> UserResponse:
    return UserResponse(id=user.id, username=user.username, name=user.name, role=user.role)


def seed_default_admin() -> None:
    db = SessionLocal()
    try:
        existing_user = db.query(User).filter(User.username == DEFAULT_ADMIN_USERNAME).first()
        if existing_user:
            if existing_user.name != DEFAULT_ADMIN_NAME:
                existing_user.name = DEFAULT_ADMIN_NAME
            if existing_user.role != DEFAULT_ADMIN_ROLE:
                existing_user.role = DEFAULT_ADMIN_ROLE
            db.commit()
            return

        if db.query(User).count() > 0:
            return

        user = User(
            username=DEFAULT_ADMIN_USERNAME,
            password_hash=_hash_password(DEFAULT_ADMIN_PASSWORD),
            name=DEFAULT_ADMIN_NAME,
            role=DEFAULT_ADMIN_ROLE,
            is_active=True,
        )
        db.add(user)
        db.commit()
    finally:
        db.close()


def _get_token_from_header(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail='Authorization header is required.')

    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or not token:
        raise HTTPException(status_code=401, detail='Bearer token is required.')

    return token


def _get_current_user(authorization: str | None, db: Session) -> User:
    _cleanup_sessions()
    token = _get_token_from_header(authorization)
    payload = _sessions.get(token)
    if not payload:
        raise HTTPException(status_code=401, detail='Session is invalid or expired.')

    user = db.query(User).filter(User.id == payload['user_id'], User.is_active.is_(True)).first()
    if not user:
        _sessions.pop(token, None)
        raise HTTPException(status_code=401, detail='User not found.')

    return user


@router.get('/bootstrap', response_model=BootstrapResponse)
def get_bootstrap_credentials():
    return BootstrapResponse(username=DEFAULT_ADMIN_USERNAME, name=DEFAULT_ADMIN_NAME, role=DEFAULT_ADMIN_ROLE)


@router.post('/login', response_model=LoginResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username.strip()).first()
    if not user or not _verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail='Invalid username or password.')

    if not user.is_active:
        raise HTTPException(status_code=403, detail='User is inactive.')

    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        'user_id': user.id,
        'expires_at': datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS),
    }
    return LoginResponse(access_token=token, user=_build_user_response(user))


@router.get('/me', response_model=UserResponse)
def get_me(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    user = _get_current_user(authorization, db)
    return _build_user_response(user)


@router.post('/logout', response_model=LogoutResponse)
def logout(authorization: str | None = Header(default=None)):
    token = _get_token_from_header(authorization)
    _sessions.pop(token, None)
    return LogoutResponse(message='Logged out.')
