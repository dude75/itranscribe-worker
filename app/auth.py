"""Bearer-авторизация. Токен не писать в логи."""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.schemas import ErrorCode, error_payload

_bearer = HTTPBearer(auto_error=False)


def api_token_is_valid(authorization: str | None) -> bool:
    expected = get_settings().API_TOKEN
    if not authorization or not expected:
        return False
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    return hmac.compare_digest(parts[1], expected)


def require_api_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    header = f"{creds.scheme} {creds.credentials}" if creds is not None else None
    if not api_token_is_valid(header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_payload(ErrorCode.unauthorized),
        )
    return creds.credentials if creds is not None else ""
