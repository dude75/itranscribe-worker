"""Bearer-авторизация. Токен не писать в логи."""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

_bearer = HTTPBearer(auto_error=False)

UNAUTHORIZED_BODY = {
    "status": "error",
    "error": {"code": "unauthorized"},
}


def require_api_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    expected = get_settings().API_TOKEN
    provided = creds.credentials if creds is not None else ""
    scheme_ok = creds is not None and creds.scheme.lower() == "bearer"
    token_ok = bool(expected) and hmac.compare_digest(provided, expected)
    if not scheme_ok or not token_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=UNAUTHORIZED_BODY,
        )
    return provided
