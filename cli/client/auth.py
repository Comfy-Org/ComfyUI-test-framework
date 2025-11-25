"""Firebase authentication for Comfy Cloud."""

import json
import time
from dataclasses import dataclass
from urllib import error, request

from pydantic import BaseModel


class FirebaseAuthRequest(BaseModel):
    """Firebase sign-in request."""

    email: str
    password: str
    returnSecureToken: bool = True
    clientType: str = "CLIENT_TYPE_WEB"


class FirebaseAuthResponse(BaseModel):
    """Firebase sign-in response."""

    idToken: str
    localId: str
    refreshToken: str
    expiresIn: str

    model_config = {"extra": "allow"}


@dataclass
class AuthToken:
    """Authenticated token with expiration."""

    token: str
    user_id: str
    refresh_token: str
    expires_at: float  # Unix timestamp

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def needs_refresh(self) -> bool:
        return time.time() >= (self.expires_at - 300)  # 5 min buffer


class FirebaseAuth:
    """Firebase authentication provider."""

    FIREBASE_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

    def __init__(self, api_key: str, email: str, password: str):
        self._api_key = api_key
        self._email = email
        self._password = password
        self._token: AuthToken | None = None
        self._session_cookie: str | None = None

    @property
    def token(self) -> AuthToken | None:
        return self._token

    def authenticate(self) -> AuthToken:
        """Sign in with Firebase."""
        from .errors import FirebaseAuthError, InvalidCredentialsError

        url = f"{self.FIREBASE_URL}?key={self._api_key}"
        payload = FirebaseAuthRequest(email=self._email, password=self._password)

        try:
            data = json.dumps(payload.model_dump()).encode()
            req = request.Request(url, data=data, headers={"Content-Type": "application/json"})

            with request.urlopen(req) as resp:
                result = FirebaseAuthResponse.model_validate_json(resp.read())

            self._token = AuthToken(
                token=result.idToken,
                user_id=result.localId,
                refresh_token=result.refreshToken,
                expires_at=time.time() + int(result.expiresIn) - 60,
            )
            return self._token

        except error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            if "INVALID" in body or "EMAIL_NOT_FOUND" in body:
                raise InvalidCredentialsError(f"Invalid credentials: {body}") from e
            raise FirebaseAuthError(f"Firebase auth failed: {body}") from e

    def create_session(self, base_url: str) -> None:
        """Create session cookie via /api/auth/session."""
        from .errors import AuthenticationError

        if not self._token:
            raise AuthenticationError("Not authenticated")

        url = f"{base_url.rstrip('/')}/api/auth/session"
        payload = json.dumps({"id_token": self._token.token}).encode()

        req = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token.token}",
            },
        )

        try:
            with request.urlopen(req) as resp:
                set_cookie = resp.headers.get("Set-Cookie", "")
                # Extract just the cookie value (before any attributes like Max-Age, HttpOnly, etc.)
                # Format: "session=value; Max-Age=7200; HttpOnly; ..."
                # We only want: "session=value"
                if set_cookie:
                    cookie_part = set_cookie.split(";")[0].strip()
                    self._session_cookie = cookie_part
                else:
                    self._session_cookie = None
        except error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise AuthenticationError(
                f"Session creation failed ({e.code}): {body[:200]}"
            ) from e

    def get_headers(self) -> dict[str, str]:
        """Get authentication headers."""
        if not self._token:
            return {}
        headers = {"Authorization": f"Bearer {self._token.token}"}
        if self._session_cookie:
            headers["Cookie"] = self._session_cookie
        return headers

    def ensure_valid(self, base_url: str) -> None:
        """Re-authenticate if token needs refresh."""
        if not self._token or self._token.needs_refresh:
            self.authenticate()
            self.create_session(base_url)
