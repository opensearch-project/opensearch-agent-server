# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.auth_middleware.

Exercises the AuthenticationMiddleware dispatch chain (health short-circuit,
auth-disabled, header/token/apikey success, strict 401, permissive fallback),
each per-strategy authenticator (header, JWT with HS256/RS256/expired/invalid/
missing-secret/unsupported-algorithm/missing-claim, API key JSON parsing and
lookup branches), and create_auth_middleware config parsing.

Hermetic + deterministic: fake Starlette Request objects (SimpleNamespace),
JWTs generated in-test with PyJWT (HS256 + an in-test RSA keypair), and a
SimpleNamespace config double so the real config singleton is never touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.responses import JSONResponse

from server.auth_middleware import (
    AuthenticationMiddleware,
    AuthMode,
    AuthStrategy,
    create_auth_middleware,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_request(*, headers=None, path="/runs", method="POST", client_host="1.2.3.4"):
    return SimpleNamespace(
        headers=headers or {},
        url=SimpleNamespace(path=path),
        method=method,
        state=SimpleNamespace(),
        client=SimpleNamespace(host=client_host),
    )


def _jwt_config(
    *,
    algorithm="HS256",
    secret="test-secret",
    public_key=None,
    user_id_claim="sub",
    api_keys=None,
):
    return SimpleNamespace(
        jwt_algorithm=algorithm,
        jwt_secret=secret,
        jwt_public_key=public_key,
        jwt_user_id_claim=user_id_claim,
        api_keys=api_keys,
    )


def _mw(**kwargs):
    """Build middleware with a fake config, avoiding get_config()."""
    kwargs.setdefault("config", _jwt_config())
    return AuthenticationMiddleware(app=SimpleNamespace(), **kwargs)


async def _call_next_ok(request):
    return SimpleNamespace(marker="next-response")


def _rsa_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return priv_pem, pub_pem


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    async def test_health_short_circuits(self):
        mw = _mw(enabled=True, mode=AuthMode.STRICT)
        req = _make_request(path="/health")
        resp = await mw.dispatch(req, _call_next_ok)
        assert resp.marker == "next-response"

    async def test_auth_disabled_allows(self):
        mw = _mw(enabled=False)
        req = _make_request(headers={})
        resp = await mw.dispatch(req, _call_next_ok)
        assert resp.marker == "next-response"

    async def test_header_success_sets_state(self):
        mw = _mw(enabled=True, strategies=[AuthStrategy.HEADER])
        req = _make_request(headers={"X-User-Id": "alice"})
        resp = await mw.dispatch(req, _call_next_ok)
        assert resp.marker == "next-response"
        assert req.state.user_id == "alice"
        assert req.state.authenticated is True

    async def test_strict_unauthenticated_returns_401(self):
        mw = _mw(enabled=True, mode=AuthMode.STRICT, strategies=[AuthStrategy.HEADER])
        req = _make_request(headers={})
        resp = await mw.dispatch(req, _call_next_ok)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 401

    async def test_permissive_unauthenticated_falls_back(self):
        mw = _mw(
            enabled=True, mode=AuthMode.PERMISSIVE, strategies=[AuthStrategy.HEADER]
        )
        req = _make_request(headers={})
        resp = await mw.dispatch(req, _call_next_ok)
        assert resp.marker == "next-response"
        # fallback user id derived from client host
        assert req.state.user_id == "client_1.2.3.4"

    async def test_token_strategy_success(self):
        cfg = _jwt_config(algorithm="HS256", secret="s3cr3t", user_id_claim="sub")
        mw = _mw(enabled=True, strategies=[AuthStrategy.TOKEN], config=cfg)
        token = jwt.encode({"sub": "bob"}, "s3cr3t", algorithm="HS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        resp = await mw.dispatch(req, _call_next_ok)
        assert resp.marker == "next-response"
        assert req.state.user_id == "bob"

    async def test_apikey_strategy_success(self):
        cfg = _jwt_config(api_keys=json.dumps({"key-123": "carol"}))
        mw = _mw(enabled=True, strategies=[AuthStrategy.API_KEY], config=cfg)
        req = _make_request(headers={"X-API-Key": "key-123"})
        resp = await mw.dispatch(req, _call_next_ok)
        assert resp.marker == "next-response"
        assert req.state.user_id == "carol"


# ---------------------------------------------------------------------------
# _authenticate_header
# ---------------------------------------------------------------------------


class TestAuthenticateHeader:
    def test_present(self):
        mw = _mw()
        req = _make_request(headers={"X-User-Id": "dave"})
        assert mw._authenticate_header(req) == "dave"

    def test_absent(self):
        mw = _mw()
        assert mw._authenticate_header(_make_request(headers={})) is None


# ---------------------------------------------------------------------------
# _authenticate_token
# ---------------------------------------------------------------------------


class TestAuthenticateToken:
    def test_no_authorization_header(self):
        mw = _mw()
        assert mw._authenticate_token(_make_request(headers={})) is None

    def test_non_bearer_scheme(self):
        mw = _mw()
        req = _make_request(headers={"Authorization": "Basic abc"})
        assert mw._authenticate_token(req) is None

    def test_hs256_missing_secret(self):
        cfg = _jwt_config(algorithm="HS256", secret=None)
        mw = _mw(config=cfg)
        req = _make_request(headers={"Authorization": "Bearer whatever"})
        assert mw._authenticate_token(req) is None

    def test_rs256_missing_public_key(self):
        cfg = _jwt_config(algorithm="RS256", secret=None, public_key=None)
        mw = _mw(config=cfg)
        req = _make_request(headers={"Authorization": "Bearer whatever"})
        assert mw._authenticate_token(req) is None

    def test_unsupported_algorithm(self):
        cfg = _jwt_config(algorithm="ES256")
        mw = _mw(config=cfg)
        req = _make_request(headers={"Authorization": "Bearer whatever"})
        assert mw._authenticate_token(req) is None

    def test_hs256_valid_custom_claim(self):
        cfg = _jwt_config(algorithm="HS256", secret="k", user_id_claim="email")
        mw = _mw(config=cfg)
        token = jwt.encode({"email": "e@x.com"}, "k", algorithm="HS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        assert mw._authenticate_token(req) == "e@x.com"

    def test_hs256_valid_user_id_fallback(self):
        # claim configured to a key that is absent -> falls back to user_id
        cfg = _jwt_config(algorithm="HS256", secret="k", user_id_claim="missing")
        mw = _mw(config=cfg)
        token = jwt.encode({"user_id": "uid-9"}, "k", algorithm="HS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        assert mw._authenticate_token(req) == "uid-9"

    def test_hs256_valid_sub_fallback(self):
        cfg = _jwt_config(algorithm="HS256", secret="k", user_id_claim="missing")
        mw = _mw(config=cfg)
        token = jwt.encode({"sub": "subject-7"}, "k", algorithm="HS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        assert mw._authenticate_token(req) == "subject-7"

    def test_rs256_valid(self):
        priv, pub = _rsa_keypair()
        cfg = _jwt_config(
            algorithm="RS256", secret=None, public_key=pub, user_id_claim="sub"
        )
        mw = _mw(config=cfg)
        token = jwt.encode({"sub": "rsa-user"}, priv, algorithm="RS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        assert mw._authenticate_token(req) == "rsa-user"

    def test_missing_user_id_claim(self):
        cfg = _jwt_config(algorithm="HS256", secret="k", user_id_claim="sub")
        mw = _mw(config=cfg)
        token = jwt.encode({"other": "x"}, "k", algorithm="HS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        assert mw._authenticate_token(req) is None

    def test_expired_token(self):
        cfg = _jwt_config(algorithm="HS256", secret="k", user_id_claim="sub")
        mw = _mw(config=cfg)
        token = jwt.encode({"sub": "u", "exp": 1}, "k", algorithm="HS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        assert mw._authenticate_token(req) is None

    def test_invalid_signature(self):
        cfg = _jwt_config(algorithm="HS256", secret="right-key", user_id_claim="sub")
        mw = _mw(config=cfg)
        token = jwt.encode({"sub": "u"}, "wrong-key", algorithm="HS256")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})
        assert mw._authenticate_token(req) is None

    def test_decode_error(self):
        cfg = _jwt_config(algorithm="HS256", secret="k", user_id_claim="sub")
        mw = _mw(config=cfg)
        req = _make_request(headers={"Authorization": "Bearer not.a.jwt"})
        assert mw._authenticate_token(req) is None

    def test_generic_exception(self):
        # jwt_algorithm without .upper() -> AttributeError caught by broad except
        cfg = _jwt_config(algorithm=123, secret="k", user_id_claim="sub")
        mw = _mw(config=cfg)
        req = _make_request(headers={"Authorization": "Bearer abc"})
        assert mw._authenticate_token(req) is None


# ---------------------------------------------------------------------------
# _authenticate_api_key
# ---------------------------------------------------------------------------


class TestAuthenticateApiKey:
    def test_no_config(self):
        cfg = _jwt_config(api_keys=None)
        mw = _mw(config=cfg)
        req = _make_request(headers={"X-API-Key": "k"})
        assert mw._authenticate_api_key(req) is None

    def test_invalid_json(self):
        cfg = _jwt_config(api_keys="{not json")
        mw = _mw(config=cfg)
        req = _make_request(headers={"X-API-Key": "k"})
        assert mw._authenticate_api_key(req) is None

    def test_non_dict_json(self):
        cfg = _jwt_config(api_keys=json.dumps(["a", "b"]))
        mw = _mw(config=cfg)
        req = _make_request(headers={"X-API-Key": "k"})
        assert mw._authenticate_api_key(req) is None

    def test_x_api_key_valid(self):
        cfg = _jwt_config(api_keys=json.dumps({"secret-key-long": "u1"}))
        mw = _mw(config=cfg)
        req = _make_request(headers={"X-API-Key": "secret-key-long"})
        assert mw._authenticate_api_key(req) == "u1"

    def test_authorization_apikey_scheme(self):
        cfg = _jwt_config(api_keys=json.dumps({"abc12345xyz": "u2"}))
        mw = _mw(config=cfg)
        req = _make_request(headers={"Authorization": "ApiKey abc12345xyz"})
        assert mw._authenticate_api_key(req) == "u2"

    def test_no_key_present(self):
        cfg = _jwt_config(api_keys=json.dumps({"k": "u"}))
        mw = _mw(config=cfg)
        req = _make_request(headers={})
        assert mw._authenticate_api_key(req) is None

    def test_invalid_key_short_prefix(self):
        cfg = _jwt_config(api_keys=json.dumps({"known": "u"}))
        mw = _mw(config=cfg)
        req = _make_request(headers={"X-API-Key": "short"})
        assert mw._authenticate_api_key(req) is None

    def test_caches_parsed_map(self):
        cfg = _jwt_config(api_keys=json.dumps({"longenoughkey": "u3"}))
        mw = _mw(config=cfg)
        req = _make_request(headers={"X-API-Key": "longenoughkey"})
        assert mw._authenticate_api_key(req) == "u3"
        # second call uses cached _api_keys_map (config unchanged)
        assert mw._authenticate_api_key(req) == "u3"


# ---------------------------------------------------------------------------
# __init__ strategy normalization
# ---------------------------------------------------------------------------


class TestInit:
    def test_none_strategies_defaults_to_header(self):
        mw = _mw(strategies=None)
        assert mw.strategies == [AuthStrategy.HEADER]

    def test_all_none_entries_defaults_to_header(self):
        mw = _mw(strategies=[None, None])
        assert mw.strategies == [AuthStrategy.HEADER]


# ---------------------------------------------------------------------------
# create_auth_middleware
# ---------------------------------------------------------------------------


def _cfg(*, auth_enabled=True, auth_mode="strict", auth_strategies="header"):
    return SimpleNamespace(
        auth_enabled=auth_enabled,
        auth_mode=auth_mode,
        auth_strategies=auth_strategies,
    )


class TestCreateAuthMiddleware:
    def test_enabled_default_header(self):
        result = create_auth_middleware(SimpleNamespace(), _cfg())
        assert result is not None
        assert result["enabled"] is True
        assert result["mode"] == AuthMode.STRICT
        assert result["strategies"] == [AuthStrategy.HEADER]

    def test_disabled_returns_none(self):
        assert (
            create_auth_middleware(SimpleNamespace(), _cfg(auth_enabled=False)) is None
        )

    def test_invalid_mode_defaults_strict(self):
        result = create_auth_middleware(SimpleNamespace(), _cfg(auth_mode="bogus"))
        assert result["mode"] == AuthMode.STRICT

    def test_permissive_mode(self):
        result = create_auth_middleware(SimpleNamespace(), _cfg(auth_mode="permissive"))
        assert result["mode"] == AuthMode.PERMISSIVE

    def test_invalid_strategy_skipped_defaults_header(self):
        result = create_auth_middleware(
            SimpleNamespace(), _cfg(auth_strategies="bogus,alsobad")
        )
        assert result["strategies"] == [AuthStrategy.HEADER]

    def test_multiple_valid_strategies(self):
        result = create_auth_middleware(
            SimpleNamespace(), _cfg(auth_strategies="header,token,apikey")
        )
        assert result["strategies"] == [
            AuthStrategy.HEADER,
            AuthStrategy.TOKEN,
            AuthStrategy.API_KEY,
        ]

    def test_config_none_uses_get_config(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "server.auth_middleware.get_config",
                lambda: _cfg(auth_enabled=False),
                raising=True,
            )
            assert create_auth_middleware(SimpleNamespace()) is None
