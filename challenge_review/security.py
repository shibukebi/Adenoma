from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

from .config import SESSION_HOURS
from .db import transaction, utc_now


SCRYPT_N = 2**14


def hash_password(password):
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=8, p=1)
    return f"scrypt${SCRYPT_N}${salt.hex()}${derived.hex()}"


def verify_password(password, encoded):
    try:
        algorithm, n_value, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=int(n_value), r=8, p=1
        )
        return hmac.compare_digest(derived, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


def token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id):
    raw_token = secrets.token_urlsafe(40)
    expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    with transaction() as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (utc_now(),))
        connection.execute(
            "INSERT INTO sessions(token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token_hash(raw_token), user_id, expires.isoformat(), utc_now()),
        )
    return raw_token, expires


def delete_session(raw_token):
    if not raw_token:
        return
    with transaction() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(raw_token),))


def get_session_user(raw_token):
    if not raw_token:
        return None
    with transaction() as connection:
        row = connection.execute(
            """
            SELECT u.id, u.username, u.display_name, u.role, u.active
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ? AND u.active = 1
            """,
            (token_hash(raw_token), utc_now()),
        ).fetchone()
    return dict(row) if row else None
