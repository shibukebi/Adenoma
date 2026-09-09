#!/usr/bin/env python3
import argparse
import getpass

from .db import initialize_database, transaction, utc_now
from .security import hash_password


def main():
    parser = argparse.ArgumentParser(description="Manage challenge-review users")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("username")
    create.add_argument("--display-name")
    create.add_argument("--role", choices=["reviewer", "admin"], default="reviewer")
    create.add_argument("--password")

    reset = subparsers.add_parser("reset-password")
    reset.add_argument("username")
    reset.add_argument("--password")

    toggle = subparsers.add_parser("set-active")
    toggle.add_argument("username")
    toggle.add_argument("active", choices=["0", "1"])

    args = parser.parse_args()
    initialize_database()
    if args.command in {"create", "reset-password"}:
        password = args.password or getpass.getpass("Password: ")
        encoded = hash_password(password)

    with transaction() as connection:
        if args.command == "create":
            connection.execute(
                """
                INSERT INTO users(username, password_hash, display_name, role, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (args.username, encoded, args.display_name or args.username, args.role, utc_now()),
            )
        elif args.command == "reset-password":
            cursor = connection.execute(
                "UPDATE users SET password_hash = ? WHERE username = ? COLLATE NOCASE",
                (encoded, args.username),
            )
            if cursor.rowcount != 1:
                raise SystemExit(f"Unknown user: {args.username}")
        else:
            cursor = connection.execute(
                "UPDATE users SET active = ? WHERE username = ? COLLATE NOCASE",
                (int(args.active), args.username),
            )
            if cursor.rowcount != 1:
                raise SystemExit(f"Unknown user: {args.username}")
    print(f"User operation completed: {args.username}")


if __name__ == "__main__":
    main()
