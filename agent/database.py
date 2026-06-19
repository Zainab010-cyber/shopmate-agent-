"""SQLite database initialization and query helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "campus_shop.db"
SEED_SQL = DATA_DIR / "seed_orders.sql"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the Campus Shop SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create and seed the database if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SEED_SQL.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_SQL}")

    with get_connection() as conn:
        conn.executescript(SEED_SQL.read_text(encoding="utf-8"))
        conn.commit()


def lookup_order(order_id: int) -> dict | None:
    """Fetch order details joined with customer info."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT o.id, o.status, o.items, o.tracking_number, o.order_date,
                   o.total_amount, c.name AS customer_name, c.email AS customer_email
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE o.id = ?
            """,
            (order_id,),
        ).fetchone()
    return dict(row) if row else None


def lookup_orders_by_email(email: str) -> list[dict]:
    """Fetch all orders for a customer email."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT o.id, o.status, o.items, o.tracking_number, o.order_date, o.total_amount
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE LOWER(c.email) = LOWER(?)
            ORDER BY o.order_date DESC
            """,
            (email.strip(),),
        ).fetchall()
    return [dict(r) for r in rows]


def create_escalation(
    reason: str,
    transcript: str,
    order_id: int | None = None,
    customer_email: str | None = None,
) -> dict:
    """Record a support escalation ticket."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO escalations (reason, order_id, customer_email, transcript)
            VALUES (?, ?, ?, ?)
            """,
            (reason, order_id, customer_email, transcript),
        )
        conn.commit()
        ticket_id = cursor.lastrowid
    return {
        "ticket_id": ticket_id,
        "reason": reason,
        "order_id": order_id,
        "customer_email": customer_email,
        "message": f"Escalation ticket #{ticket_id} created. A human agent will respond within 4 business hours.",
    }
