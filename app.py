"""
LifecycleBot - Joiner-mover-leaver automation lab.

All application code is intentionally kept in this single file.
The app is local-first, demo-first, and safe by default.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio.v2 as imageio
except Exception:
    import imageio  # type: ignore


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent if APP_DIR.name == "src" else APP_DIR
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
CSV_PATH = DATA_DIR / "hr_events.csv"
YAML_PATH = DATA_DIR / "role_matrix.yaml"
DB_PATH = DATA_DIR / "lifecyclebot_lab.sqlite3"

SYSTEM_ORDER = ["directory", "email", "chat", "code_host", "ticketing"]
CONTROL_ORDER = ["session", "mfa", "mailbox_forwarding", "group", "license"]

SYSTEM_ABBR = {
    "directory": "DIR",
    "email": "MAIL",
    "chat": "CHAT",
    "code_host": "CODE",
    "ticketing": "TICK",
    "session": "SESS",
    "mfa": "MFA",
    "mailbox_forwarding": "FWD",
    "group": "GRP",
    "license": "LIC",
    "all": "ALL",
}

LEVEL_COLORS = {
    "info": (0, 229, 255),
    "ok": (57, 255, 136),
    "warn": (255, 191, 0),
    "crit": (255, 59, 92),
}

BG_COLOR = (7, 9, 15)
PANEL_COLOR = (14, 19, 30)
TEXT_COLOR = (220, 231, 245)
MUTED_COLOR = (138, 160, 180)
CYAN = (0, 229, 255)
GREEN = (57, 255, 136)
AMBER = (255, 191, 0)
RED = (255, 59, 92)
GRAY = (90, 105, 120)

AS_OF = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
GRACE_DAYS = 14

DEMO_PERSONAS = {
    "EMP-1016": "joiner",
    "EMP-1002": "mover",
    "EMP-1022": "leaver",
}

JSON_EXCLUDE_KEYS = {
    "conn",
    "on_event",
    "api_key",
    "role_matrix",
    "access_state",
    "projection",
    "projection_roles",
    "events_df",
}

KPI_LABELS = {
    "provisioning_accuracy": "Provision Accuracy",
    "mean_time_to_revoke_hours": "Mean Revoke Hours",
    "creep_detection_precision": "Creep Precision",
    "orphan_detection_rate": "Orphan Rate",
}

PERCENT_KPI_KEYS = {
    "provisioning_accuracy",
    "creep_detection_precision",
    "orphan_detection_rate",
}


# -----------------------------------------------------------------------------
# Embedded synthetic data
# -----------------------------------------------------------------------------

ROLE_MATRIX_YAML = """
version: 1
systems:
  - directory
  - email
  - chat
  - code_host
  - ticketing
roles:
  software_engineer:
    directory:
      - domain_user
      - ou_engineering
    email:
      - mailbox_standard
      - dist_engineering
    chat:
      - chat_user
      - channel_engineering
    code_host:
      - repo_readonly
      - ci_trigger
    ticketing:
      - ticket_reporter
      - queue_engineering_triage
  data_analyst:
    directory:
      - domain_user
      - ou_analytics
    email:
      - mailbox_standard
      - dist_analytics
    chat:
      - chat_user
      - channel_analytics
    code_host:
      - repo_readonly
    ticketing:
      - ticket_reporter
      - queue_analytics_triage
  operations_manager:
    directory:
      - domain_user
      - ou_operations
      - manager_group
    email:
      - mailbox_standard
      - dist_operations
      - dist_managers
    chat:
      - chat_user
      - channel_operations
      - channel_leadership
    code_host:
      - repo_readonly
    ticketing:
      - ticket_reporter
      - queue_operations_approval
      - ticket_approver
  finance_analyst:
    directory:
      - domain_user
      - ou_finance
    email:
      - mailbox_standard
      - dist_finance
    chat:
      - chat_user
      - channel_finance
    code_host: []
    ticketing:
      - ticket_reporter
      - queue_finance_invoice
  it_admin:
    directory:
      - domain_user
      - ou_it
      - privileged_helpdesk
    email:
      - mailbox_standard
      - dist_it
    chat:
      - chat_user
      - channel_it
      - channel_incident
    code_host:
      - repo_readonly
      - repo_admin_infra
    ticketing:
      - ticket_reporter
      - queue_it_admin
      - ticket_admin
  contractor:
    directory:
      - contractor_user
      - ou_contractor
    email:
      - mailbox_temporary
    chat:
      - chat_guest
      - channel_contractor
    code_host:
      - repo_readonly_limited
    ticketing:
      - ticket_reporter
"""


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none"}:
        return ""
    return text


def to_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.to_pydatetime()
    text = _clean_str(value)
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def to_iso(value: Any) -> str:
    dt = to_dt(value)
    if dt is not None:
        return dt.isoformat()
    return _clean_str(value)


def make_json_safe(obj: Any, exclude_keys: Optional[set] = None) -> Any:
    if exclude_keys is None:
        exclude_keys = JSON_EXCLUDE_KEYS
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            key_text = str(key)
            if key_text in exclude_keys:
                continue
            try:
                out[key_text] = make_json_safe(value, exclude_keys)
            except Exception:
                out[key_text] = str(value)
        return out
    if isinstance(obj, set):
        try:
            return sorted((make_json_safe(item, exclude_keys) for item in obj), key=str)
        except Exception:
            return [str(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(item, exclude_keys) for item in obj]
    if isinstance(obj, list):
        return [make_json_safe(item, exclude_keys) for item in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if callable(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def sanitize_for_log(kpi: Any) -> Any:
    return make_json_safe(kpi or {})


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def type_matches(value: Any, typ: str) -> bool:
    if typ == "any":
        return True
    if typ == "string":
        return isinstance(value, str)
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "array":
        return isinstance(value, list)
    if typ == "object":
        return isinstance(value, dict)
    return True


def validate_schema(data: Any, schema: Optional[Dict[str, Any]]) -> bool:
    if not schema:
        return True
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return False
    for field in schema.get("required", []):
        if field not in data:
            return False
    for field, typ in schema.get("properties", {}).items():
        if field in data and not type_matches(data[field], typ):
            return False
    return True


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, size: int = 12) -> str:
    return sha256_text(text)[:size]


# -----------------------------------------------------------------------------
# Data generation and loading
# -----------------------------------------------------------------------------

def generate_hr_events_df() -> pd.DataFrame:
    roles = [
        "software_engineer",
        "data_analyst",
        "operations_manager",
        "finance_analyst",
        "it_admin",
        "contractor",
    ]
    departments = {
        "software_engineer": "Engineering",
        "data_analyst": "Analytics",
        "operations_manager": "Operations",
        "finance_analyst": "Finance",
        "it_admin": "IT",
        "contractor": "Contractor Services",
    }
    first_names = [
        "Ada", "Grace", "Alan", "Katherine", "Linus", "Margaret", "Dennis", "Radia",
        "Tim", "Barbara", "Ken", "Annie", "Donald", "Frances", "John", "Jean",
        "Vint", "Shafi", "Martin", "Leslie", "Guido", "Bjarne", "Denise", "Sari",
        "Peter", "Whitfield", "Adele", "Ronald", "Ivan", "Mary"
    ]
    last_names = [
        "Lovelace", "Hopper", "Turing", "Johnson", "Torvalds", "Hamilton", "Ritchie",
        "Perlman", "Berners", "Liskov", "Thompson", "Easley", "Knuth", "Allen",
        "Backus", "Sammet", "Cerf", "Goldwasser", "Hellman", "Valiant", "Rossum",
        "Stroustrup", "Engert", "Nile", "Naur", "Diffie", "Goldberg", "Rivest",
        "Sutherland", "Jackson"
    ]
    locations = ["Remote-US", "Boston", "Austin", "Berlin", "Toronto"]
    managers = ["EMP-9001", "EMP-9002", "EMP-9003", "EMP-9004", "EMP-9005"]

    rows: List[Dict[str, Any]] = []
    base = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)

    # 30 hires
    for i in range(1, 31):
        emp = f"EMP-{1000 + i}"
        role = roles[(i - 1) % len(roles)]
        received = base + timedelta(hours=2 * i)
        effective = received + timedelta(days=1)
        rows.append(
            {
                "event_id": f"EV-HIR-{i:03d}",
                "received_at": received.isoformat(),
                "effective_date": effective.isoformat(),
                "employee_id": emp,
                "full_name": f"{first_names[(i - 1) % len(first_names)]} {last_names[(i * 7) % len(last_names)]}",
                "event_type": "hire",
                "role": role,
                "previous_role": "",
                "department": departments[role],
                "manager": managers[(i - 1) % len(managers)],
                "location": locations[(i - 1) % len(locations)],
                "employment_status": "active",
                "source_system": "HRIS-SYNTH",
                "notes": "Synthetic onboarding event",
            }
        )

    # 15 transfers
    for i in range(1, 16):
        emp = f"EMP-{1000 + i}"
        old_role = roles[(i - 1) % len(roles)]
        new_role = roles[i % len(roles)]
        received = base + timedelta(days=19, hours=i)
        effective = received + timedelta(days=2)
        rows.append(
            {
                "event_id": f"EV-MOV-{i:03d}",
                "received_at": received.isoformat(),
                "effective_date": effective.isoformat(),
                "employee_id": emp,
                "full_name": f"{first_names[(i - 1) % len(first_names)]} {last_names[(i * 7) % len(last_names)]}",
                "event_type": "transfer",
                "role": new_role,
                "previous_role": old_role,
                "department": departments[new_role],
                "manager": managers[(i + 1) % len(managers)],
                "location": locations[(i + 2) % len(locations)],
                "employment_status": "active",
                "source_system": "HRIS-SYNTH",
                "notes": "Synthetic internal move",
            }
        )

    # 10 exits
    for i in range(21, 31):
        emp = f"EMP-{1000 + i}"
        role = roles[(i - 1) % len(roles)]
        received = base + timedelta(days=30, hours=i)
        effective = received
        rows.append(
            {
                "event_id": f"EV-LEA-{i - 20:03d}",
                "received_at": received.isoformat(),
                "effective_date": effective.isoformat(),
                "employee_id": emp,
                "full_name": f"{first_names[(i - 1) % len(first_names)]} {last_names[(i * 7) % len(last_names)]}",
                "event_type": "exit",
                "role": role,
                "previous_role": "",
                "department": departments[role],
                "manager": managers[(i + 3) % len(managers)],
                "location": locations[(i + 4) % len(locations)],
                "employment_status": "terminated",
                "source_system": "HRIS-SYNTH",
                "notes": "Synthetic offboarding event",
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["received_at", "event_id"]).reset_index(drop=True)


def required_event_columns() -> List[str]:
    return [
        "event_id",
        "received_at",
        "effective_date",
        "employee_id",
        "full_name",
        "event_type",
        "role",
        "previous_role",
        "department",
        "manager",
        "location",
        "employment_status",
        "source_system",
        "notes",
    ]


def validate_events_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return generate_hr_events_df()
    df = df.copy()
    for col in required_event_columns():
        if col not in df.columns:
            df[col] = ""
    df["event_id"] = df["event_id"].astype(str).str.strip()
    df.loc[df["event_id"] == "", "event_id"] = [f"EV-AUTO-{uuid.uuid4().hex[:8]}" for _ in range(len(df.loc[df['event_id'] == '']))]
    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["role"] = df["role"].astype(str).str.strip()
    df["previous_role"] = df["previous_role"].astype(str).str.strip()
    df["employment_status"] = df["employment_status"].astype(str).str.lower().str.strip()
    return df


def ensure_data_files() -> None:
    ensure_dirs()
    try:
        if not YAML_PATH.exists():
            YAML_PATH.write_text(ROLE_MATRIX_YAML, encoding="utf-8")
    except Exception:
        pass
    try:
        if not CSV_PATH.exists():
            generate_hr_events_df().to_csv(CSV_PATH, index=False)
    except Exception:
        pass


def load_role_matrix_embedded() -> Dict[str, Any]:
    return yaml.safe_load(ROLE_MATRIX_YAML)


def load_role_matrix() -> Dict[str, Any]:
    try:
        if YAML_PATH.exists():
            text = YAML_PATH.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if isinstance(data, dict) and "roles" in data:
                return data
    except Exception:
        pass
    return load_role_matrix_embedded()


def load_events(mode: str) -> pd.DataFrame:
    if mode == "Demo":
        return generate_hr_events_df()
    ensure_data_files()
    try:
        df = pd.read_csv(CSV_PATH)
        df = validate_events_df(df)
        if df.empty:
            return generate_hr_events_df()
        return df
    except Exception:
        return generate_hr_events_df()


# -----------------------------------------------------------------------------
# SQLite lab store and hash-chained audit
# -----------------------------------------------------------------------------

def get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            employee_id TEXT PRIMARY KEY,
            full_name TEXT,
            role TEXT,
            department TEXT,
            status TEXT,
            source TEXT,
            is_shared INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS grants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            system TEXT NOT NULL,
            entitlement TEXT NOT NULL,
            status TEXT NOT NULL,
            source_event TEXT,
            granted_at TEXT,
            revoked_at TEXT,
            UNIQUE(employee_id, system, entitlement)
        );

        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT NOT NULL,
            status TEXT NOT NULL,
            detail TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL
        );
        """
    )
    conn.commit()


def seed_lab(conn: sqlite3.Connection) -> None:
    init_db(conn)
    cur = conn.execute("SELECT COUNT(*) AS c FROM users")
    row = cur.fetchone()
    if row and row["c"] > 0:
        return
    ts = now_iso()
    conn.execute(
        """
        INSERT OR IGNORE INTO users
        (employee_id, full_name, role, department, status, source, is_shared, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("EMP-9999", "Orphan Lab Account", "unknown", "Unassigned", "active", "seed-orphan", 0, ts, ts),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO users
        (employee_id, full_name, role, department, status, source, is_shared, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("SVC-LEGACY", "Legacy Shared Service", "service", "IT", "active", "seed-orphan", 1, ts, ts),
    )
    for emp, system, ent in [
        ("EMP-9999", "chat", "channel_all_staff"),
        ("EMP-9999", "ticketing", "queue_legacy_admin"),
        ("SVC-LEGACY", "code_host", "repo_admin_infra"),
        ("SVC-LEGACY", "email", "mailbox_shared"),
    ]:
        conn.execute(
            """
            INSERT OR IGNORE INTO grants
            (employee_id, system, entitlement, status, source_event, granted_at, revoked_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (emp, system, ent, "active", "SEED-LAB", ts),
        )
    conn.commit()


def audit_append(
    conn: Optional[sqlite3.Connection],
    actor: str,
    action: str,
    target: str,
    status: str,
    detail: Any = "",
) -> str:
    if conn is None:
        return ""
    try:
        cur = conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        prev_hash = row["hash"] if row else "GENESIS"
        ts = now_iso()
        detail_text = json.dumps(make_json_safe(detail), sort_keys=True, default=str)
        payload = json.dumps(
            {
                "ts": ts,
                "actor": actor,
                "action": action,
                "target": target,
                "status": status,
                "detail": detail_text,
            },
            sort_keys=True,
        )
        current_hash = sha256_text(prev_hash + payload)
        conn.execute(
            """
            INSERT INTO audit (ts, actor, action, target, status, detail, prev_hash, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, actor, action, target, status, detail_text, prev_hash, current_hash),
        )
        return current_hash
    except Exception:
        return ""


def verify_audit(conn: Optional[sqlite3.Connection]) -> Tuple[bool, Optional[int]]:
    if conn is None:
        return False, None
    rows = conn.execute("SELECT * FROM audit ORDER BY id").fetchall()
    prev_hash = "GENESIS"
    for row in rows:
        payload = json.dumps(
            {
                "ts": row["ts"],
                "actor": row["actor"],
                "action": row["action"],
                "target": row["target"],
                "status": row["status"],
                "detail": row["detail"],
            },
            sort_keys=True,
        )
        calculated = sha256_text(prev_hash + payload)
        if row["prev_hash"] != prev_hash or row["hash"] != calculated:
            return False, int(row["id"])
        prev_hash = row["hash"]
    return True, None


def last_audit_hash(conn: Optional[sqlite3.Connection]) -> str:
    if conn is None:
        return "NO_DB"
    try:
        row = conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        return row["hash"] if row else "GENESIS"
    except Exception:
        return "GENESIS"


def apply_lab_action(conn: Optional[sqlite3.Connection], action: Dict[str, Any]) -> None:
    if conn is None:
        return
    typ = _clean_str(action.get("type"))
    emp = _clean_str(action.get("employee_id"))
    system = _clean_str(action.get("system"))
    entitlement = _clean_str(action.get("entitlement")) or "all"
    event_id = _clean_str(action.get("event_id"))
    ts = now_iso()

    if typ == "create_user":
        conn.execute(
            """
            INSERT INTO users
            (employee_id, full_name, role, department, status, source, is_shared, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(employee_id) DO UPDATE SET
                full_name=excluded.full_name,
                role=excluded.role,
                department=excluded.department,
                status='active',
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (
                emp,
                _clean_str(action.get("full_name")),
                _clean_str(action.get("role")),
                _clean_str(action.get("department")),
                "active",
                f"LAB-{event_id}",
                0,
                ts,
                ts,
            ),
        )
    elif typ == "grant":
        conn.execute(
            """
            INSERT INTO grants
            (employee_id, system, entitlement, status, source_event, granted_at, revoked_at)
            VALUES (?, ?, ?, 'active', ?, ?, NULL)
            ON CONFLICT(employee_id, system, entitlement) DO UPDATE SET
                status='active',
                source_event=excluded.source_event,
                granted_at=excluded.granted_at,
                revoked_at=NULL
            """,
            (emp, system, entitlement, event_id, ts),
        )
    elif typ in {
        "revoke",
        "revoke_access",
        "revoke_sessions",
        "disable_mfa",
        "remove_mailbox_forwarding",
        "remove_groups",
        "revoke_licenses",
    }:
        conn.execute(
            """
            INSERT INTO grants
            (employee_id, system, entitlement, status, source_event, granted_at, revoked_at)
            VALUES (?, ?, ?, 'revoked', ?, ?, ?)
            ON CONFLICT(employee_id, system, entitlement) DO UPDATE SET
                status='revoked',
                source_event=excluded.source_event,
                revoked_at=excluded.revoked_at
            """,
            (emp, system, entitlement, event_id, ts, ts),
        )
        if typ == "disable_mfa":
            conn.execute(
                """
                INSERT OR IGNORE INTO grants
                (employee_id, system, entitlement, status, source_event, granted_at, revoked_at)
                VALUES (?, 'mfa', 'enrollment', 'revoked', ?, ?, ?)
                """,
                (emp, event_id, ts, ts),
            )
    elif typ == "disable_lab_account":
        conn.execute(
            """
            INSERT INTO users
            (employee_id, full_name, role, department, status, source, is_shared, created_at, updated_at)
            VALUES (?, '', '', '', 'disabled', ?, 0, ?, ?)
            ON CONFLICT(employee_id) DO UPDATE SET
                status='disabled',
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (emp, f"LAB-{event_id}", ts, ts),
        )
        conn.execute(
            """
            UPDATE grants
            SET status='revoked', revoked_at=?
            WHERE employee_id=? AND status='active'
            """,
            (ts, emp),
        )
    conn.commit()


def detect_orphans(conn: Optional[sqlite3.Connection], active_employee_ids: set) -> List[Dict[str, Any]]:
    if conn is None:
        return []
    out: List[Dict[str, Any]] = []
    try:
        rows = conn.execute(
            "SELECT * FROM users WHERE status='active' AND source='seed-orphan'"
        ).fetchall()
        for row in rows:
            emp = row["employee_id"]
            if emp not in active_employee_ids:
                out.append(
                    {
                        "employee_id": emp,
                        "full_name": row["full_name"],
                        "reason": "Active lab account has no current active HR record",
                        "evidence": {
                            "source": row["source"],
                            "status": row["status"],
                            "department": row["department"],
                        },
                    }
                )
    except Exception:
        pass
    return out


def count_actual_orphans(conn: Optional[sqlite3.Connection]) -> int:
    if conn is None:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE status='active' AND source='seed-orphan'"
        ).fetchone()
        return int(row["c"]) if row else 0
    except Exception:
        return 0


def detect_shared_accounts(conn: Optional[sqlite3.Connection]) -> List[Dict[str, Any]]:
    if conn is None:
        return []
    out: List[Dict[str, Any]] = []
    try:
        rows = conn.execute("SELECT * FROM users WHERE is_shared=1").fetchall()
        for row in rows:
            out.append(
                {
                    "employee_id": row["employee_id"],
                    "full_name": row["full_name"],
                    "type": "shared_lab_account",
                    "reason": "Account is marked shared and needs owner confirmation",
                    "evidence": {"status": row["status"], "source": row["source"]},
                }
            )
    except Exception:
        pass
    return out


def access_graph_insights(projection: Dict[str, set], active_employee_ids: set) -> Tuple[List[Dict[str, Any]], List[str]]:
    G = nx.Graph()
    for emp, access in projection.items():
        G.add_node(emp, node_type="employee")
        for item in access:
            G.add_node(item, node_type="entitlement")
            G.add_edge(emp, item)
    shared: List[Dict[str, Any]] = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "entitlement" and G.degree(node) > 1:
            shared.append(
                {
                    "entitlement": node,
                    "employees": sorted(list(G.neighbors(node))),
                    "type": "shared_entitlement",
                    "reason": "Entitlement is assigned to multiple identities",
                }
            )
    isolated = [
        node
        for node, data in G.nodes(data=True)
        if data.get("node_type") == "employee" and G.degree(node) == 0 and node in active_employee_ids
    ]
    return shared, isolated


# -----------------------------------------------------------------------------
# Agent core
# -----------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        demo_fn: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.demo_fn = demo_fn
        self.schema = schema or {}

    def make_prompt(self, task: Any, context: Dict[str, Any]) -> str:
        safe_task = make_json_safe(task)
        safe_context = make_json_safe(
            {k: v for k, v in context.items() if k not in JSON_EXCLUDE_KEYS}
        )
        return (
            f"{self.system_prompt}\n\n"
            "Return only valid JSON matching this schema:\n"
            f"{json.dumps(self.schema, indent=2, default=str)}\n\n"
            f"TASK:\n{json.dumps(safe_task, indent=2, default=str)}\n\n"
            f"CONTEXT:\n{json.dumps(safe_context, indent=2, default=str)}"
        )

    def run(
        self,
        task: Any,
        context: Dict[str, Any],
        json_mode: bool = True,
        api_key: Optional[str] = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        if not api_key:
            result = self.demo_fn(task, context)
            if not validate_schema(result, self.schema):
                return {"error": "invalid_demo_output", "partial": make_json_safe(result)}
            return result

        last_reason = "unknown"
        for attempt in range(max(1, retries)):
            try:
                prompt = self.make_prompt(task, context)
                raw = call_gemini(prompt, api_key=api_key, json_mode=json_mode)
                parsed = json.loads(strip_code_fences(raw))
                if validate_schema(parsed, self.schema):
                    return parsed
                last_reason = "invalid_schema"
            except urllib.error.HTTPError as exc:
                last_reason = f"http_{exc.code}"
            except urllib.error.URLError:
                last_reason = "network_unavailable"
            except json.JSONDecodeError:
                last_reason = "invalid_json"
            except Exception as exc:
                last_reason = type(exc).__name__
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
        return {"error": "model unavailable", "reason": last_reason}


def call_gemini(prompt: str, api_key: str, json_mode: bool = True) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={urllib.parse.quote_plus(api_key)}"
    )
    generation_config: Dict[str, Any] = {"temperature": 0.2}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise ValueError("malformed_model_response") from exc


class Crew:
    def __init__(self, agents: Iterable[Agent], on_event: Callable[[Dict[str, Any]], None]) -> None:
        self.agents: Dict[str, Agent] = {agent.name: agent for agent in agents}
        self.on_event = on_event

    def emit(self, agent: str, level: str, msg: str, kpi: Any = None) -> None:
        event = {
            "ts": now_iso(),
            "agent": agent,
            "level": level if level in LEVEL_COLORS else "info",
            "msg": msg,
            "kpi": sanitize_for_log(kpi),
        }
        self.on_event(event)

    def _run_agent(self, agent: Agent, task: Any, context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        api_key = state.get("api_key")
        try:
            result = agent.run(task, context, api_key=api_key)
        except Exception as exc:
            result = {"error": "model unavailable", "reason": type(exc).__name__}
        if not isinstance(result, dict):
            result = {"error": "invalid_model_output"}
        if result.get("error") or not validate_schema(result, agent.schema):
            self.emit(
                agent.name,
                "warn",
                f"Deterministic fallback used for {agent.role}",
                {"reason": result.get("error", "schema_validation")},
            )
            try:
                result = agent.demo_fn(task, context)
            except Exception as exc:
                result = {"error": "fallback_failed", "reason": type(exc).__name__}
            if not isinstance(result, dict):
                result = {"error": "fallback_invalid"}
        return result

    def run(self, steps: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
        for step in steps:
            agent_name = step.get("agent")
            agent = self.agents.get(agent_name)
            if agent is None:
                self.emit("Orchestrator", "crit", f"Unknown agent requested: {agent_name}")
                continue
            task = step["task"](state) if callable(step.get("task")) else step.get("task")
            context = {**state, **step.get("context", {})}
            self.emit(agent.name, "info", f"Started {agent.role}", step.get("kpi"))
            result = self._run_agent(agent, task, context, state)
            state[agent.name] = result
            level = "ok" if isinstance(result, dict) and not result.get("error") else "crit"
            self.emit(
                agent.name,
                level,
                f"Finished {agent.role}",
                {"result_fields": list(result.keys()) if isinstance(result, dict) else []},
            )
        return state


def approval_gate(action: Dict[str, Any], approved: bool = False, dry_run: bool = True) -> Dict[str, Any]:
    if dry_run:
        status = "dry_run"
    elif not approved:
        status = "pending_approval"
    else:
        status = "executed"
    return {"status": status, "action": action}


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

LISTENER_SCHEMA = {
    "required": ["employee_id", "event_type", "effective_date", "role", "duplicate"],
    "properties": {
        "event_id": "string",
        "employee_id": "string",
        "full_name": "string",
        "event_type": "string",
        "effective_date": "string",
        "received_at": "string",
        "role": "string",
        "previous_role": "string",
        "department": "string",
        "duplicate": "boolean",
        "late": "boolean",
    },
}

ROLE_MAPPER_SCHEMA = {
    "required": ["employee_id", "role", "desired_bundles", "baseline_count", "missing_role"],
    "properties": {
        "employee_id": "string",
        "role": "string",
        "desired_bundles": "object",
        "baseline_count": "integer",
        "missing_role": "boolean",
    },
}

PROVISIONER_SCHEMA = {
    "required": ["actions", "summary"],
    "properties": {"actions": "array", "summary": "object"},
}

REVIEWER_SCHEMA = {
    "required": ["delta", "creep_findings", "pending_removals", "actions", "grace_end"],
    "properties": {
        "delta": "object",
        "creep_findings": "array",
        "pending_removals": "array",
        "actions": "array",
        "grace_end": "string",
    },
}

OFFBOARDING_SCHEMA = {
    "required": ["ordered_actions", "orphan_checks", "shared_checks", "certificate_draft", "completion_offset_hours"],
    "properties": {
        "ordered_actions": "array",
        "orphan_checks": "array",
        "shared_checks": "array",
        "certificate_draft": "string",
        "completion_offset_hours": "number",
    },
}

CRITIC_SCHEMA = {
    "required": ["verdict", "findings", "confidence"],
    "properties": {"verdict": "string", "findings": "array", "confidence": "number"},
}

EXECUTOR_SCHEMA = {
    "required": ["gates", "summary"],
    "properties": {"gates": "array", "summary": "object"},
}


# -----------------------------------------------------------------------------
# Action helpers
# -----------------------------------------------------------------------------

def make_action(
    action_type: str,
    system: str,
    employee_id: str,
    entitlement: str,
    event_id: str,
    reason: str,
    risk: str = "low",
    requires_approval: bool = True,
    requires_ownership: bool = False,
    immediate: bool = True,
    scheduled_for: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    raw = f"{event_id}|{action_type}|{system}|{employee_id}|{entitlement}"
    action = {
        "id": short_hash(raw),
        "type": action_type,
        "system": system,
        "employee_id": employee_id,
        "entitlement": entitlement,
        "event_id": event_id,
        "reason": reason,
        "risk": risk,
        "requires_approval": requires_approval,
        "requires_ownership": requires_ownership,
        "immediate": immediate,
        "scheduled_for": scheduled_for or "",
    }
    action.update(extra)
    return action


def bundle_to_set(bundles: Dict[str, Any]) -> set:
    out = set()
    if not isinstance(bundles, dict):
        return out
    for system in SYSTEM_ORDER:
        for ent in bundles.get(system, []) or []:
            out.add(f"{system}:{ent}")
    return out


def bundles_from_access(access: Iterable[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {s: [] for s in SYSTEM_ORDER}
    for item in access or []:
        text = _clean_str(item)
        if ":" not in text:
            continue
        system, ent = text.split(":", 1)
        if system in out:
            out[system].append(ent)
    return out


def representative_badges(bundles: Dict[str, Any], state: str = "active") -> List[Dict[str, Any]]:
    badges: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        ents = bundles.get(system, []) if isinstance(bundles, dict) else []
        if ents:
            badges.append({"label": SYSTEM_ABBR.get(system, system.upper()), "state": state})
    return badges


# -----------------------------------------------------------------------------
# Deterministic agent demo functions
# -----------------------------------------------------------------------------

def listener_demo(task: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    event = task if isinstance(task, dict) else context.get("event", {})
    emp = _clean_str(event.get("employee_id"))
    etype = _clean_str(event.get("event_type")).lower()
    role = _clean_str(event.get("role"))
    previous_role = _clean_str(event.get("previous_role"))
    received_text = to_iso(event.get("received_at"))
    effective_text = to_iso(event.get("effective_date"))
    received_dt = to_dt(received_text)
    effective_dt = to_dt(effective_text)
    late = bool(received_dt and effective_dt and received_dt > effective_dt)
    return {
        "event_id": _clean_str(event.get("event_id")),
        "employee_id": emp,
        "full_name": _clean_str(event.get("full_name")),
        "event_type": etype,
        "received_at": received_text,
        "effective_date": effective_text,
        "role": role,
        "previous_role": previous_role,
        "department": _clean_str(event.get("department")),
        "duplicate": bool(context.get("duplicate", False)),
        "late": late,
    }


def role_mapper_demo(task: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    listener = task if isinstance(task, dict) else context.get("HR Event Listener", {})
    role_matrix = context.get("role_matrix", {}) or {}
    roles = role_matrix.get("roles", {}) if isinstance(role_matrix, dict) else {}
    role = _clean_str(listener.get("role"))
    bundles = roles.get(role, {}) if isinstance(roles, dict) else {}
    desired = {system: list(bundles.get(system, []) or []) for system in SYSTEM_ORDER}
    baseline = sum(len(v) for v in desired.values())
    missing = role not in roles
    return {
        "employee_id": _clean_str(listener.get("employee_id")),
        "role": role,
        "desired_bundles": desired,
        "baseline_count": int(baseline),
        "missing_role": bool(missing),
    }


def provisioner_demo(task: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    role_map = task if isinstance(task, dict) else {}
    listener = context.get("HR Event Listener", {})
    event_id = _clean_str(listener.get("event_id"))
    emp = _clean_str(listener.get("employee_id"))
    if listener.get("duplicate") or _clean_str(listener.get("event_type")).lower() != "hire":
        return {"actions": [], "summary": {"reason": "not_a_valid_hire", "employee_id": emp}}
    if role_map.get("missing_role"):
        return {
            "actions": [],
            "summary": {
                "reason": "missing_role",
                "employee_id": emp,
                "message": "No least-privilege bundle could be resolved. Analyst review required.",
            },
        }
    actions = [
        make_action(
            "create_user",
            "directory",
            emp,
            "",
            event_id,
            "Create lab directory identity",
            risk="medium",
            full_name=listener.get("full_name", ""),
            role=listener.get("role", ""),
            department=listener.get("department", ""),
        )
    ]
    for system in SYSTEM_ORDER:
        for ent in role_map.get("desired_bundles", {}).get(system, []):
            actions.append(
                make_action(
                    "grant",
                    system,
                    emp,
                    ent,
                    event_id,
                    "Least-privilege role bundle",
                    risk="medium",
                )
            )
    return {
        "actions": actions,
        "summary": {
            "employee_id": emp,
            "expected_entitlements": int(role_map.get("baseline_count", 0)),
            "generated_actions": len(actions),
        },
    }


def reviewer_demo(task: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    role_map = task if isinstance(task, dict) else {}
    listener = context.get("HR Event Listener", {})
    role_matrix = context.get("role_matrix", {}) or {}
    roles = role_matrix.get("roles", {}) if isinstance(role_matrix, dict) else {}
    access_state = context.get("access_state", {}) or {}
    as_of = context.get("as_of", AS_OF)
    if not isinstance(as_of, datetime):
        as_of = to_dt(as_of) or AS_OF

    emp = _clean_str(listener.get("employee_id"))
    event_id = _clean_str(listener.get("event_id"))
    if listener.get("duplicate") or _clean_str(listener.get("event_type")).lower() != "transfer":
        return {
            "delta": {"grants": [], "removes": []},
            "creep_findings": [],
            "pending_removals": [],
            "actions": [],
            "grace_end": "",
        }

    old_role = _clean_str(listener.get("previous_role"))
    new_role = _clean_str(listener.get("role"))
    current_access = set(access_state.get(emp, set()))
    old_bundle = roles.get(old_role, {}) if old_role in roles else bundles_from_access(current_access)
    new_bundle = role_map.get("desired_bundles", {}) if isinstance(role_map, dict) else {}

    grace_end_dt = (to_dt(listener.get("effective_date")) or as_of) + timedelta(days=GRACE_DAYS)
    past_grace = as_of > grace_end_dt
    late = bool(listener.get("late"))
    immediate_removal = late or past_grace

    grants: List[str] = []
    removes: List[str] = []
    creep_findings: List[Dict[str, Any]] = []
    pending_removals: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    all_systems = sorted(set(list(old_bundle.keys()) + list(new_bundle.keys()) + SYSTEM_ORDER))
    for system in all_systems:
        if system not in SYSTEM_ORDER:
            continue
        old_set = set(old_bundle.get(system, []) or [])
        new_set = set(new_bundle.get(system, []) or [])
        for ent in sorted(new_set - old_set):
            item = f"{system}:{ent}"
            grants.append(item)
            actions.append(
                make_action(
                    "grant",
                    system,
                    emp,
                    ent,
                    event_id,
                    "New role least-privilege entitlement",
                    risk="medium",
                )
            )
        for ent in sorted(old_set - new_set):
            item = f"{system}:{ent}"
            removes.append(item)
            action = make_action(
                "revoke",
                system,
                emp,
                ent,
                event_id,
                "Remove legacy entitlement after transfer",
                risk="high",
                requires_ownership=True,
                immediate=immediate_removal,
                scheduled_for="" if immediate_removal else grace_end_dt.isoformat(),
            )
            actions.append(action)
            evidence = {
                "employee_id": emp,
                "event_id": event_id,
                "old_role": old_role,
                "new_role": new_role,
                "system": system,
                "entitlement": ent,
                "effective_date": listener.get("effective_date", ""),
                "grace_end": grace_end_dt.isoformat(),
                "as_of": as_of.isoformat(),
                "late_event": late,
            }
            if immediate_removal:
                creep_findings.append(
                    {
                        "code": "privilege_creep",
                        "severity": "crit" if late else "warn",
                        "system": system,
                        "entitlement": ent,
                        "message": f"Legacy access {item} remained after grace period or late HR event.",
                        "evidence": evidence,
                        "recommendation": "Approve lab revocation or record business justification.",
                        "label": "lead_for_analyst_review",
                    }
                )
            else:
                pending_removals.append(
                    {
                        "code": "scheduled_removal",
                        "severity": "info",
                        "system": system,
                        "entitlement": ent,
                        "message": f"Legacy access {item} is scheduled for removal at grace end.",
                        "evidence": evidence,
                        "label": "scheduled",
                    }
                )

    return {
        "delta": {"grants": grants, "removes": removes},
        "creep_findings": creep_findings,
        "pending_removals": pending_removals,
        "actions": actions,
        "grace_end": grace_end_dt.isoformat(),
    }


def offboarding_demo(task: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    listener = task if isinstance(task, dict) else context.get("HR Event Listener", {})
    role_matrix = context.get("role_matrix", {}) or {}
    roles = role_matrix.get("roles", {}) if isinstance(role_matrix, dict) else {}
    access_state = context.get("access_state", {}) or {}
    conn = context.get("conn")

    emp = _clean_str(listener.get("employee_id"))
    event_id = _clean_str(listener.get("event_id"))
    role = _clean_str(listener.get("role"))
    if listener.get("duplicate") or _clean_str(listener.get("event_type")).lower() != "exit":
        return {
            "ordered_actions": [],
            "orphan_checks": [],
            "shared_checks": [],
            "certificate_draft": "",
            "completion_offset_hours": 0.0,
        }

    current_access = set(access_state.get(emp, set()))
    bundles = roles.get(role, {}) if role in roles else bundles_from_access(current_access)

    ordered_actions: List[Dict[str, Any]] = []
    ordered_actions.append(
        make_action(
            "revoke_sessions",
            "session",
            emp,
            "active_sessions",
            event_id,
            "Kill active sessions first",
            risk="critical",
            requires_ownership=True,
        )
    )
    ordered_actions.append(
        make_action(
            "disable_mfa",
            "mfa",
            emp,
            "enrollment",
            event_id,
            "Disable MFA enrollment after sessions",
            risk="critical",
            requires_ownership=True,
        )
    )
    ordered_actions.append(
        make_action(
            "remove_mailbox_forwarding",
            "mailbox_forwarding",
            emp,
            "inbox_forwarding",
            event_id,
            "Remove mailbox forwarding to prevent data leakage",
            risk="high",
            requires_ownership=True,
        )
    )

    group_items = [item for item in sorted(current_access) if item.startswith("group:")]
    if group_items:
        for item in group_items:
            system, ent = item.split(":", 1)
            ordered_actions.append(
                make_action(
                    "remove_groups",
                    system,
                    emp,
                    ent,
                    event_id,
                    "Remove group membership",
                    risk="high",
                    requires_ownership=True,
                )
            )
    else:
        ordered_actions.append(
            make_action(
                "remove_groups",
                "group",
                emp,
                "all_groups",
                event_id,
                "Remove all group memberships",
                risk="high",
                requires_ownership=True,
            )
        )

    license_items = [item for item in sorted(current_access) if item.startswith("license:")]
    if license_items:
        for item in license_items:
            system, ent = item.split(":", 1)
            ordered_actions.append(
                make_action(
                    "revoke_licenses",
                    system,
                    emp,
                    ent,
                    event_id,
                    "Revoke license assignment",
                    risk="medium",
                    requires_ownership=True,
                )
            )
    else:
        ordered_actions.append(
            make_action(
                "revoke_licenses",
                "license",
                emp,
                "all_licenses",
                event_id,
                "Revoke all license assignments",
                risk="medium",
                requires_ownership=True,
            )
        )

    for system in SYSTEM_ORDER:
        for ent in bundles.get(system, []) or []:
            ordered_actions.append(
                make_action(
                    "revoke_access",
                    system,
                    emp,
                    ent,
                    event_id,
                    "Revoke residual role access",
                    risk="high",
                    requires_ownership=True,
                )
            )

    ordered_actions.append(
        make_action(
            "disable_lab_account",
            "directory",
            emp,
            "",
            event_id,
            "Disable lab account last",
            risk="critical",
            requires_ownership=True,
        )
    )

    orphan_checks = [
        {
            "employee_id": emp,
            "orphan": False,
            "reason": "Exit event provides HR evidence for offboarding",
            "evidence": {"event_id": event_id, "effective_date": listener.get("effective_date", "")},
        }
    ]
    shared_checks: List[Dict[str, Any]] = []
    if conn is not None:
        try:
            row = conn.execute("SELECT is_shared, status FROM users WHERE employee_id=?", (emp,)).fetchone()
            if row and int(row["is_shared"] or 0) == 1:
                shared_checks.append(
                    {
                        "employee_id": emp,
                        "shared": True,
                        "reason": "Lab account is marked shared and needs ownership confirmation",
                        "evidence": {"status": row["status"]},
                    }
                )
        except Exception:
            pass

    offset = 1.5 + (int(short_hash(event_id, 8), 16) % 10) / 10.0
    certificate_draft = (
        "LIFECYCLEBOT OFFBOARDING CERTIFICATE\n"
        "Employee: {employee_id}\n"
        "Event: {event_id}\n"
        "Issued: {issued_at}\n"
        "Status: {status}\n"
        "Revocation order: sessions, MFA, mailbox forwarding, groups, licenses, residual access, lab account disable.\n"
        "Steps:\n{steps}\n"
        "Audit root: {audit_root}\n"
        "Note: simulated lab certificate; no real system was changed.\n"
    )

    return {
        "ordered_actions": ordered_actions,
        "orphan_checks": orphan_checks,
        "shared_checks": shared_checks,
        "certificate_draft": certificate_draft,
        "completion_offset_hours": float(offset),
    }


def critic_demo(task: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    state = task if isinstance(task, dict) else {}
    listener = state.get("HR Event Listener", {}) or {}
    role_map = state.get("Role Mapper", {}) or {}
    provisioner = state.get("Provisioner", {}) or {}
    reviewer = state.get("Access Reviewer", {}) or {}
    offboarder = state.get("Offboarding Agent", {}) or {}
    projection_roles = context.get("projection_roles", {}) or {}
    role_matrix = context.get("role_matrix", {}) or {}
    roles = role_matrix.get("roles", {}) if isinstance(role_matrix, dict) else {}

    findings: List[Dict[str, Any]] = []
    emp = _clean_str(listener.get("employee_id"))
    event_id = _clean_str(listener.get("event_id"))
    etype = _clean_str(listener.get("event_type")).lower()

    if listener.get("duplicate"):
        findings.append(
            {
                "code": "duplicate_event",
                "severity": "crit",
                "message": "Duplicate event id detected. No state-changing action should execute.",
                "evidence": {"event_id": event_id, "employee_id": emp},
                "recommendation": "Suppress duplicate and reconcile HR source.",
                "label": "lead_for_analyst_review",
            }
        )

    if etype not in {"hire", "transfer", "exit"}:
        findings.append(
            {
                "code": "unknown_event_type",
                "severity": "crit",
                "message": f"Unsupported event type: {etype}",
                "evidence": {"event_id": event_id, "employee_id": emp},
                "recommendation": "Map event type or quarantine record.",
                "label": "lead_for_analyst_review",
            }
        )

    if role_map.get("missing_role"):
        findings.append(
            {
                "code": "missing_role",
                "severity": "warn",
                "message": "Role is absent from the role matrix. Least privilege cannot be derived.",
                "evidence": {"employee_id": emp, "role": listener.get("role", "")},
                "recommendation": "Add role mapping or assign temporary restricted bundle.",
                "label": "lead_for_analyst_review",
            }
        )

    if listener.get("late"):
        findings.append(
            {
                "code": "late_event",
                "severity": "warn",
                "message": "HR event was received after its effective date.",
                "evidence": {
                    "employee_id": emp,
                    "received_at": listener.get("received_at", ""),
                    "effective_date": listener.get("effective_date", ""),
                },
                "recommendation": "Prioritize remediation and record audit exception.",
                "label": "lead_for_analyst_review",
            }
        )

    if etype == "transfer":
        current_role = _clean_str(projection_roles.get(emp, ""))
        previous_role = _clean_str(listener.get("previous_role", ""))
        if current_role and previous_role and current_role != previous_role:
            findings.append(
                {
                    "code": "conflicting_previous_role",
                    "severity": "warn",
                    "message": "Transfer previous_role conflicts with last known lab role.",
                    "evidence": {"employee_id": emp, "previous_role": previous_role, "known_role": current_role},
                    "recommendation": "Confirm HR source before removing access.",
                    "label": "lead_for_analyst_review",
                }
            )
        for finding in reviewer.get("creep_findings", []) or []:
            findings.append(finding)

    if etype == "hire":
        expected = int(role_map.get("baseline_count", 0)) if not role_map.get("missing_role") else 0
        grant_count = sum(1 for a in provisioner.get("actions", []) or [] if a.get("type") == "grant")
        if expected != grant_count:
            findings.append(
                {
                    "code": "provisioning_mismatch",
                    "severity": "crit",
                    "message": "Generated grant actions do not match role matrix baseline.",
                    "evidence": {"employee_id": emp, "expected": expected, "generated": grant_count},
                    "recommendation": "Block execution and review role mapping.",
                    "label": "lead_for_analyst_review",
                }
            )

    if etype == "exit":
        expected_prefix = [
            "revoke_sessions",
            "disable_mfa",
            "remove_mailbox_forwarding",
            "remove_groups",
            "revoke_licenses",
        ]
        actual_prefix = [a.get("type", "") for a in offboarder.get("ordered_actions", []) or []][:5]
        if actual_prefix != expected_prefix:
            findings.append(
                {
                    "code": "unsafe_revocation_order",
                    "severity": "crit",
                    "message": "Offboarding order does not follow safe revocation sequence.",
                    "evidence": {"employee_id": emp, "expected": expected_prefix, "actual": actual_prefix},
                    "recommendation": "Do not execute until order is corrected.",
                    "label": "lead_for_analyst_review",
                }
            )

    severities = {f.get("severity") for f in findings}
    if "crit" in severities:
        verdict = "fail"
        confidence = 0.75
    elif "warn" in severities:
        verdict = "warn"
        confidence = 0.90
    else:
        verdict = "pass"
        confidence = 0.99

    return {"verdict": verdict, "findings": findings, "confidence": float(confidence)}


def executor_demo(task: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    state = task if isinstance(task, dict) else {}
    listener = state.get("HR Event Listener", {}) or {}
    critic = state.get("Critic", {}) or {}
    approved = bool(context.get("approved", False))
    dry_run = bool(context.get("dry_run", True))
    ownership = bool(context.get("ownership", False))
    block = _clean_str(critic.get("verdict", "pass")).lower() == "fail"

    actions: List[Dict[str, Any]] = []
    for key in ["Provisioner", "Access Reviewer", "Offboarding Agent"]:
        result = state.get(key, {}) or {}
        if isinstance(result, dict):
            actions.extend(result.get("actions", []) or result.get("ordered_actions", []) or [])

    gates: List[Dict[str, Any]] = []
    for action in actions:
        if block:
            gate = approval_gate(action, approved=False, dry_run=False)
        else:
            needs_approval = bool(action.get("requires_approval", True))
            needs_ownership = bool(action.get("requires_ownership", False))
            effective_approved = approved or not needs_approval
            if needs_ownership and not ownership:
                effective_approved = False
            gate = approval_gate(action, approved=effective_approved, dry_run=dry_run)
        gates.append(gate)

    status_counts: Dict[str, int] = {}
    for gate in gates:
        status = gate.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "gates": gates,
        "summary": {
            "employee_id": listener.get("employee_id", ""),
            "action_count": len(actions),
            "status_counts": status_counts,
            "blocked_by_critic": block,
        },
    }


def build_crew(on_event: Callable[[Dict[str, Any]], None]) -> Crew:
    agents = [
        Agent(
            name="HR Event Listener",
            role="Intake and normalization",
            system_prompt="You normalize synthetic HR joiner, mover, and leaver events into structured JSON.",
            demo_fn=listener_demo,
            schema=LISTENER_SCHEMA,
        ),
        Agent(
            name="Role Mapper",
            role="Least-privilege mapping",
            system_prompt="You map roles to minimal access bundles from the local role matrix.",
            demo_fn=role_mapper_demo,
            schema=ROLE_MAPPER_SCHEMA,
        ),
        Agent(
            name="Provisioner",
            role="Joiner provisioning",
            system_prompt="You create lab-only provisioning actions for hires using least privilege.",
            demo_fn=provisioner_demo,
            schema=PROVISIONER_SCHEMA,
        ),
        Agent(
            name="Access Reviewer",
            role="Mover delta and creep review",
            system_prompt="You compute transfer deltas, grace periods, and privilege-creep leads.",
            demo_fn=reviewer_demo,
            schema=REVIEWER_SCHEMA,
        ),
        Agent(
            name="Offboarding Agent",
            role="Leaver revocation",
            system_prompt="You produce safe ordered revocation steps and offboarding certificates for lab accounts.",
            demo_fn=offboarding_demo,
            schema=OFFBOARDING_SCHEMA,
        ),
        Agent(
            name="Critic",
            role="Skeptical validator",
            system_prompt="You try to disprove lifecycle decisions and surface evidence-backed exceptions.",
            demo_fn=critic_demo,
            schema=CRITIC_SCHEMA,
        ),
        Agent(
            name="Executor",
            role="Approval-gated execution",
            system_prompt="You route every consequential action through the approval gate and never bypass it.",
            demo_fn=executor_demo,
            schema=EXECUTOR_SCHEMA,
        ),
    ]
    return Crew(agents, on_event)


# -----------------------------------------------------------------------------
# Scene emission for Simulation Studio
# -----------------------------------------------------------------------------

def scene_log(on_event: Callable[[Dict[str, Any]], None], agent: str, level: str, msg: str, scene: Dict[str, Any]) -> None:
    event = {
        "ts": now_iso(),
        "agent": agent,
        "level": level,
        "msg": msg,
        "kpi": {"scene": sanitize_for_log(scene)},
    }
    on_event(event)


def emit_joiner_scene(on_event: Callable[[Dict[str, Any]], None], emp: str, role_map: Dict[str, Any]) -> None:
    badges: List[Dict[str, Any]] = []
    x = 0.15
    scene_log(
        on_event,
        "HR Event Listener",
        "info",
        f"Joiner {emp} enters the lab",
        {"actor": "joiner", "silhouette_x": x, "actor_state": "walking_in", "badges": []},
    )
    for system in SYSTEM_ORDER:
        ents = role_map.get("desired_bundles", {}).get(system, []) if isinstance(role_map, dict) else []
        if not ents:
            continue
        for badge in badges:
            badge["state"] = "active"
        badges.append({"label": SYSTEM_ABBR[system], "state": "appearing"})
        x = min(0.45, x + 0.05)
        scene_log(
            on_event,
            "Provisioner",
            "ok",
            f"{emp} badge {system} provisioned",
            {"actor": "joiner", "silhouette_x": x, "actor_state": "walking_in", "badges": badges},
        )
    for badge in badges:
        badge["state"] = "active"
    scene_log(
        on_event,
        "Critic",
        "ok",
        f"Joiner {emp} least-privilege bundle verified",
        {"actor": "joiner", "silhouette_x": x, "actor_state": "standing", "badges": badges},
    )


def emit_mover_scene(
    on_event: Callable[[Dict[str, Any]], None],
    emp: str,
    listener: Dict[str, Any],
    role_map: Dict[str, Any],
    reviewer: Dict[str, Any],
    role_matrix: Dict[str, Any],
) -> None:
    roles = role_matrix.get("roles", {}) if isinstance(role_matrix, dict) else {}
    old_role = _clean_str(listener.get("previous_role"))
    new_role = _clean_str(listener.get("role"))
    old_bundle = roles.get(old_role, {}) if old_role in roles else {}
    new_bundle = role_map.get("desired_bundles", {}) if isinstance(role_map, dict) else {}

    old_badges = representative_badges(old_bundle, "active")
    new_badges = representative_badges(new_bundle, "active")
    old_labels = [b["label"] for b in old_badges]
    new_labels = [b["label"] for b in new_badges]
    old_only = [label for label in old_labels if label not in new_labels]
    new_only = [label for label in new_labels if label not in old_labels]

    badges = [dict(b) for b in old_badges]
    x = 0.45
    scene_log(
        on_event,
        "HR Event Listener",
        "info",
        f"Mover {emp} changes role from {old_role} to {new_role}",
        {"actor": "mover", "silhouette_x": x, "actor_state": "standing", "badges": badges},
    )

    shed_labels = old_only[:1]
    retained_label = old_only[1:2]
    for label in shed_labels:
        for badge in badges:
            if badge["label"] == label:
                badge["state"] = "shedding"
        scene_log(
            on_event,
            "Access Reviewer",
            "warn",
            f"{emp} shedding legacy badge {label}",
            {"actor": "mover", "silhouette_x": x, "actor_state": "standing", "badges": badges},
        )
        badges = [badge for badge in badges if badge["label"] != label]

    if retained_label and reviewer.get("creep_findings"):
        for badge in badges:
            if badge["label"] == retained_label[0]:
                badge["state"] = "active"
        scene_log(
            on_event,
            "Critic",
            "crit",
            f"{emp} privilege creep lead: legacy badge {retained_label[0]} remained past grace",
            {"actor": "mover", "silhouette_x": x, "actor_state": "standing", "badges": badges},
        )

    for label in new_only[:1]:
        badges.append({"label": label, "state": "appearing"})
        scene_log(
            on_event,
            "Provisioner",
            "ok",
            f"{emp} new badge {label} appears",
            {"actor": "mover", "silhouette_x": x, "actor_state": "standing", "badges": badges},
        )
    for badge in badges:
        if badge["state"] == "appearing":
            badge["state"] = "active"
    scene_log(
        on_event,
        "Executor",
        "ok",
        f"Mover {emp} delta routed through approval gate",
        {"actor": "mover", "silhouette_x": x, "actor_state": "standing", "badges": badges},
    )


def emit_leaver_scene(on_event: Callable[[Dict[str, Any]], None], emp: str, offboarder: Dict[str, Any]) -> None:
    access_labels = ["DIR", "MAIL", "CHAT", "CODE", "TICK"]
    control_labels = ["SESS", "MFA", "FWD", "GRP", "LIC"]
    badges = [{"label": label, "state": "active"} for label in control_labels + access_labels]
    x = 0.55
    scene_log(
        on_event,
        "HR Event Listener",
        "info",
        f"Leaver {emp} exits; revocation sequence starts",
        {"actor": "leaver", "silhouette_x": x, "actor_state": "standing", "badges": badges},
    )

    for label in control_labels:
        for badge in badges:
            if badge["label"] == label:
                badge["state"] = "dissolving"
        x = min(0.85, x + 0.03)
        scene_log(
            on_event,
            "Offboarding Agent",
            "ok",
            f"{emp} control badge {label} dissolves",
            {"actor": "leaver", "silhouette_x": x, "actor_state": "walking_out", "badges": badges},
        )
        badges = [badge for badge in badges if badge["label"] != label]

    for label in access_labels[:3]:
        for badge in badges:
            if badge["label"] == label:
                badge["state"] = "dissolving"
        x = min(0.85, x + 0.03)
        scene_log(
            on_event,
            "Offboarding Agent",
            "ok",
            f"{emp} access badge {label} dissolves",
            {"actor": "leaver", "silhouette_x": x, "actor_state": "walking_out", "badges": badges},
        )
        badges = [badge for badge in badges if badge["label"] != label]

    scene_log(
        on_event,
        "Critic",
        "warn",
        f"{emp} orphan and shared account checks completed",
        {"actor": "leaver", "silhouette_x": x, "actor_state": "walking_out", "badges": badges},
    )
    scene_log(
        on_event,
        "Executor",
        "ok",
        f"{emp} offboarding certificate prepared",
        {"actor": "leaver", "silhouette_x": x, "actor_state": "gone", "badges": []},
    )


def emit_persona_scenes(
    on_event: Callable[[Dict[str, Any]], None],
    persona: Optional[str],
    listener: Dict[str, Any],
    role_map: Dict[str, Any],
    reviewer: Dict[str, Any],
    offboarder: Dict[str, Any],
    role_matrix: Dict[str, Any],
) -> None:
    if not persona:
        return
    emp = _clean_str(listener.get("employee_id"))
    if not emp:
        return
    if listener.get("duplicate"):
        return
    if persona == "joiner" and role_map:
        emit_joiner_scene(on_event, emp, role_map)
    elif persona == "mover" and reviewer:
        emit_mover_scene(on_event, emp, listener, role_map, reviewer, role_matrix)
    elif persona == "leaver" and offboarder:
        emit_leaver_scene(on_event, emp, offboarder)


# -----------------------------------------------------------------------------
# Processing engine
# -----------------------------------------------------------------------------

def compute_active_employee_ids(df: pd.DataFrame) -> set:
    if df is None or df.empty:
        return set()
    tmp = df.copy()
    if "received_at" not in tmp.columns:
        tmp["received_at"] = ""
    tmp["_recv"] = tmp["received_at"].map(lambda v: to_dt(v) or datetime.min.replace(tzinfo=timezone.utc))
    tmp = tmp.sort_values("_recv")
    status: Dict[str, str] = {}
    for _, row in tmp.iterrows():
        emp = _clean_str(row.get("employee_id"))
        if not emp:
            continue
        etype = _clean_str(row.get("event_type")).lower()
        stat = _clean_str(row.get("employment_status")).lower()
        if not stat:
            stat = "terminated" if etype == "exit" else "active"
        status[emp] = stat
    return {emp for emp, stat in status.items() if stat == "active"}


def update_projection_from_gates(projection: Dict[str, set], gates: List[Dict[str, Any]]) -> None:
    for gate in gates:
        if gate.get("status") not in {"dry_run", "executed"}:
            continue
        action = gate.get("action", {}) or {}
        emp = _clean_str(action.get("employee_id"))
        if not emp:
            continue
        typ = _clean_str(action.get("type"))
        system = _clean_str(action.get("system"))
        entitlement = _clean_str(action.get("entitlement"))
        if typ == "create_user":
            projection.setdefault(emp, set())
        elif typ == "grant" and system and entitlement:
            projection.setdefault(emp, set()).add(f"{system}:{entitlement}")
        elif typ in {
            "revoke",
            "revoke_access",
            "revoke_sessions",
            "disable_mfa",
            "remove_mailbox_forwarding",
            "remove_groups",
            "revoke_licenses",
        }:
            if not bool(action.get("immediate", True)):
                continue
            if system and entitlement:
                projection.setdefault(emp, set()).discard(f"{system}:{entitlement}")
        elif typ == "disable_lab_account":
            projection[emp] = set()


def process_all_events(
    events_df: pd.DataFrame,
    role_matrix: Dict[str, Any],
    on_event: Callable[[Dict[str, Any]], None],
    api_key: Optional[str],
    dry_run: bool,
    approved: bool,
    ownership: bool,
    conn: Optional[sqlite3.Connection],
) -> Dict[str, Any]:
    crew = build_crew(on_event)
    seen_event_ids: set = set()
    projection: Dict[str, set] = {}
    projection_roles: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    creep_findings: List[Dict[str, Any]] = []
    certificates: List[Dict[str, Any]] = []
    checklists: List[Dict[str, Any]] = []
    pending_actions: List[Dict[str, Any]] = []

    expected_provision = 0
    matched_provision = 0
    creep_tp = 0
    creep_fp = 0
    revoke_hours: List[float] = []

    steps_template_hire = [
        {"agent": "HR Event Listener", "task": lambda s: s["event"]},
        {"agent": "Role Mapper", "task": lambda s: s["HR Event Listener"]},
        {"agent": "Provisioner", "task": lambda s: s["Role Mapper"]},
        {"agent": "Critic", "task": lambda s: s},
        {"agent": "Executor", "task": lambda s: s},
    ]
    steps_template_transfer = [
        {"agent": "HR Event Listener", "task": lambda s: s["event"]},
        {"agent": "Role Mapper", "task": lambda s: s["HR Event Listener"]},
        {"agent": "Access Reviewer", "task": lambda s: s["Role Mapper"]},
        {"agent": "Critic", "task": lambda s: s},
        {"agent": "Executor", "task": lambda s: s},
    ]
    steps_template_exit = [
        {"agent": "HR Event Listener", "task": lambda s: s["event"]},
        {"agent": "Offboarding Agent", "task": lambda s: s["HR Event Listener"]},
        {"agent": "Critic", "task": lambda s: s},
        {"agent": "Executor", "task": lambda s: s},
    ]
    steps_template_unknown = [
        {"agent": "HR Event Listener", "task": lambda s: s["event"]},
        {"agent": "Critic", "task": lambda s: s},
        {"agent": "Executor", "task": lambda s: s},
    ]

    sorted_df = events_df.copy()
    if "received_at" in sorted_df.columns:
        sorted_df["_recv_sort"] = sorted_df["received_at"].map(lambda v: to_dt(v) or datetime.min.replace(tzinfo=timezone.utc))
        sorted_df = sorted_df.sort_values(["_recv_sort", "event_id"])

    for _, row in sorted_df.iterrows():
        raw_event = row.to_dict()
        event_id = _clean_str(raw_event.get("event_id"))
        duplicate = event_id in seen_event_ids
        seen_event_ids.add(event_id)

        state: Dict[str, Any] = {
            "event": raw_event,
            "role_matrix": role_matrix,
            "access_state": projection,
            "projection_roles": projection_roles,
            "as_of": AS_OF,
            "dry_run": dry_run,
            "approved": approved,
            "ownership": ownership,
            "api_key": api_key,
            "conn": conn,
            "duplicate": duplicate,
        }

        etype = _clean_str(raw_event.get("event_type")).lower()
        if etype == "hire":
            steps = steps_template_hire
        elif etype == "transfer":
            steps = steps_template_transfer
        elif etype == "exit":
            steps = steps_template_exit
        else:
            steps = steps_template_unknown

        try:
            crew.run(steps, state)
        except Exception as exc:
            on_event(
                {
                    "ts": now_iso(),
                    "agent": "Orchestrator",
                    "level": "crit",
                    "msg": f"Processing failed safely for {event_id}: {type(exc).__name__}",
                    "kpi": {},
                }
            )

        listener = state.get("HR Event Listener", {}) or {}
        role_map = state.get("Role Mapper", {}) or {}
        provisioner = state.get("Provisioner", {}) or {}
        reviewer = state.get("Access Reviewer", {}) or {}
        offboarder = state.get("Offboarding Agent", {}) or {}
        critic = state.get("Critic", {}) or {}
        executor = state.get("Executor", {}) or {}
        gates = executor.get("gates", []) if isinstance(executor, dict) else []
        emp = _clean_str(listener.get("employee_id"))
        persona = DEMO_PERSONAS.get(emp)

        gate_by_id: Dict[str, str] = {}
        applied_statuses = {"dry_run", "executed"}

        for gate in gates:
            action = gate.get("action", {}) or {}
            status = gate.get("status", "pending_approval")
            try:
                if status == "executed":
                    apply_lab_action(conn, action)
            except Exception as exc:
                status = "execution_failed"
                gate["status"] = status
                on_event(
                    {
                        "ts": now_iso(),
                        "agent": "Executor",
                        "level": "crit",
                        "msg": f"Lab execution failed safely: {type(exc).__name__}",
                        "kpi": {"action_id": action.get("id", "")},
                    }
                )
            gate_by_id[_clean_str(action.get("id"))] = status
            audit_append(
                conn,
                actor="LifecycleBot Executor",
                action=_clean_str(action.get("type")),
                target=emp,
                status=status,
                detail=action,
            )
            if status == "pending_approval":
                pending_actions.append({"action": action, "event_id": event_id, "employee_id": emp})

        update_projection_from_gates(projection, gates)

        if etype == "hire" and not duplicate and not role_map.get("missing_role"):
            expected = int(role_map.get("baseline_count", 0))
            matched = sum(
                1
                for gate in gates
                if gate.get("status") in applied_statuses
                and gate.get("action", {}).get("type") == "grant"
            )
            expected_provision += expected
            matched_provision += matched
            projection_roles[emp] = _clean_str(listener.get("role"))

        if etype == "transfer":
            for finding in reviewer.get("creep_findings", []) or []:
                creep_findings.append(finding)
                if finding.get("evidence", {}).get("entitlement"):
                    creep_tp += 1
                else:
                    creep_fp += 1
            projection_roles[emp] = _clean_str(listener.get("role"))

        if etype == "exit":
            disable_applied = any(
                gate.get("status") in applied_statuses
                and gate.get("action", {}).get("type") == "disable_lab_account"
                for gate in gates
            )
            if disable_applied:
                revoke_hours.append(float(offboarder.get("completion_offset_hours", 0.0)))
            steps_text = []
            checklist_steps = []
            for idx, action in enumerate(offboarder.get("ordered_actions", []) or [], start=1):
                status = gate_by_id.get(_clean_str(action.get("id")), "not_generated")
                line = f"{idx}. {action.get('type')} ({action.get('system')}/{action.get('entitlement')}) -> {status}"
                steps_text.append(line)
                checklist_steps.append({"order": idx, **action, "status": status})
            cert_status = "simulated_complete" if disable_applied else "pending_approval"
            audit_root = last_audit_hash(conn)
            cert_text = offboarder.get("certificate_draft", "").format(
                employee_id=emp,
                event_id=event_id,
                issued_at=now_iso(),
                status=cert_status,
                steps="\n".join(steps_text) if steps_text else "No steps generated.",
                audit_root=audit_root,
            )
            certificate = {
                "certificate_id": f"CERT-{event_id}",
                "employee_id": emp,
                "event_id": event_id,
                "status": cert_status,
                "issued_at": now_iso(),
                "text": cert_text,
                "audit_root": audit_root,
            }
            certificates.append(certificate)
            checklists.append(
                {
                    "event_id": event_id,
                    "employee_id": emp,
                    "status": cert_status,
                    "steps": checklist_steps,
                    "certificate_id": certificate["certificate_id"],
                }
            )
            projection_roles[emp] = "terminated"

        emit_persona_scenes(on_event, persona, listener, role_map, reviewer, offboarder, role_matrix)

        status_counts: Dict[str, int] = {}
        for gate in gates:
            status = gate.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        results.append(
            {
                "event_id": event_id,
                "employee_id": emp,
                "event_type": etype,
                "role": _clean_str(listener.get("role")),
                "effective_date": _clean_str(listener.get("effective_date")),
                "verdict": _clean_str(critic.get("verdict", "unknown")),
                "action_count": len(gates),
                "status_counts": status_counts,
                "creep_count": len(reviewer.get("creep_findings", []) or []),
                "duplicate": duplicate,
            }
        )

    active_ids = compute_active_employee_ids(events_df)
    orphans = detect_orphans(conn, active_ids)
    shared_db = detect_shared_accounts(conn)
    graph_shared, graph_isolated = access_graph_insights(projection, active_ids)
    shared_report = shared_db + graph_shared
    actual_orphans = count_actual_orphans(conn)
    detected_orphans = len(orphans)

    metrics = {
        "provisioning_accuracy": float(matched_provision / expected_provision) if expected_provision else 1.0,
        "mean_time_to_revoke_hours": float(np.mean(revoke_hours)) if revoke_hours else 0.0,
        "creep_detection_precision": float(creep_tp / (creep_tp + creep_fp)) if (creep_tp + creep_fp) else 1.0,
        "orphan_detection_rate": float(detected_orphans / actual_orphans) if actual_orphans else 1.0,
        "events_processed": len(results),
        "pending_approvals": len(pending_actions),
    }

    return {
        "events_df": events_df,
        "results": results,
        "metrics": metrics,
        "creep_findings": creep_findings,
        "certificates": certificates,
        "checklists": checklists,
        "pending_actions": pending_actions,
        "projection": projection,
        "orphan_report": orphans,
        "shared_report": shared_report,
        "graph_isolated_active": graph_isolated,
        "role_matrix": role_matrix,
    }


def empty_processed(role_matrix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "events_df": pd.DataFrame(),
        "results": [],
        "metrics": {
            "provisioning_accuracy": 0.0,
            "mean_time_to_revoke_hours": 0.0,
            "creep_detection_precision": 0.0,
            "orphan_detection_rate": 0.0,
            "events_processed": 0,
            "pending_approvals": 0,
        },
        "creep_findings": [],
        "certificates": [],
        "checklists": [],
        "pending_actions": [],
        "projection": {},
        "orphan_report": [],
        "shared_report": [],
        "graph_isolated_active": [],
        "role_matrix": role_matrix or {},
    }


# -----------------------------------------------------------------------------
# Simulation Studio: rendering and video export
# -----------------------------------------------------------------------------

_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in candidates:
        try:
            if Path(path).exists():
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
        except Exception:
            continue
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    _FONT_CACHE[size] = font
    return font


def rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline=None, width: int = 1) -> None:
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def truncate_text(text: str, max_len: int) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def current_scene_from_events(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for event in reversed(events):
        kpi = event.get("kpi", {})
        if isinstance(kpi, dict) and isinstance(kpi.get("scene"), dict):
            return kpi["scene"]
    return None


def draw_silhouette(draw: ImageDraw.ImageDraw, cx: float, cy: float, scale: float = 1.0, color=(180, 200, 220)) -> None:
    head_r = int(14 * scale)
    body_w = int(22 * scale)
    body_h = int(55 * scale)
    leg_h = int(35 * scale)
    draw.ellipse((cx - head_r, cy - body_h - leg_h - head_r * 2, cx + head_r, cy - body_h - leg_h), fill=color)
    draw.rounded_rectangle(
        (cx - body_w // 2, cy - body_h - leg_h, cx + body_w // 2, cy - leg_h),
        radius=int(8 * scale),
        fill=color,
    )
    draw.line((cx - 6, cy - leg_h, cx - 14, cy), fill=color, width=int(6 * scale))
    draw.line((cx + 6, cy - leg_h, cx + 14, cy), fill=color, width=int(6 * scale))


def render_frame(
    title: str,
    agents: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    kpis: Dict[str, Any],
    progress: float,
) -> Image.Image:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    title_font = get_font(28)
    small_font = get_font(16)
    mono_font = get_font(14)
    kpi_font = get_font(22)
    tiny_font = get_font(12)

    draw.rectangle((0, 0, width, 68), fill=PANEL_COLOR)
    draw.text((24, 20), truncate_text(title, 58), font=title_font, fill=CYAN)
    bar_x, bar_y, bar_w, bar_h = 900, 28, 340, 16
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline=MUTED_COLOR, width=1)
    fill_w = int(max(0.0, min(1.0, float(progress))) * bar_w)
    draw.rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), fill=GREEN)
    draw.text((bar_x + bar_w + 10, bar_y - 2), f"{int(progress * 100)}%", font=small_font, fill=TEXT_COLOR)

    # Agent roster
    left_x, left_y, left_w, left_h = 20, 84, 260, 430
    rounded_rect(draw, (left_x, left_y, left_x + left_w, left_y + left_h), 12, PANEL_COLOR, CYAN, 1)
    draw.text((left_x + 14, left_y + 12), "Agent roster", font=small_font, fill=CYAN)
    active_agent = events[-1].get("agent", "") if events else ""
    yy = left_y + 42
    for agent in agents[:12]:
        name = _clean_str(agent.get("name"))
        role = _clean_str(agent.get("role"))
        is_active = name == active_agent
        color = GREEN if is_active else TEXT_COLOR
        draw.text((left_x + 14, yy), truncate_text(name, 24), font=mono_font, fill=color)
        draw.text((left_x + 14, yy + 18), truncate_text(role, 30), font=tiny_font, fill=MUTED_COLOR)
        if is_active:
            draw.rectangle((left_x + 8, yy - 3, left_x + 11, yy + 32), fill=GREEN)
        yy += 42

    # Live log
    center_x, center_y, center_w, center_h = 300, 84, 640, 430
    rounded_rect(draw, (center_x, center_y, center_x + center_w, center_y + center_h), 12, PANEL_COLOR, CYAN, 1)
    draw.text((center_x + 14, center_y + 12), "Shared live log", font=small_font, fill=CYAN)
    yy = center_y + 42
    for event in events[-15:]:
        ts = _clean_str(event.get("ts"))[11:19]
        agent = _clean_str(event.get("agent"))
        level = _clean_str(event.get("level"))
        msg = _clean_str(event.get("msg"))
        color = LEVEL_COLORS.get(level, TEXT_COLOR)
        draw.text((center_x + 14, yy), ts, font=mono_font, fill=MUTED_COLOR)
        draw.text((center_x + 82, yy), truncate_text(agent, 18), font=mono_font, fill=color)
        draw.text((center_x + 230, yy), truncate_text(msg, 52), font=mono_font, fill=TEXT_COLOR)
        yy += 25

    # KPI cards
    right_x, right_y, right_w, right_h = 960, 84, 300, 430
    rounded_rect(draw, (right_x, right_y, right_x + right_w, right_y + right_h), 12, PANEL_COLOR, GREEN, 1)
    draw.text((right_x + 14, right_y + 12), "Evaluation KPIs", font=small_font, fill=GREEN)
    yy = right_y + 44
    for key, label in KPI_LABELS.items():
        if key not in kpis:
            continue
        value = kpis.get(key, 0)
        try:
            value_f = float(value)
        except Exception:
            value_f = 0.0
        if key in PERCENT_KPI_KEYS:
            display = f"{value_f * 100:.1f}%"
        else:
            display = f"{value_f:.2f}"
        rounded_rect(draw, (right_x + 14, yy, right_x + right_w - 14, yy + 78), 10, (18, 26, 40), MUTED_COLOR, 1)
        draw.text((right_x + 26, yy + 12), truncate_text(label, 26), font=mono_font, fill=MUTED_COLOR)
        draw.text((right_x + 26, yy + 36), display, font=kpi_font, fill=CYAN)
        yy += 90

    # Scene stage
    stage_x, stage_y, stage_w, stage_h = 20, 530, 1240, 170
    rounded_rect(draw, (stage_x, stage_y, stage_x + stage_w, stage_y + stage_h), 12, PANEL_COLOR, AMBER, 1)
    draw.text((stage_x + 14, stage_y + 10), "Lifecycle scene", font=small_font, fill=AMBER)
    scene = current_scene_from_events(events)
    floor_y = stage_y + 135
    draw.line((stage_x + 30, floor_y, stage_x + stage_w - 30, floor_y), fill=MUTED_COLOR, width=2)
    if not scene:
        draw.text((stage_x + 420, stage_y + 70), "No visual scene; replaying log only", font=small_font, fill=MUTED_COLOR)
    else:
        sx = stage_x + 80 + float(scene.get("silhouette_x", 0.4)) * (stage_w - 160)
        actor_state = _clean_str(scene.get("actor_state", "standing"))
        scale = 0.9 if actor_state != "gone" else 0.5
        if actor_state != "gone":
            draw_silhouette(draw, sx, floor_y, scale=scale)
        badges = scene.get("badges", []) if isinstance(scene.get("badges"), list) else []
        slots = []
        for i in range(min(len(badges), 10)):
            col = i % 5
            row = 0 if i < 5 else 1
            bx = stage_x + 110 + col * 220
            by = stage_y + 45 + row * 45
            slots.append((bx, by))
        for badge, (bx, by) in zip(badges, slots):
            label = _clean_str(badge.get("label", "?"))
            state = _clean_str(badge.get("state", "active"))
            if state in {"appearing", "active"}:
                color = GREEN if state == "active" else CYAN
            elif state in {"shedding", "dissolving"}:
                color = AMBER
            elif state == "dissolved":
                color = GRAY
            else:
                color = TEXT_COLOR
            rounded_rect(draw, (bx, by, bx + 120, by + 30), 8, (20, 28, 42), color, 2)
            draw.text((bx + 12, by + 7), truncate_text(label, 10), font=mono_font, fill=color)
            if state in {"shedding", "dissolving"}:
                draw.line((bx + 8, by + 15, bx + 112, by + 15), fill=color, width=2)

    return img


def export_mp4(
    title: str,
    agents: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    kpi_fn: Callable[[int, Dict[str, Any]], Dict[str, Any]],
    fps: int = 10,
    hold: int = 8,
) -> bytes:
    if not events:
        events = [
            {
                "ts": now_iso(),
                "agent": "Simulation",
                "level": "info",
                "msg": "No events available",
                "kpi": {},
            }
        ]
    ensure_dirs()
    tmp = tempfile.NamedTemporaryFile(prefix="lifecyclebot_", suffix=".mp4", delete=False)
    tmp.close()
    try:
        writer = imageio.get_writer(
            tmp.name,
            mode="I",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        total_frames = len(events) * max(1, hold)
        frame_no = 0
        for idx, event in enumerate(events):
            partial = events[: idx + 1]
            kpis = kpi_fn(idx, event)
            for _ in range(max(1, hold)):
                frame_no += 1
                progress = frame_no / float(total_frames)
                pil_img = render_frame(title, agents, partial, kpis, progress)
                writer.append_data(np.array(pil_img))
        writer.close()
        return Path(tmp.name).read_bytes()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def scriptwriter_events(scenario: str, log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    def add(agent: str, level: str, msg: str, scene: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "ts": now_iso(),
            "agent": agent,
            "level": level,
            "msg": msg,
            "kpi": {"scene": sanitize_for_log(scene)} if scene else {},
        }
        events.append(event)

    if scenario == "Active attack":
        badges = [{"label": "DIR", "state": "active"}, {"label": "CODE", "state": "active"}]
        add("HR Event Listener", "info", "Active attack scenario: suspicious mover detected",
            {"actor": "mover", "silhouette_x": 0.35, "actor_state": "standing", "badges": badges})
        badges.append({"label": "ADMIN", "state": "appearing"})
        add("Access Reviewer", "crit", "Unauthorized admin badge appears outside role matrix",
            {"actor": "mover", "silhouette_x": 0.38, "actor_state": "standing", "badges": badges})
        add("Critic", "crit", "Lead: privilege escalation candidate requires analyst review",
            {"actor": "mover", "silhouette_x": 0.40, "actor_state": "standing", "badges": badges})
        badges.append({"label": "ORPH", "state": "appearing"})
        add("Offboarding Agent", "warn", "Orphan lab account observed in attack path",
            {"actor": "mover", "silhouette_x": 0.42, "actor_state": "standing", "badges": badges})
        for badge in badges:
            if badge["label"] in {"ADMIN", "ORPH"}:
                badge["state"] = "dissolving"
        add("Executor", "warn", "Containment is dry-run only; no real defensive action executed",
            {"actor": "mover", "silhouette_x": 0.45, "actor_state": "standing", "badges": badges})
        badges = [b for b in badges if b["label"] not in {"ADMIN", "ORPH"}]
        add("Critic", "ok", "Attack-path leads preserved for human review",
            {"actor": "mover", "silhouette_x": 0.48, "actor_state": "standing", "badges": badges})
        for _ in range(12):
            add("Orchestrator", "info", "Scenario telemetry tick", None)
        return events

    if scenario == "Recovery":
        badges = [{"label": "OLD", "state": "active"}, {"label": "NEW", "state": "active"}]
        add("HR Event Listener", "info", "Recovery scenario: restore least privilege",
            {"actor": "mover", "silhouette_x": 0.35, "actor_state": "standing", "badges": badges})
        badges[0]["state"] = "shedding"
        add("Access Reviewer", "warn", "Legacy badge scheduled for removal",
            {"actor": "mover", "silhouette_x": 0.40, "actor_state": "standing", "badges": badges})
        badges = [b for b in badges if b["label"] != "OLD"]
        add("Executor", "ok", "Approved lab revocation completes recovery step",
            {"actor": "mover", "silhouette_x": 0.45, "actor_state": "standing", "badges": badges})
        badges.append({"label": "CERT", "state": "appearing"})
        add("Offboarding Agent", "ok", "Recovery certificate drafted",
            {"actor": "mover", "silhouette_x": 0.50, "actor_state": "standing", "badges": badges})
        for _ in range(14):
            add("Orchestrator", "info", "Recovery telemetry tick", None)
        return events

    # Normal day fallback
    emp = "EMP-DEMO"
    badges: List[Dict[str, Any]] = []
    add("HR Event Listener", "info", f"Joiner {emp} enters lab",
        {"actor": "joiner", "silhouette_x": 0.2, "actor_state": "walking_in", "badges": badges})
    for label in ["DIR", "MAIL", "CHAT", "CODE", "TICK"]:
        for b in badges:
            b["state"] = "active"
        badges.append({"label": label, "state": "appearing"})
        add("Provisioner", "ok", f"{emp} badge {label} provisioned",
            {"actor": "joiner", "silhouette_x": 0.25 + len(badges) * 0.03, "actor_state": "walking_in", "badges": badges})
    add("Critic", "ok", f"Joiner {emp} verified",
        {"actor": "joiner", "silhouette_x": 0.45, "actor_state": "standing", "badges": badges})
    add("HR Event Listener", "info", "Mover sheds legacy badge",
        {"actor": "mover", "silhouette_x": 0.50, "actor_state": "standing", "badges": badges})
    badges = [b for b in badges if b["label"] != "CODE"]
    add("Access Reviewer", "warn", "Privilege creep lead closed by dry-run removal",
        {"actor": "mover", "silhouette_x": 0.55, "actor_state": "standing", "badges": badges})
    add("HR Event Listener", "info", "Leaver exits",
        {"actor": "leaver", "silhouette_x": 0.60, "actor_state": "walking_out", "badges": badges})
    for label in ["DIR", "MAIL", "CHAT", "TICK"]:
        badges = [b for b in badges if b["label"] != label]
        add("Offboarding Agent", "ok", f"Leaver badge {label} dissolved",
            {"actor": "leaver", "silhouette_x": 0.65 + (4 - len(badges)) * 0.03, "actor_state": "walking_out", "badges": badges})
    add("Executor", "ok", "Certificate prepared",
        {"actor": "leaver", "silhouette_x": 0.85, "actor_state": "gone", "badges": []})
    return events


def get_simulation_events(scenario: str, log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scene_events = [
        event
        for event in log
        if isinstance(event.get("kpi"), dict) and isinstance(event["kpi"].get("scene"), dict)
    ]
    if scenario == "Normal day" and 15 <= len(scene_events) <= 45:
        return scene_events
    return scriptwriter_events(scenario, log)


def play_live_simulation(
    events: List[Dict[str, Any]],
    agents: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    frame_slot,
    progress_slot,
    speed: float = 0.2,
) -> None:
    total = max(1, len(events))
    for idx, event in enumerate(events, start=1):
        progress = idx / float(total)
        img = render_frame("LifecycleBot Simulation", agents, events[:idx], metrics, progress)
        frame_slot.image(np.array(img), use_container_width=True)
        progress_slot.progress(progress, text=f"Event {idx}/{total}: {truncate_text(event.get('msg', ''), 70)}")
        time.sleep(max(0.01, float(speed)))


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #07090F;
            color: #DCE7F5;
            font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }
        header, [data-testid="column"], .stSidebar, .stSidebar > div:first-child {
            background-color: #07090F !important;
        }
        h1, h2, h3 {
            color: #00E5FF !important;
        }
        p, li, label {
            color: #DCE7F5 !important;
        }
        .stButton > button, .stDownloadButton > button {
            background-color: #0E131E;
            color: #39FF88;
            border: 1px solid #00E5FF;
            border-radius: 8px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #102033;
            color: #00E5FF;
        }
        [data-testid="stMetricValue"] {
            color: #39FF88 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #8AA0B4 !important;
        }
        code {
            color: #00E5FF !important;
            background-color: #0E131E !important;
        }
        .stDataFrame {
            background-color: #0E131E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(metrics: Dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Provisioning accuracy", f"{float(metrics.get('provisioning_accuracy', 0)) * 100:.1f}%")
    cols[1].metric("Mean time to revoke", f"{float(metrics.get('mean_time_to_revoke_hours', 0)):.2f} h")
    cols[2].metric("Creep detection precision", f"{float(metrics.get('creep_detection_precision', 0)) * 100:.1f}%")
    cols[3].metric("Orphan detection rate", f"{float(metrics.get('orphan_detection_rate', 0)) * 100:.1f}%")


def render_automatic_demo(agent_roster: List[Dict[str, Any]], log: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    st.markdown("### Automatic working demo")
    frame_slot = st.empty()
    progress_slot = st.empty()
    note_slot = st.empty()
    video_slot = st.empty()

    if not st.session_state.get("auto_demo_done", False):
        events = get_simulation_events("Normal day", log)
        note_slot.write("Starting deterministic crew and live Simulation Studio...")
        try:
            play_live_simulation(events, agent_roster, metrics, frame_slot, progress_slot, speed=0.2)
        except Exception as exc:
            note_slot.warning(f"Live playback stopped safely: {type(exc).__name__}. Continuing to video render.")
        try:
            note_slot.write("Rendering local H.264 MP4 from the same event stream...")
            mp4 = export_mp4(
                "LifecycleBot Demo",
                agent_roster,
                events,
                lambda idx, event: metrics,
                fps=10,
                hold=8,
            )
            st.session_state["auto_mp4"] = mp4
            video_slot.video(mp4, format="video/mp4")
            note_slot.success("Local MP4 generated. Scroll down to interact with tabs and controls.")
            try:
                ensure_dirs()
                (REPORTS_DIR / "simulation_demo.mp4").write_bytes(mp4)
                (REPORTS_DIR / "events.json").write_text(
                    json.dumps(events, indent=2, default=str), encoding="utf-8"
                )
            except Exception:
                pass
        except Exception as exc:
            st.session_state["auto_mp4"] = None
            note_slot.error(
                "Video rendering failed on this machine, but the app remains usable. "
                f"Reason: {type(exc).__name__}. Check that imageio-ffmpeg installed correctly."
            )
        st.session_state["auto_demo_done"] = True
    else:
        events = get_simulation_events("Normal day", log)
        if events:
            img = render_frame("LifecycleBot Simulation", agent_roster, events, metrics, 1.0)
            frame_slot.image(np.array(img), use_container_width=True)
        mp4 = st.session_state.get("auto_mp4")
        if mp4:
            video_slot.video(mp4, format="video/mp4")


def event_feed_tab(processed: Dict[str, Any]) -> None:
    st.subheader("Event Feed")
    results = pd.DataFrame(processed.get("results", []))
    if results.empty:
        st.info("No processed events available.")
        return
    try:
        fig = px.scatter(
            results,
            x="effective_date",
            y="event_type",
            color="verdict",
            hover_data=["employee_id", "event_id", "action_count"],
            template="plotly_dark",
            title="HR lifecycle events and critic verdicts",
        )
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart unavailable: {type(exc).__name__}")
    st.dataframe(results, use_container_width=True)
    with st.expander("Shared live log tail"):
        log_rows = [
            {
                "ts": event.get("ts", ""),
                "agent": event.get("agent", ""),
                "level": event.get("level", ""),
                "msg": event.get("msg", ""),
            }
            for event in processed.get("log", [])[-200:]
        ]
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)


def access_matrix_tab(processed: Dict[str, Any]) -> None:
    st.subheader("Access Matrix")
    projection = processed.get("projection", {}) or {}
    rows: List[Dict[str, Any]] = []
    for emp, access in projection.items():
        for item in sorted(access):
            if ":" not in item:
                continue
            system, ent = item.split(":", 1)
            rows.append({"employee_id": emp, "system": system, "entitlement": ent, "status": "projected"})
    if not rows:
        st.info("No projected access yet. Run Reprocess lifecycle events.")
        return
    df = pd.DataFrame(rows)
    selected = st.multiselect("Filter systems", sorted(df["system"].unique()), default=sorted(df["system"].unique()))
    df_view = df[df["system"].isin(selected)] if selected else df
    try:
        counts = df_view.groupby(["system", "entitlement"]).size().reset_index(name="employees")
        fig = px.bar(counts, x="system", y="employees", color="entitlement", template="plotly_dark",
                     title="Projected entitlement density")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart unavailable: {type(exc).__name__}")
    st.dataframe(df_view, use_container_width=True)
    with st.expander("Role matrix bundles"):
        st.json(processed.get("role_matrix", {}))


def privilege_creep_tab(processed: Dict[str, Any]) -> None:
    st.subheader("Privilege Creep Report")
    findings = processed.get("creep_findings", [])
    if not findings:
        st.success("No privilege-creep leads were detected in the current run.")
        return
    df = pd.DataFrame(findings)
    try:
        fig = px.bar(df, x="system", color="severity", title="Creep leads by system and severity", template="plotly_dark")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart unavailable: {type(exc).__name__}")
    st.dataframe(df[["code", "severity", "system", "entitlement", "message"]], use_container_width=True)
    for idx, finding in enumerate(findings):
        with st.expander(f"Finding {idx + 1}: {finding.get('system')}/{finding.get('entitlement')}"):
            st.write(finding.get("message", ""))
            st.json(finding.get("evidence", {}))
            st.caption(f"Recommendation: {finding.get('recommendation', '')}")
            st.caption(f"Label: {finding.get('label', 'lead_for_analyst_review')}")


def render_pending_actions(processed: Dict[str, Any], dry_run: bool, ownership: bool) -> None:
    pending = processed.get("pending_actions", [])
    if not pending:
        st.info("No pending approvals. Dry-run demo actions are recorded but not persisted as executed.")
        return
    st.warning("Pending lab approvals. These actions are not real-system actions.")
    for idx, item in enumerate(list(pending)):
        action = item.get("action", {})
        col1, col2 = st.columns([4, 1])
        col1.markdown(
            f"**{action.get('type')}** {action.get('system')}/{action.get('entitlement')} "
            f"for {action.get('employee_id')} (event {action.get('event_id')})"
        )
        if col2.button("Approve lab action", key=f"approve_{action.get('id')}_{idx}"):
            if action.get("requires_ownership") and not ownership:
                st.error("Ownership confirmation is required before executing lab state changes.")
                continue
            gate = approval_gate(action, approved=True, dry_run=dry_run)
            conn = get_conn(DB_PATH)
            try:
                init_db(conn)
                if gate["status"] == "executed":
                    apply_lab_action(conn, action)
                audit_append(conn, "Human Approver", action.get("type", ""), action.get("employee_id", ""), gate["status"], action)
                conn.commit()
                if gate["status"] in {"executed", "dry_run"}:
                    update_projection_from_gates(processed.get("projection", {}), [gate])
                if gate["status"] != "pending_approval":
                    processed["pending_actions"].pop(idx)
                st.success(f"Approval recorded: {gate['status']}")
                st.rerun()
            finally:
                conn.close()


def offboarding_tab(processed: Dict[str, Any], dry_run: bool, ownership: bool) -> None:
    st.subheader("Offboarding Checklist")
    checklists = processed.get("checklists", [])
    certificates = processed.get("certificates", [])
    if not checklists:
        st.info("No exit checklists generated.")
    else:
        labels = [f"{c['employee_id']} - {c['event_id']} - {c['status']}" for c in checklists]
        choice = st.selectbox("Select offboarding checklist", labels)
        selected = checklists[labels.index(choice)]
        steps_df = pd.DataFrame(selected.get("steps", []))
        if not steps_df.empty:
            st.dataframe(steps_df[["order", "type", "system", "entitlement", "risk", "status"]], use_container_width=True)
        render_pending_actions(processed, dry_run, ownership)

    st.divider()
    st.subheader("Completion certificates")
    if not certificates:
        st.info("No certificates generated.")
        return
    cert_labels = [f"{c['certificate_id']} - {c['employee_id']} - {c['status']}" for c in certificates]
    cert_choice = st.selectbox("Select certificate", cert_labels)
    cert = certificates[cert_labels.index(cert_choice)]
    st.text_area("Certificate text", cert.get("text", ""), height=320)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download certificate TXT",
        cert.get("text", "").encode("utf-8"),
        file_name=f"{cert.get('certificate_id', 'certificate')}.txt",
        mime="text/plain",
    )
    c2.download_button(
        "Download certificate JSON",
        json.dumps(cert, indent=2, default=str).encode("utf-8"),
        file_name=f"{cert.get('certificate_id', 'certificate')}.json",
        mime="application/json",
    )

    st.divider()
    st.subheader("Orphan and shared account leads")
    orphan_df = pd.DataFrame(processed.get("orphan_report", []))
    shared_df = pd.DataFrame(processed.get("shared_report", []))
    if orphan_df.empty:
        st.success("No orphan lab accounts detected.")
    else:
        st.dataframe(orphan_df, use_container_width=True)
    if shared_df.empty:
        st.success("No shared lab accounts or entitlements detected.")
    else:
        st.dataframe(shared_df, use_container_width=True)


def simulation_tab(agent_roster: List[Dict[str, Any]], log: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    st.subheader("Simulation Studio")
    scenario = st.selectbox("Scenario", ["Normal day", "Active attack", "Recovery"])
    speed = st.slider("Speed (seconds per event)", min_value=0.2, max_value=2.0, value=0.2, step=0.1)
    events = get_simulation_events(scenario, log)

    col1, col2 = st.columns(2)
    frame_slot = st.empty()
    progress_slot = st.empty()
    video_slot = st.empty()
    note_slot = st.empty()

    if col1.button("Run live simulation", key=f"run_{scenario}"):
        try:
            play_live_simulation(events, agent_roster, metrics, frame_slot, progress_slot, speed=speed)
        except Exception as exc:
            note_slot.error(f"Live simulation failed safely: {type(exc).__name__}")

    if col2.button("Record video", key=f"record_{scenario}"):
        try:
            with note_slot.container():
                st.write("Rendering MP4 locally...")
            mp4 = export_mp4(
                f"LifecycleBot {scenario}",
                agent_roster,
                events,
                lambda idx, event: metrics,
                fps=10,
                hold=8,
            )
            st.session_state[f"mp4_{scenario}"] = mp4
            try:
                ensure_dirs()
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario.lower())
                (REPORTS_DIR / f"simulation_{safe_name}.mp4").write_bytes(mp4)
            except Exception:
                pass
            note_slot.success("MP4 ready.")
        except Exception as exc:
            st.session_state[f"mp4_{scenario}"] = None
            note_slot.error(f"Video export failed: {type(exc).__name__}. Install imageio-ffmpeg and retry.")

    mp4 = st.session_state.get(f"mp4_{scenario}")
    if scenario == "Normal day" and not mp4:
        mp4 = st.session_state.get("auto_mp4")
    if mp4:
        video_slot.video(mp4, format="video/mp4")
        st.download_button(
            "Download simulation.mp4",
            mp4,
            file_name="simulation.mp4",
            mime="video/mp4",
            key=f"dl_mp4_{scenario}",
        )
    st.download_button(
        "Download events.json",
        json.dumps(events, indent=2, default=str).encode("utf-8"),
        file_name="events.json",
        mime="application/json",
        key=f"dl_json_{scenario}",
    )
    events_df = pd.DataFrame(
        [
            {
                "ts": e.get("ts", ""),
                "agent": e.get("agent", ""),
                "level": e.get("level", ""),
                "msg": e.get("msg", ""),
            }
            for e in events
        ]
    )
    st.dataframe(events_df, use_container_width=True)


def render_builtin_checks() -> None:
    checks = run_self_checks()
    for name, passed, detail in checks:
        if passed:
            st.success(f"{name}: {detail}")
        else:
            st.error(f"{name}: {detail}")


def main() -> None:
    st.set_page_config(
        page_title="LifecycleBot",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    st.title("LifecycleBot")
    st.caption("Joiner-mover-leaver automation lab with approval-gated execution and local Simulation Studio.")

    st.sidebar.header("Control")
    api_key = st.sidebar.text_input(
        "Optional Gemini API key",
        type="password",
        value="",
        help="Leave blank for Demo Mode. The app is fully functional without a key.",
    )
    mode = st.sidebar.radio("Data mode", ["Demo", "Live"], index=0)
    try:
        dry_run = st.sidebar.toggle("Dry-run (safe default)", value=True, help="Dry-run records plans without persisting lab state changes.")
    except AttributeError:
        dry_run = st.sidebar.checkbox("Dry-run (safe default)", value=True, help="Dry-run records plans without persisting lab state changes.")
    auto_approve = st.sidebar.checkbox(
        "Auto-approve lab-only actions",
        value=False,
        help="Only affects the local SQLite lab. Real systems are never targeted.",
    )
    ownership = st.sidebar.checkbox(
        "I confirm synthetic lab data and no real host/domain",
        value=False,
        help="Required for any executed lab state change.",
    )

    if st.sidebar.button("Reprocess lifecycle events"):
        st.session_state.pop("processed", None)
        st.session_state["auto_demo_done"] = False
        st.session_state["auto_mp4"] = None
        st.rerun()

    if st.session_state.get("last_mode") != mode:
        st.session_state.pop("processed", None)
        st.session_state["auto_demo_done"] = False
        st.session_state["auto_mp4"] = None
        st.session_state["last_mode"] = mode

    if "processed" not in st.session_state:
        role_matrix = load_role_matrix()
        log_list: List[Dict[str, Any]] = []
        conn = None
        try:
            with st.spinner("Loading synthetic HR events and running deterministic crew..."):
                ensure_data_files()
                events_df = load_events(mode)
                conn = get_conn(DB_PATH)
                init_db(conn)
                seed_lab(conn)
                processed = process_all_events(
                    events_df=events_df,
                    role_matrix=role_matrix,
                    on_event=log_list.append,
                    api_key=api_key or None,
                    dry_run=dry_run,
                    approved=auto_approve,
                    ownership=ownership,
                    conn=conn,
                )
                processed["log"] = log_list
                st.session_state["processed"] = processed
                st.session_state["log"] = log_list
                st.session_state["auto_demo_done"] = False
                st.session_state["auto_mp4"] = None
        except Exception as exc:
            st.error(f"Startup processing failed safely: {type(exc).__name__}. Using empty demo state.")
            log_list.append(
                {
                    "ts": now_iso(),
                    "agent": "Orchestrator",
                    "level": "crit",
                    "msg": f"Processing error: {type(exc).__name__}",
                    "kpi": {},
                }
            )
            st.session_state["processed"] = empty_processed(role_matrix)
            st.session_state["processed"]["log"] = log_list
            st.session_state["log"] = log_list
        finally:
            if conn is not None:
                try:
                    conn.commit()
                except Exception:
                    pass
                conn.close()

    processed = st.session_state.get("processed", empty_processed())
    log = processed.get("log", st.session_state.get("log", []))
    metrics = processed.get("metrics", {})

    render_kpi_row(metrics)

    crew_for_ui = build_crew(lambda event: None)
    agent_roster = [{"name": agent.name, "role": agent.role} for agent in crew_for_ui.agents.values()]

    render_automatic_demo(agent_roster, log, metrics)

    st.divider()
    tabs = st.tabs(["Event Feed", "Access Matrix", "Privilege Creep Report", "Offboarding Checklist", "Simulation Studio"])
    with tabs[0]:
        event_feed_tab(processed)
    with tabs[1]:
        access_matrix_tab(processed)
    with tabs[2]:
        privilege_creep_tab(processed)
    with tabs[3]:
        offboarding_tab(processed, dry_run, ownership)
    with tabs[4]:
        simulation_tab(agent_roster, log, metrics)

    with st.sidebar.expander("Built-in verification"):
        st.caption("Runs embedded checks without separate test files.")
        if st.button("Run checks"):
            render_builtin_checks()

    with st.sidebar.expander("Audit chain"):
        try:
            conn = get_conn(DB_PATH)
            ok, bad_id = verify_audit(conn)
            if ok:
                st.success("Hash-chained audit trail verifies locally.")
            else:
                st.error(f"Audit chain broken at row {bad_id}.")
            rows = conn.execute("SELECT ts, actor, action, target, status FROM audit ORDER BY id DESC LIMIT 20").fetchall()
            st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True)
            conn.close()
        except Exception as exc:
            st.warning(f"Audit view unavailable: {type(exc).__name__}")


# -----------------------------------------------------------------------------
# Embedded tests / self checks
# -----------------------------------------------------------------------------

def test_role_matrix_has_five_systems() -> None:
    rm = load_role_matrix_embedded()
    assert isinstance(rm.get("systems"), list), "systems must be a list"
    assert len(rm["systems"]) == 5, "expected five mock systems"
    assert "software_engineer" in rm.get("roles", {}), "expected software_engineer role"


def test_approval_gate_dry_run_precedes_pending() -> None:
    action = {"type": "grant", "employee_id": "EMP-TEST"}
    assert approval_gate(action, approved=False, dry_run=True)["status"] == "dry_run"
    assert approval_gate(action, approved=False, dry_run=False)["status"] == "pending_approval"
    assert approval_gate(action, approved=True, dry_run=False)["status"] == "executed"


def test_duplicate_event_is_handled() -> None:
    listener = {
        "event_id": "EV-DUP",
        "employee_id": "EMP-DUP",
        "event_type": "hire",
        "role": "software_engineer",
        "effective_date": "2026-01-01T00:00:00+00:00",
        "received_at": "2026-01-01T00:00:00+00:00",
        "duplicate": True,
        "late": False,
    }
    critic = critic_demo({"HR Event Listener": listener}, {})
    codes = [f.get("code") for f in critic.get("findings", [])]
    assert "duplicate_event" in codes, "duplicate event must be flagged"
    assert critic.get("verdict") == "fail", "duplicate event should fail critic"


def test_missing_role_is_handled() -> None:
    rm = load_role_matrix_embedded()
    listener = {
        "event_id": "EV-MISS",
        "employee_id": "EMP-MISS",
        "event_type": "hire",
        "role": "unknown_role",
        "effective_date": "2026-01-01T00:00:00+00:00",
        "received_at": "2026-01-01T00:00:00+00:00",
        "duplicate": False,
        "late": False,
    }
    role_map = role_mapper_demo(listener, {"role_matrix": rm})
    assert role_map["missing_role"] is True
    critic = critic_demo({"HR Event Listener": listener, "Role Mapper": role_map}, {"role_matrix": rm})
    codes = [f.get("code") for f in critic.get("findings", [])]
    assert "missing_role" in codes


def test_privilege_creep_detected_after_grace() -> None:
    rm = load_role_matrix_embedded()
    listener = {
        "event_id": "EV-CREEP",
        "employee_id": "EMP-CREEP",
        "event_type": "transfer",
        "role": "it_admin",
        "previous_role": "software_engineer",
        "effective_date": "2026-01-01T00:00:00+00:00",
        "received_at": "2026-01-01T00:00:00+00:00",
        "duplicate": False,
        "late": False,
    }
    role_map = role_mapper_demo(listener, {"role_matrix": rm})
    ctx = {
        "HR Event Listener": listener,
        "role_matrix": rm,
        "access_state": {"EMP-CREEP": set()},
        "as_of": datetime(2026, 3, 1, tzinfo=timezone.utc),
    }
    review = reviewer_demo(role_map, ctx)
    assert review["creep_findings"], "past-grace legacy access should create creep findings"
    assert any(f.get("code") == "privilege_creep" for f in review["creep_findings"])


def test_offboarding_order_is_safe() -> None:
    rm = load_role_matrix_embedded()
    listener = {
        "event_id": "EV-EXIT",
        "employee_id": "EMP-EXIT",
        "event_type": "exit",
        "role": "software_engineer",
        "effective_date": "2026-01-01T00:00:00+00:00",
        "received_at": "2026-01-01T00:00:00+00:00",
        "duplicate": False,
        "late": False,
    }
    off = offboarding_demo(listener, {"role_matrix": rm, "access_state": {"EMP-EXIT": set()}, "conn": None})
    types = [a.get("type") for a in off.get("ordered_actions", [])]
    expected = [
        "revoke_sessions",
        "disable_mfa",
        "remove_mailbox_forwarding",
        "remove_groups",
        "revoke_licenses",
    ]
    assert types[:5] == expected, f"unsafe order: {types[:5]}"


def test_audit_hash_chain_verifies() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    audit_append(conn, "Tester", "grant", "EMP-1", "dry_run", {"x": 1})
    audit_append(conn, "Tester", "revoke", "EMP-1", "executed", {"x": 2})
    ok, bad = verify_audit(conn)
    assert ok and bad is None
    conn.execute("UPDATE audit SET detail='tampered' WHERE id=1")
    ok2, bad2 = verify_audit(conn)
    assert not ok2
    conn.close()


def test_orphan_accounts_detected() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    seed_lab(conn)
    orphans = detect_orphans(conn, set())
    assert len(orphans) >= 2, "seeded orphan accounts should be detected"
    conn.close()


def run_self_checks() -> List[Tuple[str, bool, str]]:
    checks = []
    tests = [
        test_role_matrix_has_five_systems,
        test_approval_gate_dry_run_precedes_pending,
        test_duplicate_event_is_handled,
        test_missing_role_is_handled,
        test_privilege_creep_detected_after_grace,
        test_offboarding_order_is_safe,
        test_audit_hash_chain_verifies,
        test_orphan_accounts_detected,
    ]
    for fn in tests:
        try:
            fn()
            checks.append((fn.__name__, True, "passed"))
        except AssertionError as exc:
            checks.append((fn.__name__, False, str(exc) or "assertion failed"))
        except Exception as exc:
            checks.append((fn.__name__, False, f"{type(exc).__name__}: {exc}"))
    return checks


if __name__ == "__main__":
    main()