import hashlib
import json
import os
import random
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DB_PATH = Path(os.environ.get("AB_DB", "study.db"))
PAIRS_PATH = Path(os.environ.get("AB_PAIRS", "pairs.json"))
TOKENS_PATH = Path(os.environ.get("AB_TOKENS", "tokens.json"))
IP_SALT = os.environ.get("AB_IP_SALT", "dev-salt")
ALLOW_OPEN = os.environ.get("AB_ALLOW_OPEN", "0") == "1"

EXPERIENCE_BRACKETS = {"none", "casual", "1-4digit", "5-6digit", "lapsed"}
# none: never played | casual: plays, unranked/low | 1-4digit / 5-6digit:
# global rank bracket | lapsed: used to play seriously

app = FastAPI(title="osunator A/B study", docs_url=None, redoc_url=None)


# storage

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS raters (
                id           TEXT PRIMARY KEY,
                token        TEXT,
                cohort       TEXT NOT NULL,
                experience   TEXT NOT NULL,
                started_at   TEXT NOT NULL,
                submitted_at TEXT,
                user_agent   TEXT,
                ip_hash      TEXT
            );
            CREATE TABLE IF NOT EXISTS responses (
                rater_id        TEXT NOT NULL REFERENCES raters(id),
                pair_id         TEXT NOT NULL,
                chosen_slot     TEXT NOT NULL CHECK (chosen_slot IN ('A','B')),
                slot_a_is_human INTEGER NOT NULL,
                decision_ms     INTEGER,
                revised         INTEGER NOT NULL DEFAULT 0,
                answered_at     TEXT NOT NULL,
                PRIMARY KEY (rater_id, pair_id)
            );
            CREATE TABLE IF NOT EXISTS used_tokens (
                token    TEXT PRIMARY KEY,
                rater_id TEXT NOT NULL,
                used_at  TEXT NOT NULL
            );
            """
        )


def load_pairs() -> dict:
    """pairs.json: {pair_id: {"a": url, "b": url, "a_is_human": bool,
                              "map": str, "section": str}}"""
    if not PAIRS_PATH.exists():
        raise RuntimeError(f"missing {PAIRS_PATH}")
    data = json.loads(PAIRS_PATH.read_text())
    for pid, p in data.items():
        for key in ("a", "b", "a_is_human"):
            if key not in p:
                raise RuntimeError(f"pairs.json: pair {pid} missing '{key}'")
    return data


def load_tokens() -> dict:
    """tokens.json: {token: cohort}. Optional file."""
    if TOKENS_PATH.exists():
        return json.loads(TOKENS_PATH.read_text())
    return {}


PAIRS: dict = {}
TOKENS: dict = {}


@app.on_event("startup")
def startup() -> None:
    global PAIRS, TOKENS
    init_db()
    PAIRS = load_pairs()
    TOKENS = load_tokens()


# models

class SessionIn(BaseModel):
    experience: str
    token: str | None = None


class ResponseIn(BaseModel):
    rater_id: str
    pair_id: str
    chosen_slot: str = Field(pattern="^[AB]$")
    decision_ms: int | None = None


class SubmitIn(BaseModel):
    rater_id: str


# helpers

def ip_hash(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host or "?")
    return hashlib.sha256(f"{IP_SALT}:{ip}".encode()).hexdigest()[:16]


def get_rater(conn, rater_id: str):
    row = conn.execute("SELECT * FROM raters WHERE id = ?", (rater_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown rater")
    return row


# routes

@app.post("/api/session")
def create_session(body: SessionIn, request: Request):
    if body.experience not in EXPERIENCE_BRACKETS:
        raise HTTPException(422, f"experience must be one of {sorted(EXPERIENCE_BRACKETS)}")

    if body.token:
        if body.token not in TOKENS:
            raise HTTPException(403, "invalid invite token")
        cohort = TOKENS[body.token]
    elif ALLOW_OPEN:
        cohort = "open"
    else:
        raise HTTPException(403, "invite token required")

    rater_id = str(uuid.uuid4())

    with db() as conn:
        if body.token:
            used = conn.execute(
                "SELECT rater_id FROM used_tokens WHERE token = ?", (body.token,)
            ).fetchone()
            if used is not None:
                raise HTTPException(403, "token already used")
            conn.execute(
                "INSERT INTO used_tokens (token, rater_id, used_at) VALUES (?,?,?)",
                (body.token, rater_id, utcnow()),
            )
        conn.execute(
            "INSERT INTO raters (id, token, cohort, experience, started_at,"
            " user_agent, ip_hash) VALUES (?,?,?,?,?,?,?)",
            (rater_id, body.token, cohort, body.experience, utcnow(),
             request.headers.get("user-agent", "")[:200], ip_hash(request)),
        )

    # per-rater order shuffle + per-pair left/right shuffle, no labels
    order = list(PAIRS.keys())
    random.shuffle(order)
    pairs_out = [
        {
            "pair_id": pid,
            "slots": {"A": PAIRS[pid]["a"], "B": PAIRS[pid]["b"]},
            "left_slot": random.choice(["A", "B"]),
        }
        for pid in order
    ]
    return {"rater_id": rater_id, "pairs": pairs_out}


@app.post("/api/response")
def record_response(body: ResponseIn):
    if body.pair_id not in PAIRS:
        raise HTTPException(404, "unknown pair")
    with db() as conn:
        rater = get_rater(conn, body.rater_id)
        if rater["submitted_at"] is not None:
            raise HTTPException(409, "already submitted")
        existing = conn.execute(
            "SELECT 1 FROM responses WHERE rater_id = ? AND pair_id = ?",
            (body.rater_id, body.pair_id),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO responses
                (rater_id, pair_id, chosen_slot, slot_a_is_human,
                 decision_ms, revised, answered_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(rater_id, pair_id) DO UPDATE SET
                chosen_slot = excluded.chosen_slot,
                decision_ms = excluded.decision_ms,
                revised     = 1,
                answered_at = excluded.answered_at
            """,
            (body.rater_id, body.pair_id, body.chosen_slot,
             int(PAIRS[body.pair_id]["a_is_human"]),
             body.decision_ms, 1 if existing else 0, utcnow()),
        )
    return {"ok": True, "revised": bool(existing)}


@app.post("/api/submit")
def submit(body: SubmitIn):
    with db() as conn:
        rater = get_rater(conn, body.rater_id)
        if rater["submitted_at"] is not None:
            raise HTTPException(409, "already submitted")
        rows = conn.execute(
            "SELECT pair_id, chosen_slot, slot_a_is_human FROM responses"
            " WHERE rater_id = ?", (body.rater_id,),
        ).fetchall()
        missing = sorted(set(PAIRS) - {r["pair_id"] for r in rows})
        if missing:
            raise HTTPException(409, f"unanswered pairs: {missing}")
        conn.execute(
            "UPDATE raters SET submitted_at = ? WHERE id = ?",
            (utcnow(), body.rater_id),
        )
    # reveal AFTER submit is locked in
    correct = sum(
        (r["chosen_slot"] == "A") == bool(r["slot_a_is_human"]) for r in rows
    )
    reveal = [
        {"pair_id": r["pair_id"],
         "human_slot": "A" if r["slot_a_is_human"] else "B",
         "your_choice": r["chosen_slot"]}
        for r in rows
    ]
    return {"score": correct, "total": len(rows), "reveal": reveal}


@app.get("/")
def root():
    return RedirectResponse("/static/index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"),
          name="static")