"""Questions & answers board for Spec2MRM.

Anyone using the app can ask a question or post an answer. Developer answers
are badged, and only the developer can delete posts: on the local copy of the
app that is automatic; on the public site the developer signs in with the
admin password.

Storage: a local SQLite file, data/qa.sqlite3 (or the path in SPEC2MRM_QA_DB,
used by the tests). A public server's disk is wiped on every restart, so there
the board uses an online PostgreSQL database instead, given as a connection
URL in SPEC2MRM_QA_URL (set from the app's secrets).
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3

MAX_NAME, MAX_QUESTION, MAX_ANSWER = 60, 2000, 3000
DEFAULT_DB = Path(__file__).resolve().parent / 'data' / 'qa.sqlite3'


def db_path():
    return Path(os.environ.get('SPEC2MRM_QA_DB') or DEFAULT_DB)


def online():
    """True when the board is stored in an online PostgreSQL database."""
    return bool(os.environ.get('SPEC2MRM_QA_URL', '').strip())


SQLITE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY, author TEXT NOT NULL, body TEXT NOT NULL, created TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS answers (
        id INTEGER PRIMARY KEY,
        question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
        author TEXT NOT NULL, body TEXT NOT NULL, developer INTEGER NOT NULL DEFAULT 0,
        created TEXT NOT NULL);
"""
POSTGRES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS questions (
        id SERIAL PRIMARY KEY, author TEXT NOT NULL, body TEXT NOT NULL, created TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS answers (
        id SERIAL PRIMARY KEY,
        question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
        author TEXT NOT NULL, body TEXT NOT NULL, developer INTEGER NOT NULL DEFAULT 0,
        created TEXT NOT NULL);
"""


class Board:
    """One connection with '?' placeholders for either backend; use as a context manager."""

    def __init__(self):
        if online():
            import psycopg2
            self.con = psycopg2.connect(os.environ['SPEC2MRM_QA_URL'].strip(), connect_timeout=10)
            self.postgres = True
            with self.con, self.con.cursor() as cur:
                cur.execute(POSTGRES_SCHEMA)
        else:
            path = db_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(path, timeout=10)
            self.postgres = False
            self.con.execute('PRAGMA foreign_keys = ON')
            self.con.executescript(SQLITE_SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, kind, *_):
        try:
            if kind is None:
                self.con.commit()
            else:
                self.con.rollback()
        finally:
            self.con.close()

    def run(self, sql, params=(), returning=False):
        """Execute; returns all rows, or the new id when returning=True."""
        if self.postgres:
            sql = sql.replace('?', '%s') + (' RETURNING id' if returning else '')
            with self.con.cursor() as cur:
                cur.execute(sql, params)
                if returning:
                    return cur.fetchone()[0]
                return cur.fetchall() if cur.description else []
        cur = self.con.execute(sql, params)
        return cur.lastrowid if returning else cur.fetchall()


def _clean(text, limit, what):
    text = (text or '').strip()
    if not text:
        raise ValueError(f'Write your {what} first.')
    if len(text) > limit:
        raise ValueError(f'Keep your {what} under {limit} characters.')
    return ''.join(c for c in text if c in '\n\t' or ord(c) >= 32)


def _author(name, developer=False):
    name = (name or '').strip()[:MAX_NAME]
    return name or ('Developer' if developer else 'Anonymous')


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def ask(name, body):
    body = _clean(body, MAX_QUESTION, 'question')
    with Board() as db:
        return db.run('INSERT INTO questions (author, body, created) VALUES (?, ?, ?)',
                      (_author(name), body, _now()), returning=True)


def answer(question_id, name, body, developer=False):
    body = _clean(body, MAX_ANSWER, 'answer')
    with Board() as db:
        if not db.run('SELECT 1 FROM questions WHERE id = ?', (question_id,)):
            raise ValueError('That question no longer exists.')
        return db.run('INSERT INTO answers (question_id, author, body, developer, created) VALUES (?, ?, ?, ?, ?)',
                      (question_id, _author(name, developer), body, int(developer), _now()), returning=True)


def delete_question(question_id):
    with Board() as db:
        db.run('DELETE FROM questions WHERE id = ?', (question_id,))


def delete_answer(answer_id):
    with Board() as db:
        db.run('DELETE FROM answers WHERE id = ?', (answer_id,))


def board(search=''):
    """Questions newest first, each with its answers (developer answers first)."""
    with Board() as db:
        questions = db.run('SELECT id, author, body, created FROM questions ORDER BY id DESC')
        answers = db.run('SELECT id, question_id, author, body, developer, created FROM answers '
                         'ORDER BY developer DESC, id')
    by_question = {}
    for a in answers:
        by_question.setdefault(a[1], []).append(dict(id=a[0], author=a[2], body=a[3], developer=bool(a[4]), created=a[5]))
    out = [dict(id=q[0], author=q[1], body=q[2], created=q[3], answers=by_question.get(q[0], [])) for q in questions]
    term = (search or '').strip().lower()
    if term:
        out = [q for q in out if term in q['body'].lower() or any(term in a['body'].lower() for a in q['answers'])]
    return out
