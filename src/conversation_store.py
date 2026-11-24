"""Persistent SQLite conversations locally, existing Supabase store when configured."""
import os
import sqlite3
import uuid
from pathlib import Path
from src.config import SUPABASE_DB_URL

DB_PATH = os.getenv('CONVERSATIONS_DB', str(Path(__file__).resolve().parent.parent / 'conversations.db'))


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_conversation_tables():
    with _connect() as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, mode TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE INDEX IF NOT EXISTS messages_conv ON messages(conversation_id, id);
        ''')


def create_conversation(mode='text', title='New conversation'):
    conv_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute('INSERT INTO conversations(id,title,mode) VALUES (?,?,?)', (conv_id, title[:60], mode))
    return conv_id


def save_turn(conversation_id, user_message, assistant_message, is_first_turn=False):
    with _connect() as conn:
        conn.executemany('INSERT INTO messages(conversation_id,role,content) VALUES (?,?,?)',
                         [(conversation_id, 'user', user_message), (conversation_id, 'assistant', assistant_message)])
        if is_first_turn:
            conn.execute('UPDATE conversations SET title=? WHERE id=?', (user_message[:60], conversation_id))
        conn.execute('UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?', (conversation_id,))


def get_conversation_history(conversation_id, max_turns=5):
    with _connect() as conn:
        rows = conn.execute('SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?',
                            (conversation_id, max_turns * 2)).fetchall()
    return [dict(row) for row in reversed(rows)]


def list_conversations(limit=50):
    with _connect() as conn:
        return [dict(row) for row in conn.execute('''SELECT c.*, COUNT(m.id) message_count FROM conversations c
            LEFT JOIN messages m ON c.id=m.conversation_id GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?''', (limit,))]


def get_full_conversation(conversation_id):
    with _connect() as conn:
        conv = conn.execute('SELECT * FROM conversations WHERE id=?', (conversation_id,)).fetchone()
        if not conv:
            return None
        return {**dict(conv), 'messages': [dict(row) for row in conn.execute(
            'SELECT * FROM messages WHERE conversation_id=? ORDER BY id', (conversation_id,))]}


def delete_conversation(conversation_id):
    with _connect() as conn:
        return conn.execute('DELETE FROM conversations WHERE id=?', (conversation_id,)).rowcount > 0


if SUPABASE_DB_URL and not SUPABASE_DB_URL.startswith('sqlite:'):
    from src.postgres_conversations import (init_conversation_tables, create_conversation, save_turn,
        get_conversation_history, list_conversations, get_full_conversation, delete_conversation)
