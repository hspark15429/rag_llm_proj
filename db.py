import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('chat_history_db/chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS responses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query TEXT,
                  response TEXT,
                  latency REAL,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

def store_response(query: str, response: str, latency: float):
    conn = sqlite3.connect('chat_history_db/chat_history.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO responses (query, response, latency, timestamp) VALUES (?, ?, ?, ?)",
              (query, response, latency, timestamp))
    conn.commit()
    conn.close()

def get_previous_responses(limit: int = 5):
    conn = sqlite3.connect('chat_history_db/chat_history.db')
    c = conn.cursor()
    c.execute("SELECT query, response, latency, timestamp FROM responses ORDER BY timestamp DESC LIMIT ?",
              (limit,))
    results = c.fetchall()
    conn.close()
    return results