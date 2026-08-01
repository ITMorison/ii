import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "alerts_history.db")

def init_db():
    """Создаёт таблицу инцидентов при первом запуске."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_alert(camera_id: str, event_type: str, confidence: float):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, camera_id, event_type, confidence)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_id, event_type, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Ошибка при сохранении инцидента в БД: {e}")

def get_alerts_dataframe():
    """Возвращает историю алертов в формате DataFrame для графиков и таблиц."""
    if not os.path.exists(DB_PATH):
        init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM alerts ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df