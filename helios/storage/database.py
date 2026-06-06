import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent.parent / "helios_eval.db"

def init_db():
    """Initializes the relational database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        run_id TEXT PRIMARY KEY,
        timestamp DATETIME,
        target_model TEXT,
        notes TEXT
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_cases (
        case_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        question TEXT,
        ground_truth TEXT,
        difficulty TEXT,
        FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id)
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics_results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id INTEGER,
        system_answer TEXT,
        score INTEGER,
        hallucination_flag BOOLEAN,
        reasoning TEXT,
        FOREIGN KEY(case_id) REFERENCES test_cases(case_id)
    )''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized successfully at {DB_PATH}")

def save_test_cases(run_id: str, target_model: str, cases: list):
    """Saves a list of generated Pydantic test cases to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO evaluation_runs (run_id, timestamp, target_model) VALUES (?, ?, ?)",
        (run_id, datetime.now(), target_model)
    )

    for case in cases:
        cursor.execute(
            "INSERT INTO test_cases (run_id, question, ground_truth, difficulty) VALUES (?, ?, ?, ?)",
            (run_id, case.question, case.ground_truth, case.difficulty)
        )
        
    conn.commit()
    conn.close()
    print(f"Saved {len(cases)} test cases to database under Run ID: {run_id}")

if __name__ == "__main__":
    init_db()