import datetime
import logging
import os
import sqlite3
from typing import Any, Dict, Optional

TRAIN_JOB_OUTPUT_DIR = os.getenv("TRAIN_JOB_OUTPUT_DIR", "/tmp/trained_models/")
DB_PATH = os.path.join(TRAIN_JOB_OUTPUT_DIR, "training_jobs.db")


def get_db_connection():
    """Establishes a database connection."""
    logging.debug(f"Using sqlite database located in {DB_PATH}")
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes the database table.
    This is the primary place to define the table schema.
    """
    conn = get_db_connection()
    try:
        # best_finetuned_model_path is the path to latest checkpoint
        # folder will contain model weights and config that were saved with hf peft
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                detail TEXT,
                best_finetuned_model_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )
        conn.commit()
    finally:
        conn.close()


def create_job(job_id: str, **kwargs: Any):
    """
    Creates a new job record using provided data.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    job_data = {
        "id": job_id,
        "status": "queued",
        "detail": "Job is waiting to start.",
        "created_at": now,
        "updated_at": now,
    }
    job_data.update(kwargs)  # Overwrite defaults with provided data

    columns = ", ".join(job_data.keys())
    placeholders = ", ".join(["?"] * len(job_data))
    sql = f"INSERT INTO jobs ({columns}) VALUES ({placeholders})"

    conn = get_db_connection()
    try:
        conn.execute(sql, tuple(job_data.values()))
        conn.commit()
    finally:
        conn.close()


def update_job(job_id: str, **kwargs: Any):
    """
    Updates an existing job with the given key-value pairs.
    """
    if not kwargs:
        return  # Nothing to update

    print(f"Updating job {job_id}")
    # Automatically update the 'updated_at' timestamp
    kwargs["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
    sql = f"UPDATE jobs SET {set_clause} WHERE id = ?"

    conn = get_db_connection()
    try:
        conn.execute(sql, (*kwargs.values(), job_id))
        conn.commit()
    finally:
        conn.close()


def get_job(job_id: str) -> Optional[Dict]:
    """Fetches a job record by its ID and returns it as a dictionary."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_jobs() -> Optional[list[Dict]]:
    """Fetches all job records and returns them as a list of dictionaries."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs")
        rows = cursor.fetchall()
        return [dict(row) for row in rows] if rows else []
    finally:
        conn.close()


def create_job_callback(job_id):
    """Create a job callback function that you can pass to training loop if needed

    Args:
        job_id (str): The ID of the job to update.
    """

    def callback(**kwargs):
        update_job(job_id, **kwargs)

    return callback
