import os
import psycopg2
from psycopg2.extras import RealDictCursor

# =========================================================
# Neon Postgres connection
# =========================================================
# Set DATABASE_URL to the connection string Neon gives you, e.g.:
#   postgresql://user:password@ep-xxxx.neon.tech/wellness?sslmode=require
# Neon requires SSL — make sure sslmode=require (or similar) is in the string,
# or the fallback below adds it automatically if it's missing.

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and "sslmode" not in DATABASE_URL:
    separator = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = f"{DATABASE_URL}{separator}sslmode=require"

def get_db_connection():
    """Opens a fresh connection to the Neon Postgres database."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set. Add your Neon connection string as an env var.")
    return psycopg2.connect(DATABASE_URL)

def initialize_database():
    """Creates the necessary tables if they don't exist."""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            age INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            injuries TEXT,
            goals TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS weekly_plans (
            plan_id SERIAL PRIMARY KEY,
            user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
            week_number INTEGER NOT NULL,
            workout_plan TEXT NOT NULL,
            yoga_plan TEXT NOT NULL,
            diet_plan TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, week_number)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            message_id SERIAL PRIMARY KEY,
            user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        conn.commit()
        cur.close()
        print("✅ Neon Postgres database initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing Postgres database: {e}")
    finally:
        if conn is not None:
            conn.close()

def save_user_profile(user_id: str, age: int, weight: float, injuries: str, goals: str):
    query = """
        INSERT INTO users (user_id, age, weight_kg, injuries, goals)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id)
        DO UPDATE SET
            age = EXCLUDED.age,
            weight_kg = EXCLUDED.weight_kg,
            injuries = EXCLUDED.injuries,
            goals = EXCLUDED.goals;
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, (user_id, age, weight, injuries, goals))
    conn.commit()
    cur.close()
    conn.close()

def get_user_profile_string(user_id: str) -> str:
    query = "SELECT age, weight_kg, injuries, goals FROM users WHERE user_id = %s;"
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return "No profile found for this User ID."

    profile_str = (
        f"User ID: {user_id}\n"
        f"- Age: {row['age']} years old\n"
        f"- Weight: {row['weight_kg']} kg\n"
        f"- Documented Injuries/Pains: {row['injuries'] if row['injuries'] else 'None'}\n"
        f"- Stated Goals: {row['goals'] if row['goals'] else 'None specified'}"
    )
    return profile_str

def save_weekly_plan(user_id: str, week_number: int, workout: str, yoga: str, diet: str):

    def clean_agent_output(output) -> str:
        if isinstance(output, list):
            extracted_chunks = []
            for item in output:
                if isinstance(item, dict) and "text" in item:
                    extracted_chunks.append(item["text"])
                elif isinstance(item, dict) and "content" in item:
                    extracted_chunks.append(item["content"])
                else:
                    extracted_chunks.append(str(item))
            return "\n".join(extracted_chunks)

        if isinstance(output, dict):
            return output.get("text", output.get("content", str(output)))

        return str(output) if output else ""

    workout = clean_agent_output(workout)
    yoga = clean_agent_output(yoga)
    diet = clean_agent_output(diet)

    query = """
        INSERT INTO weekly_plans (user_id, week_number, workout_plan, yoga_plan, diet_plan)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, week_number)
        DO UPDATE SET
            workout_plan = EXCLUDED.workout_plan,
            yoga_plan = EXCLUDED.yoga_plan,
            diet_plan = EXCLUDED.diet_plan;
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, (user_id, week_number, workout, yoga, diet))
    conn.commit()
    cur.close()
    conn.close()

def get_last_week_number(user_id: str) -> int:
    query = "SELECT COALESCE(MAX(week_number), 0) FROM weekly_plans WHERE user_id = %s;"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, (user_id,))
    max_week = cur.fetchone()[0]
    cur.close()
    conn.close()
    return max_week

# =========================================================
# Conversation memory
# =========================================================

def save_chat_message(user_id: str, role: str, message: str):
    if not message:
        return
    query = "INSERT INTO conversation_messages (user_id, role, message) VALUES (%s, %s, %s);"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, (user_id, role, message))
    conn.commit()
    cur.close()
    conn.close()

def get_recent_history(user_id: str, turns: int = 6) -> str:
    query = """
        SELECT role, message FROM conversation_messages
        WHERE user_id = %s
        ORDER BY message_id DESC
        LIMIT %s;
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, (user_id, turns))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return ""

    rows = list(reversed(rows))
    lines = []
    for row in rows:
        speaker = "User" if row["role"] == "user" else "Assistant"
        text = row["message"][:400]
        lines.append(f"{speaker}: {text}")

    return "\n".join(lines)
