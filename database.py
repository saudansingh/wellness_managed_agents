import sqlite3

DB_FILE = "wellness.db"

def get_db_connection():
    """Establishes and returns a connection to the local SQLite database file."""
    conn = sqlite3.connect(DB_FILE)
    # This setting allows fetching rows like a Python dictionary (by column name)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Creates the necessary local tables for tracking user states if they don't exist."""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            age INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            injuries TEXT,
            dietary_restrictions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS weekly_plans (
            plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
            week_number INTEGER NOT NULL,
            workout_plan TEXT NOT NULL,
            yoga_plan TEXT NOT NULL,
            diet_plan TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, week_number)
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
        print("✅ Local SQLite database file ('wellness.db') initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing SQLite database: {e}")
    finally:
        if conn is not None:
            conn.close()

def save_user_profile(user_id: str, age: int, weight: float, injuries: str, diet_restrictions: str):
    """Inserts or updates a user's physiological profile using SQLite syntax."""
    query = """
        INSERT INTO users (user_id, age, weight_kg, injuries, dietary_restrictions)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) 
        DO UPDATE SET 
            age = excluded.age,
            weight_kg = excluded.weight_kg,
            injuries = excluded.injuries,
            dietary_restrictions = excluded.dietary_restrictions;
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, (user_id, age, weight, injuries, diet_restrictions))
    conn.commit()
    cur.close()
    conn.close()

def get_user_profile_string(user_id: str) -> str:
    """Fetches user metrics and formats them for the agent context loop."""
    query = "SELECT age, weight_kg, injuries, dietary_restrictions FROM users WHERE user_id = ?;"
    conn = get_db_connection()
    cur = conn.cursor()
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
        f"- Food Allergies/Diet Restrictions: {row['dietary_restrictions'] if row['dietary_restrictions'] else 'None'}"
    )
    return profile_str

def save_weekly_plan(user_id: str, week_number: int, workout: str, yoga: str, diet: str):
    """Saves a pristine snapshot of the generated weekly routine, stripping out AI metadata."""
    
    def clean_agent_output(output) -> str:
        """Extracts pure markdown text, discarding thought signatures and block wrappers."""
        if isinstance(output, list):
            extracted_chunks = []
            for item in output:
                # If it's a dictionary containing a 'text' key (Gemini 2.5 format)
                if isinstance(item, dict) and "text" in item:
                    extracted_chunks.append(item["text"])
                elif isinstance(item, dict) and "content" in item:
                    extracted_chunks.append(item["content"])
                else:
                    extracted_chunks.append(str(item))
            return "\n".join(extracted_chunks)
        
        if isinstance(output, dict):
            return output.get("text", output.get("content", str(output)))
            
        return str(output)

    # Clean the agent outputs before sending them to the SQLite tables
    workout = clean_agent_output(workout)
    yoga = clean_agent_output(yoga)
    diet = clean_agent_output(diet)

    query = """
        INSERT INTO weekly_plans (user_id, week_number, workout_plan, yoga_plan, diet_plan)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, week_number)
        DO UPDATE SET 
            workout_plan = excluded.workout_plan,
            yoga_plan = excluded.yoga_plan,
            diet_plan = excluded.diet_plan;
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, (user_id, week_number, workout, yoga, diet))
    conn.commit()
    cur.close()
    conn.close()

def get_last_week_number(user_id: str) -> int:
    """Finds the maximum week number generated so far for a user."""
    query = "SELECT COALESCE(MAX(week_number), 0) FROM weekly_plans WHERE user_id = ?;"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, (user_id,))
    max_week = cur.fetchone()[0]
    cur.close()
    conn.close()
    return max_week