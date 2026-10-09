import sqlite3

DB_PATH = "saferide.db"

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

# Add Google identity fields if they don't already exist
try:
    cursor.execute(
        "ALTER TABLE drivers ADD COLUMN google_sub TEXT"
    )
    print("Added google_sub column.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e).lower():
        print("google_sub already exists.")
    else:
        raise

try:
    cursor.execute(
        "ALTER TABLE drivers ADD COLUMN email TEXT"
    )
    print("Added email column.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e).lower():
        print("email already exists.")
    else:
        raise

# Unique index for Google account identity
cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS
    ix_drivers_google_sub
    ON drivers(google_sub)
""")

connection.commit()
connection.close()

print("Google authentication database migration completed successfully.")