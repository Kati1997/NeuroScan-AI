"""Reset admin password to default."""
import database as db
from werkzeug.security import generate_password_hash

db.init_db()
email = "admin@hospital.com"
password = "Admin@123"

with db.get_db() as conn:
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE users SET password_hash = ?, role = 'admin', name = 'System Admin' WHERE email = ?",
            (generate_password_hash(password), email),
        )
        print(f"Password reset for {email}")
    else:
        conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("System Admin", email, generate_password_hash(password), "admin"),
        )
        print(f"Created admin user {email}")

print("Login with:")
print(f"  Email:    {email}")
print(f"  Password: {password}")
