from app import app, db
from sqlalchemy import text
with app.app_context():
    db.session.execute(text("ALTER TABLE users ADD COLUMN password VARCHAR(255) NOT NULL DEFAULT '123' "))
    db.session.commit()
