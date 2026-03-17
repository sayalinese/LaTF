from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime

db = SQLAlchemy()
migrate = Migrate()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(50), primary_key=True)
    nickname = db.Column(db.String(50), nullable=False)
    avatar = db.Column(db.String(255))

class DisputeSession(db.Model):
    __tablename__ = 'dispute_sessions'
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    buyer_id = db.Column(db.String(50), db.ForeignKey('users.id'))
    seller_id = db.Column(db.String(50), db.ForeignKey('users.id'))
    topic_name = db.Column(db.String(255), nullable=False, default='未命名交易纠纷')
    status = db.Column(db.String(20), default='open')  # open, ai_judging, closed
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = db.relationship('Message', backref='session', cascade='all, delete-orphan')

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = db.Column(UUID(as_uuid=True), db.ForeignKey('dispute_sessions.id', ondelete='CASCADE'), nullable=False)
    sender_id = db.Column(db.String(50), nullable=False)
    sender_role = db.Column(db.String(20), nullable=False)  # buyer, seller, ai, system
    content_type = db.Column(db.String(20), default='text', nullable=False)  # text, image
    content = db.Column(db.Text, nullable=False)
    ai_detect_data = db.Column(JSONB)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
