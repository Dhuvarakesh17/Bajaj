# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from base import Base  # ✅ use Base from base.py

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    answer = Column(Text)
    document_url = Column(String)
    token_used = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
