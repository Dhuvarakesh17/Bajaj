#db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from base import Base  # your Base declarative model

load_dotenv()

# Directly load the full DATABASE_URL from .env (which includes user, password, host, port, db, sslmode)
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

from models import QueryLog  # safe to import after engine creation
Base.metadata.create_all(bind=engine)
