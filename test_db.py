# test_db.py
from db import SessionLocal
from models import QueryLog

db = SessionLocal()
try:
    log = QueryLog(
        question="Does it work?",
        answer="Yes it works!",
        document_url="http://localhost/temp.pdf",
        token_used="456"
    )
    db.add(log)
    db.commit()
    print("✅ DB insert success!")
except Exception as e:
    print("❌ Insert failed:")
    print(e)
finally:
    db.close()
