"""Local structured events. Not a tamper-proof forensic record."""
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import json
import uuid
import csv

class AuditStore:
    def __init__(self,directory):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.session=uuid.uuid4().hex
        self.db=sqlite3.connect(str(self.directory/"events.sqlite3"))
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, timestamp TEXT, session TEXT, attempt TEXT,
            track TEXT, candidate TEXT, event TEXT, details TEXT)""")
        self.db.commit()

    def add(self,event,attempt=None,track=None,candidate=None,**details):
        self.db.execute("INSERT INTO events(timestamp,session,attempt,track,candidate,event,details) VALUES(?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(),self.session,attempt,track,candidate,event,json.dumps(details,ensure_ascii=False)))
        self.db.commit()

    def recent(self,limit=200):
        return self.db.execute("SELECT timestamp,event,attempt,track,candidate,details FROM events ORDER BY id DESC LIMIT ?",(limit,)).fetchall()

    def export(self,path):
        with open(path,"w",encoding="utf-8-sig",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["timestamp_utc","event","attempt","track","candidate_profile_not_verified","details"])
            for row in self.db.execute("SELECT timestamp,event,attempt,track,candidate,details FROM events ORDER BY id"):
                # Prevent spreadsheet formula interpretation in exported fields.
                writer.writerow([("'"+str(v)) if str(v).startswith(("=","+","-","@","\t","\r")) else v for v in row])

    def close(self):
        self.db.close()
