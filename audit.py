import json
import datetime
import os

LOG_FILE = "/tmp/audit_log.jsonl"   # ✅ FIXED PATH

def log_run(model_name, drift, fairness, stability, jurisdiction):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "drift": round(float(drift), 4),
        "fairness": round(float(fairness), 4),
        "stability": round(float(stability), 4),
        "jurisdiction": jurisdiction
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_logs():
    if not os.path.exists(LOG_FILE):
        return []

    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except:
                continue
    return logs
