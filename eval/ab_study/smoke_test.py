import json
import os
import sqlite3

os.environ["AB_DB"] = "smoke.db"
if os.path.exists("smoke.db"):
    os.remove("smoke.db")

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod                        # noqa: E402

_ctx = TestClient(appmod.app)
client = _ctx.__enter__()  # trigger startup events
PAIRS = json.load(open("pairs.json"))
failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


print("1. session creation")
r = client.post("/api/session", json={"experience": "1-4digit", "token": "friend-tok-001"})
check("session 200", r.status_code == 200, r.text)
sess = r.json()
rid = sess["rater_id"]
check("all pairs served", {p["pair_id"] for p in sess["pairs"]} == set(PAIRS))
payload = json.dumps(sess)
check("no label leakage in payload",
      "human" not in payload and "a_is_human" not in payload and "gen" not in payload)
check("left_slot present per pair", all(p["left_slot"] in ("A", "B") for p in sess["pairs"]))

print("2. token single-use")
r = client.post("/api/session", json={"experience": "casual", "token": "friend-tok-001"})
check("token reuse rejected", r.status_code == 403, r.text)
r = client.post("/api/session", json={"experience": "casual", "token": "nope"})
check("bad token rejected", r.status_code == 403)
r = client.post("/api/session", json={"experience": "casual"})
check("open access off by default", r.status_code == 403)
r = client.post("/api/session", json={"experience": "wizard", "token": "reddit-tok-001"})
check("bad experience rejected", r.status_code == 422, r.text)

print("3. responses + revision")
for pid in PAIRS:
    r = client.post("/api/response", json={"rater_id": rid, "pair_id": pid,
                                           "chosen_slot": "A", "decision_ms": 4200})
    check(f"answer {pid}", r.status_code == 200 and r.json()["revised"] is False, r.text)
r = client.post("/api/response", json={"rater_id": rid, "pair_id": "pair01",
                                       "chosen_slot": "B", "decision_ms": 900})
check("revision flagged", r.status_code == 200 and r.json()["revised"] is True, r.text)
r = client.post("/api/response", json={"rater_id": rid, "pair_id": "pairXX", "chosen_slot": "A"})
check("unknown pair 404", r.status_code == 404)

print("4. submit gating + scoring")
r2 = client.post("/api/session", json={"experience": "5-6digit", "token": "reddit-tok-001"})
rid2 = r2.json()["rater_id"]
r = client.post("/api/submit", json={"rater_id": rid2})
check("submit blocked when incomplete", r.status_code == 409, r.text)

r = client.post("/api/submit", json={"rater_id": rid})
check("submit 200", r.status_code == 200, r.text)
result = r.json()
# expected: chose B on pair01 (human=A -> wrong), A on pair02 (human=B -> wrong),
#           A on pair03 (human=B -> wrong)  => 0/3
check("score computed server-side, matches key", result["score"] == 0 and result["total"] == 3,
      str(result))
check("reveal only after submit", all("human_slot" in x for x in result["reveal"]))
r = client.post("/api/response", json={"rater_id": rid, "pair_id": "pair01", "chosen_slot": "A"})
check("responses locked after submit", r.status_code == 409)
r = client.post("/api/submit", json={"rater_id": rid})
check("double submit rejected", r.status_code == 409)

print("5. db snapshot integrity")
conn = sqlite3.connect("smoke.db")
rows = conn.execute("SELECT pair_id, slot_a_is_human, revised FROM responses"
                    " WHERE rater_id = ?", (rid,)).fetchall()
key_ok = all(bool(s) == PAIRS[p]["a_is_human"] for p, s, _ in rows)
check("slot_a_is_human snapshot matches key", key_ok)
check("revised flag persisted", any(rv == 1 for _, _, rv in rows))

print()
if failures:
    print(f"FAILURES: {failures}")
    raise SystemExit(1)
_ctx.__exit__(None, None, None)
print("ALL PASS")