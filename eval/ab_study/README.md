# osunator A/B study harness

Blinded forced-choice study: raters watch human/generated replay pairs
and pick the human. Backend: FastAPI + SQLite. Frontend: static page (TODO).

## Security invariant
The client NEVER receives ground-truth labels. Truth lives only in
pairs.json (server-side) and the responses table. Verify with the
smoke test's "no label leakage" check before any deploy.

## Run locally
    pip install fastapi uvicorn            # or: uv add --group eval fastapi uvicorn
    python smoke_test.py                   # must print ALL PASS
    AB_ALLOW_OPEN=1 uvicorn app:app --reload

## Deploy checklist
- [ ] real pairs.json (20 pairs) + clips in static/clips/
- [ ] tokens.json with one token per invited rater (cohort-tagged)
- [ ] AB_IP_SALT set to a random string
- [ ] AB_ALLOW_OPEN=0 unless running the open Reddit cohort
- [ ] persistent volume for study.db
- [ ] smoke_test.py passes against deployed pairs.json
- [ ] devtools check on deployed site: no labels in network responses

## Analysis
    python analysis.py study.db pairs.json
Stdlib-only: exact binomial test vs 50%, Wilson CIs, breakdowns by
cohort / experience / map, revision effect, per-rater distribution.

## pairs.json format
    {"pair01": {"a": "/static/clips/pair01_A.mp4",
                "b": "/static/clips/pair01_B.mp4",
                "a_is_human": true,
                "map": "Artist - Title [Diff]",
                "section": "00:45-01:05"}, ...}
a_is_human is assigned by coin flip AT RENDER TIME (keep ~10/10 balance
across 20 pairs). Slot letters carry no meaning to raters.