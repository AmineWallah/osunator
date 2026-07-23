# osunator A/B study harness

Blinded **two-alternative forced-choice (2AFC)** study: raters watch a pair of
replay clips — one human, one generated, same map and same section — and pick
the one they believe is human. Backend: FastAPI + SQLite. Frontend: static page.

Not a Turing test (no interrogation, no interaction): a discrimination task.

---

## Methodology

**Design.** 10 pairs per rater, one pair per held-out test map. ~30 s clips,
rendered with danser-go, no HUD (score/combo/accuracy counters would leak the
D-rank placeholder in generated replays). Both clips of a pair cover the same
time window of the same map, at the same resolution, framerate and encode
settings. Generation at temperature 0 (deterministic), same checkpoint as the
objective battery.

**Section selection.** Picked random 30 second segments from a map, which then
would be performed by both a human and the AI model.

**Slot assignment.** Which clip became slot A vs slot B was decided by a coin
flip per pair, executed mechanically by `make_pairs.py` (the flip, the filename
and the `a_is_human` flag are produced by the same line of code, so they cannot
disagree). Slot letters carry no meaning to raters: presentation order is
shuffled per rater, and which slot renders on the left is shuffled per pair per
rater. Clip filenames are content-free (`pairNN_A.mp4`).

**Raters.** Recruited via r/osugame. Self-reported experience bracket collected
before the task: `none` / `casual` / `5-6digit` / `1-4digit` / `lapsed`.
No accounts, no names. Cohort tags: `open` (public link), `friend` /
`reddit` (invite tokens), `test` (author).

**Collection window.** Opened 2026-07-22. Pre-declared close: FILL_ME (date or N,
whichever first). Interim analyses: FILL_ME (looks taken before close, so the
record is honest).

**Known limitations of the protocol.**
- Open mode (`AB_ALLOW_OPEN=1`) was used for public recruitment: open-cohort
  participation is deduplicated heuristically (per-browser localStorage flag,
  per-IP-hash session cap, post-hoc inspection of hashed IPs), not enforced.
  Invite-token cohorts enforce single participation server-side.
- 10 pairs per rater bounds precision; the pooled 95% CI is roughly ±5 points at
  N≈40 raters.
- Experience brackets are self-reported and unverified.

---

## Results

### TO BE FILLED

---

## Privacy

Stored per rater: a random UUID, cohort, self-reported experience bracket,
timestamps, truncated user-agent, and a **salted SHA-256 hash of the IP
address** — the raw IP is never written. The salt lives only in the deployment
environment (`AB_IP_SALT`); without it the hashes are not reversible. No names,
emails or accounts.

Questions, or want your responses removed? Send me a Discord message: @aminewallah

The public data release (after collection closes) drops the `ip_hash`,
`user_agent` and `token` columns and reports small cohorts in aggregate only.

---

## Security invariant

The client NEVER receives ground-truth labels. Truth lives only in `pairs.json`
(server-side) and in the `slot_a_is_human` snapshot on each response row.
Scoring happens server-side; the reveal is served only after submission is
locked. `smoke_test.py` greps the session payload for label-shaped strings —
that check must pass before any deploy.

---

## Run locally

    uv sync                                 # deps incl. the `eval` group
    uv run python smoke_test.py             # must print ALL PASS
    AB_ALLOW_OPEN=1 AB_DB=scratch.db uv run uvicorn app:app --reload

`smoke_test.py` currently expects a real `pairs.json` / `tokens.json`; example
stubs are TODO (see August polish list below).

## Deployment (Railway)

- Service root directory: `eval/ab_study`; auto-deploys on push to master.
- Start command:
  `mkdir -p /data/clips && ln -sfn /data/clips static/clips && uvicorn app:app --host 0.0.0.0 --port $PORT`
- Volume mounted at `/data` holds `study.db`, `pairs.json`, `tokens.json`,
  `clips/` — these are deliberately absent from git and survive every deploy.
- Environment: `AB_DB`, `AB_PAIRS`, `AB_TOKENS` → `/data/...`;
  `AB_IP_SALT` (random hex); `AB_ALLOW_OPEN` (1 = public link, 0 = invite-only);
  `AB_SESSION_CAP` (default 10 tokenless sessions per IP-hash per day; invite
  tokens are exempt).
- `StaticFiles(..., follow_symlink=True)` is required: Starlette refuses to
  follow symlinks out of the static root by default, which 404s every clip.
- The canonical database is the deployed `/data/study.db`. The local
  `study.db` contains only the excluded pre-deployment session above.

Pull the database for analysis:

    railway ssh -- python3 -c "import base64,sys; sys.stdout.write(base64.b64encode(open('/data/study.db','rb').read()).decode())" | base64 -d > pulled_study.db
    uv run python analysis.py pulled_study.db pairs.json

## Analysis

    python analysis.py study.db pairs.json

Stdlib only: exact two-sided binomial test vs 50%, Wilson confidence intervals,
breakdowns by cohort / experience / map, revision effect, per-rater score
distribution. Non-submitted raters are excluded and the count printed.

## pairs.json format

    {"pair01": {"a": "/static/clips/pair01_A.mp4",
                "b": "/static/clips/pair01_B.mp4",
                "a_is_human": true,
                "map": "Artist - Title [Diff]"}, ...}

`a_is_human` is set at render time by coin flip; keep the balance near 5/5
across the 10 pairs. Published after collection closes, alongside the
anonymized responses, so the results can be checked.

## TODO (post-collection)

- [ ] `pairs.example.json` + `tokens.example.json` so `smoke_test.py` runs on a
      fresh clone without the real key
- [ ] restore `make_pairs.py` (documents mechanical slot assignment)
- [ ] export scrubber for the anonymized data release
- [ ] tear down the Railway service once the final database is pulled and backed up