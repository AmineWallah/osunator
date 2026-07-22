#!/usr/bin/env fish
cd (dirname (status filename))
set -x AB_DB scratch.db          # never the real name locally
set -x AB_ALLOW_OPEN 1           # tokenless for local clicking-around
uv run uvicorn app:app --reload