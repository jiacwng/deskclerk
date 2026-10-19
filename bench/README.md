bench/ measures deskclerk's agent on WorkBench (olly-styles/WorkBench), a set of workplace tasks with ground-truth outcomes, and compares it with WorkBench's own agent loop.

## Files

- `workbench.py`: the task files, the dev/test split, the date line WorkBench puts before every task, and the columns of a results file.
- `workbench_server.py`: serves the WorkBench tools over MCP and logs each executed call in the format WorkBench's grader replays.
- `run_reference.py`: runs WorkBench's own agent loop, unchanged, on one split.
- `run_deskclerk.py`: runs deskclerk's agent loop on one split against the WorkBench MCP server, with the write gate set to none, rules or check, and with the settings the product uses: its system instructions (followed by WorkBench's date line and act-without-confirmation sentence), its output limit, its effort setting and its step limit.
- `compare.py`: scores results files with WorkBench's own scorer and compares runs pairwise with an exact McNemar test.
- `jev_calls.py`: replays the calls logged in results files and lists every write call with its request, arguments, earlier results and whether the ground truth contains it.
- `jev_score.py`: scores those write calls with Jev and with Claude Haiku 5.5 and reports how well each score separates wanted from unwanted calls.
- `test_*.py`: tests for the files above. They need `WORKBENCH_PATH` and `SERVER_PYTHON` set (see below).
- `requirements.txt`: what the bench virtualenv installs.

## Split

Per domain, tasks are ordered by the sha256 of `<domain>-<index>`. The first third is dev (231 tasks) and the rest is test (459 tasks).

## Reproduce M5.10

Clone WorkBench at the commit the runs used:

```
git clone https://github.com/olly-styles/WorkBench.git
git -C WorkBench checkout 49c7dfd
```

Make the bench virtualenv. It holds WorkBench's dependencies, pytest and the MCP server. The runners need deskclerk in the same environment, so install the repository too:

```
python3 -m venv ~/.venvs/bench
~/.venvs/bench/bin/pip install -r bench/requirements.txt
~/.venvs/bench/bin/pip install -e .
```

Start nothing by hand. `run_deskclerk.py` starts one WorkBench server per worker and stops it at the end.

The model is Claude Haiku 5.5. The runners read `ANTHROPIC_API_KEY` from the environment, and `--gate check` also reads `TYPESAFE_API_KEY`.

Reference loop, one results file per split:

```
~/.venvs/bench/bin/python bench/run_reference.py --workbench WorkBench --split dev \
  --model claude-haiku-5.5 --model-id claude-haiku-5-5 \
  --price-input 0.10 --price-output 0.50 --out runs/reference-dev.csv
```

deskclerk, one run per gate:

```
for gate in none rules check; do
  ~/.venvs/bench/bin/python bench/run_deskclerk.py --workbench WorkBench --split dev \
    --gate $gate --server-python ~/.venvs/bench/bin/python \
    --out runs/deskclerk-$gate-dev.csv
done
```

Use `--split test` and the matching file names for the test split. The model defaults to `anthropic:claude-haiku-5-5`. The step limit is the default of `DESKCLERK_STEP_LIMIT`, and the task text is sent as the request with nothing added.

Score and compare, here for dev:

```
~/.venvs/bench/bin/python bench/compare.py --workbench WorkBench \
  --run reference=runs/reference-dev.csv \
  --run none=runs/deskclerk-none-dev.csv \
  --run rules=runs/deskclerk-rules-dev.csv \
  --run check=runs/deskclerk-check-dev.csv \
  --pair reference:none --pair none:rules --pair none:check \
  --out runs/compare-dev.md
```

## Measure Jev as a classifier of write calls

Stage 1 needs WorkBench's pandas, so it runs in the bench virtualenv. It makes no paid call. The file names must contain `_dev_` or `_test_`:

```
~/.venvs/bench/bin/python bench/jev_calls.py --workbench WorkBench   --results runs/dev-none.csv runs/dev-rules.csv runs/test-none.csv --out runs/jev-calls.jsonl
```

Stage 2 makes the paid calls. It runs where deskclerk's dependencies and scipy are installed, reads `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY`, and skips calls already in the `--scores` file, so a rerun costs nothing for them:

```
PYTHONPATH=src python bench/jev_score.py --calls runs/jev-calls.jsonl   --scores runs/jev-scores.jsonl --jev-budget-usd 2 --out runs/jev-report.md
```

The report gives, per judge and per split, the ROC AUC with a 95% bootstrap interval. It then applies one rule: the threshold is the highest unwanted score on dev, and a test call runs without confirmation when its score is strictly above it.

## Run the bench tests

```
WORKBENCH_PATH=$PWD/WorkBench SERVER_PYTHON=~/.venvs/bench/bin/python \
  ~/.venvs/bench/bin/python -m pytest bench -q
```
