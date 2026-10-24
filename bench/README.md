bench/ measures deskclerk's agent on WorkBench (olly-styles/WorkBench), a set of workplace tasks with ground-truth outcomes, against WorkBench's own agent loop, and measures prompt injection screens on AgentDojo, LLMail-Inject and InjecAgent.

## Files

- `workbench.py`: the task files, the dev/test split, the date line WorkBench puts before every task, and the columns of a results file.
- `workbench_server.py`: serves the WorkBench tools over MCP and logs each executed call in the format WorkBench's grader replays.
- `run_reference.py`: runs WorkBench's own agent loop, unchanged, on one split.
- `run_deskclerk.py`: runs deskclerk's agent loop on one split against the WorkBench MCP server, with the write gate set to none or rules, and with the settings the product uses: its system instructions (followed by WorkBench's date line and act-without-confirmation sentence), its output limit, its effort setting and its step limit. `--selection all|server|llm` offers every tool, the servers Jev picks, or the tools LangChain's LLMToolSelectorMiddleware picks. The WorkBench tools are served as five MCP servers, one per domain, in every arm.
- `compare.py`: scores results files with WorkBench's own scorer and compares runs pairwise with an exact McNemar test.
- `jev_calls.py`: replays the calls logged in results files and lists every write call with its request, arguments, earlier results and whether the ground truth contains it.
- `jev_score.py`: scores those write calls with Jev and with Claude Haiku 5.5 and reports how well each score separates wanted from unwanted calls.
- `run_agentdojo.py`: runs the agent through AgentDojo v1.2.2 with one defense (none, spotlighting, deberta, promptguard or jev) and writes a JSON summary. The jev defense asks Jev through `src/deskclerk/screen.py`, the file the product screens tool outputs with.
- `compare_agentdojo.py`: prints one table of the summaries and an exact McNemar test of each defense against no defense.
- `dojo_summary.py`: the summary arithmetic and the model cost used by the two files above.
- `llmail_inject.py`: scores the phase 2 attacks of Microsoft's LLMail-Inject challenge and clean texts with Jev, Prompt Guard 2 and DeBERTa.
- `run_injecagent.py`: runs InjecAgent's own prompted agent with no defense and with Jev screening the injected tool response, and compares the two case by case.
- `agentdojo-requirements.txt`: what the AgentDojo virtualenv installs.
- `test_*.py`: tests for the files above. They need `WORKBENCH_PATH` and `SERVER_PYTHON` set (see below).
- `requirements.txt`: what the bench virtualenv installs.

## Split

Per domain, tasks are ordered by the sha256 of `<domain>-<index>`. The first third is dev (231 tasks) and the rest is test (459 tasks).

## Reproduce the WorkBench runs

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

The model is Claude Haiku 5.5. The runners read `ANTHROPIC_API_KEY` from the environment, and `--defense jev` of run_agentdojo.py also reads `TYPESAFE_API_KEY`.

Reference loop, one results file per split:

```
~/.venvs/bench/bin/python bench/run_reference.py --workbench WorkBench --split dev \
  --model claude-haiku-5.5 --model-id claude-haiku-5-5 \
  --price-input 0.10 --price-output 0.50 --out runs/haiku55_dev_reference.csv
```

deskclerk, one run per gate. The README row is the none gate on the test split with the standing instructions of commit f25341c. The first runs also had a check arm, which asked Jev about each write call; 7dafdbf is the last commit with that gate:

```
for gate in none rules; do
  ~/.venvs/bench/bin/python bench/run_deskclerk.py --workbench WorkBench --split dev \
    --gate $gate --server-python ~/.venvs/bench/bin/python \
    --out runs/haiku55_dev_$gate.csv
done
```

Use `--split test` and the matching file names for the test split. The model defaults to `anthropic:claude-haiku-5-5`. The step limit is the default of `DESKCLERK_STEP_LIMIT`, and the task text is sent as the request with nothing added.

Score and compare, here for dev:

```
~/.venvs/bench/bin/python bench/compare.py --workbench WorkBench \
  --run reference=runs/haiku55_dev_reference.csv \
  --run none=runs/haiku55_dev_none.csv \
  --run rules=runs/haiku55_dev_rules.csv \
  --pair reference:none --pair none:rules \
  --out runs/compare-dev.md
```

## Measure Jev as a classifier of write calls

Stage 1 needs WorkBench's pandas, so it runs in the bench virtualenv. It makes no paid call. The file names must contain `_dev_` or `_test_`:

```
~/.venvs/bench/bin/python bench/jev_calls.py --workbench WorkBench --results runs/haiku55_dev_none.csv runs/haiku55_dev_rules.csv runs/haiku55_test_none.csv --out runs/jev-calls.jsonl
```

Stage 2 makes the paid calls. It runs where deskclerk's dependencies and scipy are installed, reads `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY`, and skips calls already in the `--scores` file, so a rerun costs nothing for them:

```
PYTHONPATH=src python bench/jev_score.py --calls runs/jev-calls.jsonl --scores runs/jev-scores.jsonl --jev-budget-usd 2 --out runs/jev-report.md
```

The report gives, per judge and per split, the ROC AUC with a 95% bootstrap interval. It then applies one rule: the threshold is the highest unwanted score on dev, and a test call runs without confirmation when its score is strictly above it.

## Measure prompt injection defenses on AgentDojo

AgentDojo needs its own virtualenv, because it pins versions the other bench scripts do not:

```
python3 -m venv ~/.venvs/dojo
~/.venvs/dojo/bin/pip install -r bench/agentdojo-requirements.txt
~/.venvs/dojo/bin/pip install -e .
```

Run one defense on all four suites. Each run reads `ANTHROPIC_API_KEY`, and `--defense jev` also reads `TYPESAFE_API_KEY`. The deberta and promptguard defenses download their models from Hugging Face on first use, and promptguard needs a Hugging Face login that has accepted Meta's licence for Llama Prompt Guard 2. Run the defenses one at a time, because a second run in parallel reaches the model's input token rate limit:

```
for defense in none spotlighting deberta promptguard jev; do
  ~/.venvs/dojo/bin/python bench/run_agentdojo.py --suite all --defense $defense \
    --logdir runs/dojo --out runs/dojo-$defense.json
done
```

Compare the summaries. The first file is the baseline:

```
~/.venvs/dojo/bin/python bench/compare_agentdojo.py runs/dojo-none.json runs/dojo-spotlighting.json \
  runs/dojo-deberta.json runs/dojo-promptguard.json runs/dojo-jev.json > runs/dojo-compare.md
```

A summary reports, per suite, the share of user tasks completed without an attack, the share completed under attack and the share of attacks that succeeded. Jev is shown each tool output together with the user request and flags it at a probability of 0.5 or more.

## Measure the screens on saved AgentDojo tool outputs

This reads the logs of a finished run, scores each distinct pair of user request and tool output with Jev, Prompt Guard 2 and DeBERTa, and writes `scores.jsonl` and `report.md` to the output directory. A rerun skips scored outputs. It runs in the AgentDojo virtualenv, reads `TYPESAFE_API_KEY`, and uses the local Hugging Face cache offline:

```
HF_HUB_OFFLINE=1 PYTHONPATH=src:bench ~/.venvs/dojo/bin/python bench/screen_outputs.py \
  runs/dojo/claude-haiku-5-5-none runs/screen-outputs --jev-budget-usd 1
```

The report gives, per screen at threshold 0.5, the flagged share of outputs that carry the `<INFORMATION>` marker and of outputs that do not, with a 95% Wilson interval on each share and the ROC AUC, then two rules that combine Prompt Guard 2 and Jev: both flag, or either flags.

## Measure the screens on LLMail-Inject

This needs the AgentDojo run with no defense from the section above, and the Prompt Guard 2 and DeBERTa models in the local Hugging Face cache. Download `labelled_unique_submissions_phase2.json`, `emails_for_fp_tests.json` and `scenarios.json` from the `data` folder of the Hugging Face dataset microsoft/llmail-inject-challenge into `llmail-inject/data/`. Then, in the AgentDojo virtualenv:

```
HF_HUB_OFFLINE=1 PYTHONPATH=src:bench ~/.venvs/dojo/bin/python bench/llmail_inject.py \
  llmail-inject runs/dojo/claude-haiku-5-5-none runs/llmail --jev-budget-usd 3
```

It takes the 5,000 attacks with the lowest sha256 of their email text, uses the challenge's clean emails and the clean AgentDojo outputs as negatives, and writes `scores.jsonl` and `report.md`. A rerun skips scored texts.

## Run InjecAgent with and without the screen

Clone InjecAgent at the commit the run used. Its own requirements are not needed. Run it in the bench virtualenv:

```
git clone https://github.com/uiuc-kang-lab/InjecAgent.git
git -C InjecAgent checkout f19c9f2
PYTHONPATH=src ~/.venvs/bench/bin/python bench/run_injecagent.py --vendor InjecAgent --out runs/injecagent
```

It reads `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY`. `--per-setting 20` runs a pilot of 20 cases per setting.

## Run the bench tests

```
WORKBENCH_PATH=$PWD/WorkBench SERVER_PYTHON=~/.venvs/bench/bin/python \
  ~/.venvs/bench/bin/python -m pytest bench -q
```
