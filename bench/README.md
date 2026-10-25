bench/ measures deskclerk's agent on WorkBench (olly-styles/WorkBench), a set of workplace tasks with ground-truth outcomes, against WorkBench's own agent loop, and measures prompt injection screens on AgentDojo, LLMail-Inject and InjecAgent.

## Results

Claude Haiku 5.5 is the agent in every run. The sections below give the commands to rerun each row.

| What | Result | Limit |
|---|---|---|
| WorkBench, 459 held-out workplace tasks over five MCP servers, graded by WorkBench's own scorer | deskclerk: 441 correct, 7 harmful. WorkBench's own agent loop: 447 correct, 7 harmful. | No measurable difference (exact McNemar p = 0.18 correct, p = 1.0 harmful). One test run per loop, with deskclerk's write check off. The reference run is from an earlier date with the same model. It is near the ceiling at 447 of 459. |
| The same tasks and servers, with Jev picking the servers a task needs | 442 correct, 9 harmful, at 57% of the cost per task of offering every tool, Jev included. | No measurable difference from offering every tool (p = 1.00 correct, p = 0.69 harmful). The threshold was fitted on the 231 dev tasks. Single-turn tasks only. |
| AgentDojo v1.2.2, 949 attacked runs over four suites | Attack success with no defense: 2.0% (19 of 949), with 84.5% of tasks completed with no attack and 81.9% under attack. With spotlighting: 1.9%. With a DeBERTa screen: 0.0%, with 45.4% and 38.7% of tasks completed. With Prompt Guard 2 as the screen: 0.4% (4 of 949), with 84.5% and 67.7%. With Jev withholding the results it flags: 0.0%, with 84.5% and 20.7%. | One agent model, one attack, one run per task. The baseline has 19 successes. Jev's question was rewritten once after a first run that completed 82.5% and 21.0%. |
| The same 949 attacked runs, with a result Jev flags passed to the agent behind a warning, as deskclerk does | If the employee approves every call: 1.9% (18 of 949), with 84.5% of tasks completed with no attack and 81.7% under attack. If the employee rejects every call after a flag: 0.0%, with 84.5% and 0.1%. Jev flagged none of the 97 runs with no attack. | AgentDojo has no human, so these are the two bounds. An employee who rejects only the injected call lands between them. One run per task. |
| Three injection screens on 1,786 distinct tool outputs that the undefended agent read in AgentDojo | Outputs with the attack's marker flagged (1,174): Jev 100.0%, Prompt Guard 2 31.7%, DeBERTa 64.3%. Outputs without it flagged (612): Jev 0.0%, Prompt Guard 2 0.0%, DeBERTa 30.6%. | One attack with a fixed text, labelled by its marker. Outputs repeat across runs. |
| LLMail-Inject: 5,000 of the 37,303 prompt injections people wrote in Microsoft's phase 2 challenge, against 160 of the challenge's clean emails | Injections flagged: Jev 75.2%, Prompt Guard 2 16.1%, DeBERTa 34.1%. Clean emails flagged: none by any screen. ROC AUC: 0.995, 0.986, 0.986. With the 612 clean AgentDojo outputs added, DeBERTa flags 187 of them and the AUC is 0.990, 0.805, 0.631. | Every submission counts as an injection, including broken ones. One request for every email. Submissions from one team are alike, so the intervals in the report are too narrow. None of 160 clean emails flagged still allows a rate up to about 2.3%. |
| InjecAgent, 2,108 cases in its base and enhanced settings | With no defense, 0 attacks succeed and 171 outputs are invalid, so the screen has nothing to stop. Jev flagged 88.2% of the injected tool results, and invalid outputs rose to 404, mostly the agent calling the user tool again. | One model, one run. None of 2,108 still allows a rate up to about 0.14%. The cases are templated and single-step. The runner calls the model through Anthropic's Messages API. |
| TypeSafe's Jev scoring 1,742 write calls from the WorkBench runs | ROC AUC 0.47 on test and 0.55 on dev | An AUC of 0.5 is chance. Haiku 5.5 as the judge scored 0.63 on test. |

## Files

- `workbench.py`: the task files, the dev/test split, the date line WorkBench puts before every task, and the columns of a results file.
- `workbench_server.py`: serves the WorkBench tools over MCP and logs each executed call in the format WorkBench's grader replays.
- `run_reference.py`: runs WorkBench's own agent loop, unchanged, on one split.
- `run_deskclerk.py`: runs deskclerk's agent loop on one split against the WorkBench MCP server, with the write gate set to none or rules, and with the settings the product uses: its system instructions (followed by WorkBench's date line and act-without-confirmation sentence), its output limit, its effort setting and its step limit. `--selection all|server|llm` offers every tool, the servers Jev picks, or the tools LangChain's LLMToolSelectorMiddleware picks. The WorkBench tools are served as five MCP servers, one per domain, in every arm.
- `compare.py`: scores results files with WorkBench's own scorer and compares runs pairwise with an exact McNemar test.
- `jev_calls.py`: replays the calls logged in results files and lists every write call with its request, arguments, earlier results and whether the ground truth contains it.
- `jev_score.py`: scores those write calls with Jev and with Claude Haiku 5.5 and reports how well each score separates wanted from unwanted calls.
- `run_agentdojo.py`: runs the agent through AgentDojo v1.2.2 with one defense (none, spotlighting, deberta, promptguard, jev or jev-flag) and writes a JSON summary. The jev defense withholds an output Jev flags, and jev-flag passes it behind a warning and reports the two bounds of an employee who approves or rejects every later call. Both ask Jev through `src/deskclerk/screen.py`, the file the product screens tool outputs with.
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
for defense in none spotlighting deberta promptguard jev jev-flag; do
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
