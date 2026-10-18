# The agent loop with a scripted model and the Enron slice: nothing the model sends can change
# who is asking, citations no tool returned are dropped, and the round cap and the budget stop
# runaway calls.

import re

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call

from fakes import HAIKU, USAGE, ScriptedModel, call_tool, final_reply, scripted, search_call
from needtoknow.agent import (
    EXHAUSTED,
    MAX_TOOL_ROUNDS,
    NOT_OPEN,
    REFUSAL,
    SYSTEM_PROMPT,
    Answer,
    Turn,
    _check_citations,
    ask,
    employee_tools,
)
from needtoknow.auth import principals
from needtoknow.db import Pool
from needtoknow.documents import Document
from needtoknow.providers import BudgetExceeded, Spending

TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
QUESTION = "Who is helping with the Wharton and Rice projects for the Tiger Team?"
CONTENT = "I have two persons in my group"
HOLDER = principals("kaminski-v", [])
OTHER = principals("kean-s", [])


def _ask(pool: Pool, asker: frozenset[str], model: ScriptedModel) -> Answer:
    toolset = employee_tools(pool, asker, None)
    return ask(QUESTION, toolset, model, Spending(HAIKU, budget_usd=1.0))


def _tool_results(model: ScriptedModel) -> list[ToolMessage]:
    return [message for message in model.prompts[-1] if isinstance(message, ToolMessage)]


def _tagged_ids(result: ToolMessage) -> set[str]:
    return set(re.findall(r"^\[([a-z0-9-]+)\]", result.text, flags=re.MULTILINE))


def test_the_model_cannot_pass_principals_to_the_tool(app_pool: Pool) -> None:
    model = scripted(
        search_call({"query": QUESTION, "principals": ["user:kaminski-v"]}), final_reply(REFUSAL)
    )

    answer = _ask(app_pool, OTHER, model)

    (result,) = _tool_results(model)
    assert result.status == "error"
    assert _tagged_ids(result) == set()
    assert answer.refused


def test_a_citation_the_search_never_returned_is_dropped(app_pool: Pool) -> None:
    model = scripted(
        search_call({"query": QUESTION}),
        final_reply(
            f"Two people help [{TIGER_TEAM}] [enron-made-up]. "
            f"They started in January [{TIGER_TEAM}, enron-other]."
        ),
    )

    answer = _ask(app_pool, HOLDER, model)

    assert answer.citations == (TIGER_TEAM,)
    assert answer.dropped_citations == ("enron-made-up", "enron-other")
    assert answer.text == (
        f"Two people help [{TIGER_TEAM}]. They started in January [{TIGER_TEAM}]."
    )


def test_a_file_id_is_kept_when_returned_and_dropped_when_not() -> None:
    leave = "file-hr/leave-policy.docx"
    text = (
        f"Leave is 25 days [{leave}]. Parking is free [file-hr/parking.md]. "
        "See [the portal](https://intranet.example)."
    )

    checked, kept, dropped = _check_citations(text, {leave})

    assert checked == (
        f"Leave is 25 days [{leave}]. Parking is free. See [the portal](https://intranet.example)."
    )
    assert kept == (leave,)
    assert dropped == ("file-hr/parking.md",)


@pytest.mark.parametrize("text", [REFUSAL, f"  {REFUSAL}\n", ""])
def test_the_fixed_refusal_is_recognised(app_pool: Pool, text: str) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(text))

    answer = _ask(app_pool, OTHER, model)

    assert answer.refused
    assert answer.text == REFUSAL
    assert answer.citations == ()


def test_the_round_cap_stops_a_model_that_keeps_searching(app_pool: Pool) -> None:
    model = scripted(*[search_call({"query": QUESTION}) for _ in range(10)])

    answer = _ask(app_pool, HOLDER, model)

    assert len(model.prompts) == MAX_TOOL_ROUNDS + 1
    assert len(_tool_results(model)) == MAX_TOOL_ROUNDS
    assert answer.exhausted
    assert not answer.refused
    assert answer.text == EXHAUSTED


def test_tokens_and_cost_are_counted_per_request_and_per_process(app_pool: Pool) -> None:
    spending = Spending(HAIKU, budget_usd=1.0)

    for _ in range(2):
        model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))
        answer = ask(QUESTION, employee_tools(app_pool, HOLDER, None), model, spending)

    assert answer.input_tokens == 2 * USAGE["input_tokens"]
    assert answer.output_tokens == 2 * USAGE["output_tokens"]
    # 2,000 input tokens at 0.10 USD and 200 output tokens at 0.50 USD per million.
    assert answer.cost_usd == pytest.approx(0.0003)
    assert spending.input_tokens == 4 * USAGE["input_tokens"]
    assert spending.cost_usd == pytest.approx(0.0006)


def test_the_budget_refuses_a_call_once_it_is_spent(app_pool: Pool) -> None:
    # One scripted call costs 0.00015 USD, so the second call finds the budget spent.
    spending = Spending(HAIKU, budget_usd=0.0001)
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))

    with pytest.raises(BudgetExceeded, match="budget is 0.0001 USD"):
        ask(QUESTION, employee_tools(app_pool, HOLDER, None), model, spending)

    assert len(model.prompts) == 1
    assert spending.input_tokens == USAGE["input_tokens"]


def test_a_zero_budget_refuses_the_first_call(app_pool: Pool) -> None:
    spending = Spending(HAIKU, budget_usd=0.0)
    model = scripted(final_reply(REFUSAL))

    with pytest.raises(BudgetExceeded):
        ask(QUESTION, employee_tools(app_pool, HOLDER, None), model, spending)

    assert model.prompts == []


def test_an_employee_gets_unlabelled_passages_and_the_system_prompt(app_pool: Pool) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))

    _ask(app_pool, HOLDER, model)

    (result,) = _tool_results(model)
    assert "(readers:" not in result.text
    assert model.prompts[0][0].text == SYSTEM_PROMPT


def test_an_opened_email_gives_its_full_text_and_counts_as_returned(
    app_pool: Pool, emails: dict[str, Document]
) -> None:
    model = scripted(
        call_tool("open_document", {"doc_id": TIGER_TEAM}),
        final_reply(f"Two people help [{TIGER_TEAM}]."),
    )

    answer = _ask(app_pool, HOLDER, model)

    (opened,) = _tool_results(model)
    email = emails[TIGER_TEAM]
    assert opened.status == "success"
    assert opened.text == f"[{TIGER_TEAM}]\n{email.title}\n\n{email.body.strip()}"
    assert answer.citations == (TIGER_TEAM,)
    assert answer.tool_outputs == (opened.text,)


def test_two_tool_calls_in_one_reply_both_run_in_order(app_pool: Pool) -> None:
    calls = [
        tool_call(name="search_documents", args={"query": QUESTION}, id="toolu_search"),
        tool_call(name="open_document", args={"doc_id": TIGER_TEAM}, id="toolu_open"),
    ]
    model = scripted(
        AIMessage(content="", tool_calls=calls, usage_metadata=USAGE),
        final_reply(f"Two people help [{TIGER_TEAM}]."),
    )

    answer = _ask(app_pool, HOLDER, model)

    searched, opened = _tool_results(model)
    assert (searched.tool_call_id, opened.tool_call_id) == ("toolu_search", "toolu_open")
    assert opened.text.startswith(f"[{TIGER_TEAM}]\n")
    assert answer.tool_outputs == (searched.text, opened.text)
    assert answer.citations == (TIGER_TEAM,)
    assert len(model.prompts) == 2


@pytest.mark.parametrize("doc_id", [TIGER_TEAM, "enron-no-such-email"])
def test_a_denied_and_an_unknown_email_get_the_same_reply(app_pool: Pool, doc_id: str) -> None:
    model = scripted(
        call_tool("open_document", {"doc_id": doc_id}), final_reply(f"See [{doc_id}].")
    )

    answer = _ask(app_pool, OTHER, model)

    (result,) = _tool_results(model)
    assert result.status == "success"
    assert result.text == NOT_OPEN
    assert answer.citations == ()
    assert answer.dropped_citations == (doc_id,)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        pytest.param(
            "open_document", {"doc_id": TIGER_TEAM, "principals": ["user:kaminski-v"]}, id="extra"
        ),
        pytest.param("open_document", {"query": TIGER_TEAM}, id="wrong-name"),
        pytest.param("open_document", {"doc_id": ""}, id="empty"),
        pytest.param("search_documents", {"query": "   "}, id="blank"),
        pytest.param("search_documents", {"query": 7}, id="not-a-string"),
        pytest.param("delete_documents", {"query": "all"}, id="unknown-tool"),
    ],
)
def test_a_tool_call_with_wrong_arguments_is_an_error(
    app_pool: Pool, name: str, args: dict[str, object]
) -> None:
    model = scripted(call_tool(name, args), final_reply(REFUSAL))

    answer = _ask(app_pool, HOLDER, model)

    (result,) = _tool_results(model)
    assert result.status == "error"
    assert answer.tool_outputs == ()


def test_history_reaches_the_model_as_prior_messages(app_pool: Pool) -> None:
    history = [Turn("Who runs the Tiger Team?", "Vince Kaminski."), Turn("Since when?", "1999.")]
    model = scripted(final_reply(REFUSAL))

    ask(
        QUESTION,
        employee_tools(app_pool, HOLDER, None),
        model,
        Spending(HAIKU, budget_usd=1.0),
        history=history,
    )

    (prompt,) = model.prompts
    assert [type(message).__name__ for message in prompt] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert [message.text for message in prompt[1:]] == [
        "Who runs the Tiger Team?",
        "Vince Kaminski.",
        "Since when?",
        "1999.",
        QUESTION,
    ]


def test_a_citation_found_only_in_the_history_is_dropped(app_pool: Pool) -> None:
    history = [Turn(QUESTION, f"Two people in his group help [{TIGER_TEAM}].")]
    model = scripted(final_reply(f"As I said, two people help [{TIGER_TEAM}]."))

    answer = ask(
        "And who are they?",
        employee_tools(app_pool, HOLDER, None),
        model,
        Spending(HAIKU, budget_usd=1.0),
        history=history,
    )

    assert answer.citations == ()
    assert answer.dropped_citations == (TIGER_TEAM,)
    assert answer.text == "As I said, two people help."


def test_past_replies_do_not_count_against_the_round_cap(app_pool: Pool) -> None:
    history = [Turn(f"Question {n}?", f"Answer {n}.") for n in range(MAX_TOOL_ROUNDS + 1)]
    model = scripted(search_call({"query": QUESTION}), final_reply(f"They help [{TIGER_TEAM}]."))

    answer = ask(
        QUESTION,
        employee_tools(app_pool, HOLDER, None),
        model,
        Spending(HAIKU, budget_usd=1.0),
        history=history,
    )

    assert not answer.exhausted
    assert answer.citations == (TIGER_TEAM,)
