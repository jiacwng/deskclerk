# The agent with the real model, run by hand with -m llm because every run costs money: the
# custodian who holds an Enron email gets an answer that cites it, and one who does not gets the
# fixed refusal.

from collections.abc import Iterator

import pytest

from needtoknow.agent import REFUSAL, ask
from needtoknow.auth import principals
from needtoknow.config import load_settings
from needtoknow.db import Pool
from needtoknow.providers import Spending, chat_model

pytestmark = pytest.mark.llm

TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
QUESTION = "Who is helping with the Wharton and Rice projects for the Tiger Team?"


@pytest.fixture(scope="module")
def spending() -> Iterator[Spending]:
    settings = load_settings()
    spending = Spending(settings.model, settings.budget_usd)
    yield spending
    print(
        f"\n{settings.model}: {spending.input_tokens} input and {spending.output_tokens} output "
        f"tokens, {spending.cost_usd:.6f} USD"
    )


def test_the_holder_gets_an_answer_that_cites_the_email(app_pool: Pool, spending: Spending) -> None:
    holder = principals("kaminski-v", [])

    answer = ask(QUESTION, app_pool, holder, chat_model(load_settings()), spending)

    assert not answer.refused, answer.text
    assert TIGER_TEAM in answer.citations
    assert answer.dropped_citations == ()


def test_another_custodian_gets_the_fixed_refusal(app_pool: Pool, spending: Spending) -> None:
    other = principals("kean-s", [])

    answer = ask(QUESTION, app_pool, other, chat_model(load_settings()), spending)

    assert answer.refused, answer.text
    assert answer.text == REFUSAL
    assert TIGER_TEAM not in answer.citations
