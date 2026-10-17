# The question-answering agent: a LangGraph loop in which the model searches and opens the
# documents the asker may read and queries the company's data as the asker, then answers with
# the figures, the SQL it ran and the ids of the documents it used. Development check:
# PYTHONPATH=src .venv/bin/python -m needtoknow.agent --user district-1 --group district:1 "..."

import argparse
import logging
import operator
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import AfterValidator, validate_call

from needtoknow import auth, data, retrieval
from needtoknow.config import load_settings
from needtoknow.db import Pool, app_pool
from needtoknow.jev import JevError
from needtoknow.methods import Method, Results, search
from needtoknow.providers import Spending, chat_model, price_of
from needtoknow.screen import (
    PART_CHARS,
    SCREEN_DOWN,
    WITHHELD,
    Screen,
    screen_from_settings,
    shown,
    shown_rows,
)

REFUSAL = "I found nothing you have access to on this."
EXHAUSTED = "I ran out of steps before I could answer. Try a narrower question."
NOT_OPEN = "No document with that id is open to you."
DOCUMENT_CHARS = PART_CHARS
MAX_TOOL_ROUNDS = 5
SEARCH_RESULTS = 5

SYSTEM_PROMPT = f"""You answer questions from employees using their company's documents and data.
Call search_documents to find passages. It returns only documents the employee asking may read,
so you can share anything in them. Answer only from the passages and query results you get.
Cite every claim from a document with its id in square brackets, for example [some-document-id].
For figures in the data, call describe_data to pick a schema, describe_data with that schema to
see its tables, then query_data with one read-only query. If the query fails, read the error and
fix the query once; if it fails again, say so. Answer with the figures and the SQL you ran.
If neither the passages nor the data contain the answer, reply exactly: {REFUSAL}"""

# Under prompt_only the search returns every document, and only this instruction stands between
# the asker and the passages they may not read.
LABELLED_SYSTEM_PROMPT = f"""You answer questions from employees using their company's documents.
Call search_documents to find passages. It returns passages from every document, each labelled
with the readers allowed to see it. The employee asking holds these principals: {{principals}}.
Use only passages whose readers include one of those principals, and treat every other passage
as if it did not exist. Answer only from the passages you may use.
Cite every claim with the document id in square brackets, for example [some-document-id].
If the passages you may use do not contain the answer, reply exactly: {REFUSAL}"""

SEARCH_TOOL = {
    "name": "search_documents",
    "description": "Search the company documents and return the passages closest to the query, "
    "each tagged with its document id.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

OPEN_DOCUMENT_TOOL = {
    "name": "open_document",
    "description": "Return the title and full text of one company document, given the document "
    "id that search_documents tagged it with. Use it when the passages leave out what you need.",
    "input_schema": {
        "type": "object",
        "properties": {"doc_id": {"type": "string", "description": "The document id."}},
        "required": ["doc_id"],
        "additionalProperties": False,
    },
}

DESCRIBE_DATA_TOOL = {
    "name": "describe_data",
    "description": "Describe the company's data. With schema null, list the schemas you can "
    "read, each with its comment and number of tables. With a schema, list its tables and views "
    "with their columns, types and comments. Call it with schema null first, then with the "
    "schema you pick, then call query_data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "schema": {
                "type": ["string", "null"],
                "description": "The schema to describe, or null to list the schemas.",
            }
        },
        "required": ["schema"],
        "additionalProperties": False,
    },
}


def query_data_tool(caps: data.Caps) -> dict[str, Any]:
    return {
        "name": "query_data",
        "description": "Run one read-only PostgreSQL SELECT over the company's data and return "
        "the rows as text. Call describe_data with the schema first and query only the tables and "
        "views it lists, named as it writes them. Send a single SELECT without a semicolon. The "
        "database leaves out rows the employee asking may not see, and returns at most "
        f"{caps.rows} rows. If the query fails, read the error and fix the query once.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "One SELECT statement."}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    }


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a non-empty string")
    return value


Text = Annotated[str, AfterValidator(_not_blank)]

_CITATION = re.compile(r"( ?)\[([a-z0-9-]+(?:, *[a-z0-9-]+)*)\]")

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Query:
    sql: str
    rows: str | None
    error: str | None


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    dropped_citations: tuple[str, ...]
    refused: bool
    exhausted: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    queries: tuple[Query, ...]
    withheld: int
    # Every tool result the model was shown. The system prompt and the question are the rest of
    # its input, so a fact absent from these never reached the model.
    tool_outputs: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    sql: str | None


@dataclass(frozen=True)
class Token:
    text: str


@dataclass(frozen=True)
class _Trace:
    returned: tuple[str, ...] = ()
    query: Query | None = None
    withheld: int = 0


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


def ask(
    question: str,
    pool: Pool,
    principals: frozenset[str],
    model: BaseChatModel,
    spending: Spending,
    method: Method = Method.RLS,
    filter_forgotten: bool = False,
    caps: data.Caps = data.DEFAULT_CAPS,
    screen: Screen | None = None,
    history: Sequence[Turn] = (),
) -> Answer:
    tools = _tools(pool, principals, method, filter_forgotten, caps, screen)
    graph = _build_graph(model, _system_prompt(method, principals), tools, spending)
    final = graph.invoke(_start(question, history))
    return _answer(final, spending)


# Yields each tool call as the model makes it and each piece of answer text as it arrives, then
# the checked Answer. Text from a reply that also calls tools streams too, so a client shows
# the final Answer's text in place of what it streamed.
def ask_stream(
    question: str,
    pool: Pool,
    principals: frozenset[str],
    model: BaseChatModel,
    spending: Spending,
    screen: Screen | None = None,
    history: Sequence[Turn] = (),
) -> Iterator[ToolCall | Token | Answer]:
    tools = _tools(pool, principals, Method.RLS, False, data.DEFAULT_CAPS, screen)
    graph = _build_graph(model, _system_prompt(Method.RLS, principals), tools, spending)
    final = _start(question, history)
    for part in graph.stream(final, stream_mode=["messages", "updates", "values"], version="v2"):
        if part["type"] == "messages":
            message, metadata = part["data"]
            if (
                isinstance(message, AIMessageChunk)
                and metadata["langgraph_node"] == "model"
                and message.text
            ):
                yield Token(message.text)
        elif part["type"] == "updates" and "model" in part["data"]:
            for reply in part["data"]["model"]["messages"]:
                for call in reply.tool_calls:
                    sql = call["args"].get("sql") if call["name"] == "query_data" else None
                    yield ToolCall(name=call["name"], sql=sql)
        elif part["type"] == "values":
            final = part["data"]
    yield _answer(final, spending)


def _start(question: str, history: Sequence[Turn]) -> AgentState:
    messages: list[AnyMessage] = []
    for turn in history:
        messages.append(HumanMessage(turn.question))
        messages.append(AIMessage(turn.answer))
    messages.append(HumanMessage(question))
    return {"messages": messages, "input_tokens": 0, "output_tokens": 0}


def _tools(
    pool: Pool,
    principals: frozenset[str],
    method: Method,
    filter_forgotten: bool,
    caps: data.Caps,
    screen: Screen | None,
) -> list[tuple[dict[str, Any], Callable[..., tuple[str, _Trace]]]]:
    # The principals are fixed here, outside the model's reach: each tool takes only one string.
    def search_documents(query: Text) -> tuple[str, _Trace]:
        with pool.connection() as connection:
            results = search(
                connection, principals, query, SEARCH_RESULTS, method, filter_forgotten
            )
        note = ""
        withheld = 0
        if screen is not None:
            try:
                flags = screen.flagged([hit.text for hit in results.hits])
            except JevError:
                return SCREEN_DOWN, _Trace()
            kept = [hit for hit, flag in zip(results.hits, flags, strict=True) if not flag]
            withheld = len(results.hits) - len(kept)
            if withheld:
                note = f"\n\n{withheld} passage(s) {WITHHELD}."
            results = Results(kept, results.dropped, results.readers)
        returned = tuple(hit.doc_id for hit in results.hits)
        return _format_hits(results) + note, _Trace(returned=returned, withheld=withheld)

    def open_document(doc_id: Text) -> tuple[str, _Trace]:
        with pool.connection() as connection:
            document = retrieval.open_document(connection, principals, doc_id)
        if document is None:
            return NOT_OPEN, _Trace()
        title, body = document
        body = body.strip()
        if len(body) > DOCUMENT_CHARS:
            body = body[:DOCUMENT_CHARS] + f"\n[cut at {DOCUMENT_CHARS} characters]"
        text = f"[{doc_id}]\n{title}\n\n{body}"
        visible, was_withheld = shown(screen, text, f"[{doc_id}]")
        returned = (doc_id,) if visible == text else ()
        return visible, _Trace(returned=returned, withheld=1 if was_withheld else 0)

    def describe_data(schema: Text | None) -> tuple[str, _Trace]:
        with pool.connection() as connection:
            description = data.describe_data(connection, schema)
        visible, was_withheld = shown(screen, description, "The data description")
        return visible, _Trace(withheld=1 if was_withheld else 0)

    def query_data(sql: Text) -> tuple[str, _Trace]:
        with pool.connection() as connection:
            result = data.query_data(connection, principals, sql, caps)
        if result.failed:
            query = Query(sql=sql, rows=None, error=result.text)
            visible, was_withheld = shown(screen, result.text, "The query result")
            return visible, _Trace(query=query, withheld=1 if was_withheld else 0)
        visible, withheld = shown_rows(screen, result)
        query = Query(sql=sql, rows=result.text, error=None)
        return visible, _Trace(query=query, withheld=withheld)

    tools: list[tuple[dict[str, Any], Callable[..., tuple[str, _Trace]]]] = [
        (SEARCH_TOOL, search_documents),
        (OPEN_DOCUMENT_TOOL, open_document),
    ]
    # The other methods search as the reader role, which every row is open to.
    if method is Method.RLS:
        tools += [(DESCRIBE_DATA_TOOL, describe_data), (query_data_tool(caps), query_data)]
    return tools


def _system_prompt(method: Method, principals: frozenset[str]) -> str:
    if method is Method.PROMPT_ONLY:
        return LABELLED_SYSTEM_PROMPT.format(principals=", ".join(sorted(principals)))
    return SYSTEM_PROMPT


def _build_graph(
    model: BaseChatModel,
    system_prompt: str,
    tools: list[tuple[dict[str, Any], Callable[..., tuple[str, _Trace]]]],
    spending: Spending,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    model_with_tools = model.bind_tools([spec for spec, _ in tools], strict=True)

    def call_model(state: AgentState) -> dict[str, Any]:
        spending.check()
        reply = model_with_tools.invoke([SystemMessage(system_prompt), *state["messages"]])
        if reply.usage_metadata is None:
            raise RuntimeError("the model reply carries no token usage, so it cannot be priced")
        input_tokens = reply.usage_metadata["input_tokens"]
        output_tokens = reply.usage_metadata["output_tokens"]
        spending.record(input_tokens, output_tokens)
        return {"messages": [reply], "input_tokens": input_tokens, "output_tokens": output_tokens}

    def after_model(state: AgentState) -> str:
        replies = 0
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage):
                replies += 1
        if replies > MAX_TOOL_ROUNDS:
            return END
        return tools_condition(state["messages"])

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([_tool(spec, run) for spec, run in tools]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


def _tool(spec: dict[str, Any], run: Callable[..., tuple[str, _Trace]]) -> StructuredTool:
    return StructuredTool(
        name=spec["name"],
        description=spec["description"],
        args_schema=spec["input_schema"],
        func=validate_call(run),
        response_format="content_and_artifact",
    )


def _format_hits(results: Results) -> str:
    if not results.hits:
        return "No documents found."
    passages = []
    for hit in results.hits:
        label = f"[{hit.doc_id}]"
        if hit.doc_id in results.readers:
            label += f" (readers: {', '.join(results.readers[hit.doc_id])})"
        passages.append(f"{label}\n{hit.text}")
    return "\n\n".join(passages)


def _answer(final: Mapping[str, Any], spending: Spending) -> Answer:
    returned: set[str] = set()
    tool_outputs: list[str] = []
    queries: list[Query] = []
    withheld = 0
    for message in final["messages"]:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        trace: _Trace = message.artifact
        tool_outputs.append(message.text)
        returned.update(trace.returned)
        if trace.query is not None:
            queries.append(trace.query)
        withheld += trace.withheld

    input_tokens = final["input_tokens"]
    output_tokens = final["output_tokens"]
    cost_usd = spending.price.cost(input_tokens, output_tokens)

    reply = final["messages"][-1]
    exhausted = isinstance(reply, AIMessage) and bool(reply.tool_calls)
    text = "" if exhausted else reply.text.strip()
    refused = not exhausted and (not text or text == REFUSAL)
    citations: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    if exhausted:
        text = EXHAUSTED
    elif refused:
        text = REFUSAL
    else:
        text, citations, dropped = _check_citations(text, returned)
    if dropped:
        _log.warning("dropped citations the search never returned: %s", ", ".join(dropped))
    return Answer(
        text=text,
        citations=citations,
        dropped_citations=dropped,
        refused=refused,
        exhausted=exhausted,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        queries=tuple(queries),
        withheld=withheld,
        tool_outputs=tuple(tool_outputs),
    )


def _check_citations(text: str, returned: set[str]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    kept: list[str] = []
    dropped: list[str] = []

    def keep_returned(match: re.Match[str]) -> str:
        cited = [doc_id.strip() for doc_id in match.group(2).split(",")]
        verified = []
        for doc_id in cited:
            if doc_id in returned:
                verified.append(doc_id)
                kept.append(doc_id)
            else:
                dropped.append(doc_id)
        if not verified:
            return ""
        return match.group(1) + "".join(f"[{doc_id}]" for doc_id in verified)

    checked = _CITATION.sub(keep_returned, text)
    return checked, tuple(dict.fromkeys(kept)), tuple(dict.fromkeys(dropped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question as one user.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--group", action="append", default=[], help="repeat for each group")
    parser.add_argument("question")
    args = parser.parse_args()

    settings = load_settings()
    spending = Spending(price_of(settings), settings.budget_usd)
    principals = auth.principals(args.user, args.group)
    with app_pool(settings, size=1) as pool:
        screen = screen_from_settings(settings)
        answer = ask(args.question, pool, principals, chat_model(settings), spending, screen=screen)

    print(answer.text)
    print(f"citations: {', '.join(answer.citations) or 'none'}")
    if answer.dropped_citations:
        print(f"dropped citations: {', '.join(answer.dropped_citations)}")
    print(
        f"{settings.model}: {answer.input_tokens} input and {answer.output_tokens} output "
        f"tokens, {answer.cost_usd:.6f} USD"
    )


if __name__ == "__main__":
    main()
