# The question-answering agent: a LangGraph loop in which the model searches and opens the
# documents the asker may read and queries the company's data as the asker, then answers with
# the figures, the SQL it ran and the ids of the documents it used. Development check:
# PYTHONPATH=src .venv/bin/python -m needtoknow.agent --user district-1 --group district:1 "..."

import argparse
import logging
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from needtoknow import auth, data, retrieval
from needtoknow.config import load_settings
from needtoknow.db import connect_app
from needtoknow.methods import Method, Results, search
from needtoknow.providers import Spending, chat_model

REFUSAL = "I found nothing you have access to on this."
NOT_OPEN = "No document with that id is open to you."
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
    "description": "Describe the company's data. Without a schema, list the schemas you can "
    "read, each with its comment and number of tables. With a schema, list its tables and views "
    "with their columns, types and comments. Call it without a schema first, then with the "
    "schema you pick, then call query_data.",
    "input_schema": {
        "type": "object",
        "properties": {"schema": {"type": "string", "description": "The schema to describe."}},
        "required": [],
        "additionalProperties": False,
    },
}

QUERY_DATA_TOOL = {
    "name": "query_data",
    "description": "Run one read-only PostgreSQL SELECT over the company's data and return the "
    "rows as text. Call describe_data with the schema first and query only the tables and views "
    "it lists, named as it writes them. Send a single SELECT without a semicolon. The database "
    "leaves out rows the employee asking may not see, and returns at most "
    f"{data.ROW_CAP} rows. If the query fails, read the error and fix the query once.",
    "input_schema": {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "One SELECT statement."}},
        "required": ["sql"],
        "additionalProperties": False,
    },
}

TOOL_SPECS = [
    SEARCH_TOOL,
    OPEN_DOCUMENT_TOOL,
    DESCRIBE_DATA_TOOL,
    QUERY_DATA_TOOL,
]

# A tool's argument name, whether the model may leave it out, and the function that runs it.
Tools = dict[str, tuple[str, bool, Callable[..., str]]]

_CITATION = re.compile(r"( ?)\[([a-z0-9-]+(?:, *[a-z0-9-]+)*)\]")

_log = logging.getLogger(__name__)


# One query_data call: the rows as query_data printed them, or its error.
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
    input_tokens: int
    output_tokens: int
    cost_usd: float
    queries: tuple[Query, ...]
    # Every tool result the model was shown. The system prompt and the question are the rest of
    # its input, so a fact absent from these never reached the model.
    tool_outputs: tuple[str, ...] = field(repr=False)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


def ask(
    question: str,
    connection: psycopg.Connection[Any],
    principals: frozenset[str],
    model: BaseChatModel,
    spending: Spending,
    method: Method = Method.RLS,
    filter_forgotten: bool = False,
) -> Answer:
    returned: set[str] = set()
    outputs: list[str] = []
    queries: list[Query] = []

    # The principals are fixed here, outside the model's reach: each tool takes only one string.
    def search_documents(query: str) -> str:
        results = search(connection, principals, query, SEARCH_RESULTS, method, filter_forgotten)
        returned.update(hit.doc_id for hit in results.hits)
        outputs.append(_format_hits(results))
        return outputs[-1]

    def open_document(doc_id: str) -> str:
        document = retrieval.open_document(connection, principals, doc_id)
        if document is None:
            outputs.append(NOT_OPEN)
        else:
            returned.add(doc_id)
            title, body = document
            outputs.append(f"[{doc_id}]\n{title}\n\n{body.strip()}")
        return outputs[-1]

    def describe_data(schema: str | None = None) -> str:
        outputs.append(data.describe_data(connection, schema))
        return outputs[-1]

    def query_data(sql: str) -> str:
        result = data.query_data(connection, principals, sql)
        if result.failed:
            queries.append(Query(sql=sql, rows=None, error=result.text))
        else:
            queries.append(Query(sql=sql, rows=result.text, error=None))
        outputs.append(result.text)
        return result.text

    tools: Tools = {
        "search_documents": ("query", False, search_documents),
        "open_document": ("doc_id", False, open_document),
    }
    # The other methods search as the reader role, which every row is open to.
    if method is Method.RLS:
        tools["describe_data"] = ("schema", True, describe_data)
        tools["query_data"] = ("sql", False, query_data)
    graph = _build_graph(model, _system_prompt(method, principals), tools, spending)
    final = graph.invoke(
        {
            "messages": [HumanMessage(question)],
            "tool_rounds": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )
    return _answer(final, returned, tuple(outputs), tuple(queries), spending)


def _system_prompt(method: Method, principals: frozenset[str]) -> str:
    if method is Method.PROMPT_ONLY:
        return LABELLED_SYSTEM_PROMPT.format(principals=", ".join(sorted(principals)))
    return SYSTEM_PROMPT


def _build_graph(
    model: BaseChatModel,
    system_prompt: str,
    tools: Tools,
    spending: Spending,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    specs = [spec for spec in TOOL_SPECS if spec["name"] in tools]
    model_with_tools = model.bind_tools(specs, strict=True)

    def call_model(state: AgentState) -> dict[str, Any]:
        spending.check()
        reply = model_with_tools.invoke([SystemMessage(system_prompt), *state["messages"]])
        if reply.usage_metadata is None:
            raise RuntimeError("the model reply carries no token usage, so it cannot be priced")
        input_tokens = reply.usage_metadata["input_tokens"]
        output_tokens = reply.usage_metadata["output_tokens"]
        spending.record(input_tokens, output_tokens)
        return {"messages": [reply], "input_tokens": input_tokens, "output_tokens": output_tokens}

    def call_tools(state: AgentState) -> dict[str, Any]:
        reply = state["messages"][-1]
        calls = reply.tool_calls if isinstance(reply, AIMessage) else []
        results = [_run_tool(call, tools) for call in calls]
        return {"messages": results, "tool_rounds": state["tool_rounds"] + 1}

    def after_model(state: AgentState) -> str:
        reply = state["messages"][-1]
        if isinstance(reply, AIMessage) and reply.tool_calls:
            if state["tool_rounds"] < MAX_TOOL_ROUNDS:
                return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


def _run_tool(call: ToolCall, tools: Tools) -> ToolMessage:
    call_id = call["id"] or ""
    if call["name"] not in tools:
        error = f"unknown tool, call one of {', '.join(tools)}"
        return ToolMessage(error, tool_call_id=call_id, status="error")
    argument, optional, run = tools[call["name"]]
    if optional and not call["args"]:
        return ToolMessage(run(), tool_call_id=call_id)
    if set(call["args"]) != {argument}:
        error = f"wrong arguments, call {call['name']} with a {argument} only"
        if optional:
            error += " or with no arguments"
        return ToolMessage(error, tool_call_id=call_id, status="error")
    value = call["args"][argument]
    if not isinstance(value, str) or not value.strip():
        error = f"{argument} must be a non-empty string"
        return ToolMessage(error, tool_call_id=call_id, status="error")
    return ToolMessage(run(value), tool_call_id=call_id)


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


def _answer(
    final: dict[str, Any],
    returned: set[str],
    tool_outputs: tuple[str, ...],
    queries: tuple[Query, ...],
    spending: Spending,
) -> Answer:
    input_tokens = final["input_tokens"]
    output_tokens = final["output_tokens"]
    cost_usd = spending.price.cost(input_tokens, output_tokens)

    reply = final["messages"][-1]
    text = ""
    if isinstance(reply, AIMessage) and not reply.tool_calls:
        text = reply.text.strip()
    refused = not text or text == REFUSAL
    citations: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    if refused:
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
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        queries=queries,
        tool_outputs=tool_outputs,
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
    spending = Spending(settings.model, settings.budget_usd)
    principals = auth.principals(args.user, args.group)
    with connect_app(settings) as app:
        answer = ask(args.question, app, principals, chat_model(settings), spending)

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
