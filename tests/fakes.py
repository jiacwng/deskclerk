# A chat model that replays scripted replies and records every prompt it is sent, and a Jev that
# answers from a function over the state, so the agent runs in tests without the paid APIs.

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx2
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field
from typesafe_sdk import RetryPolicy, TypeSafeClient

from needtoknow.jev import Jev
from needtoknow.providers import PRICES

HAIKU = PRICES["anthropic:claude-haiku-5-5"]
USAGE: UsageMetadata = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


class ScriptedModel(GenericFakeChatModel):
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    options: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        self.options.append(kwargs)
        return super()._generate(messages, stop, run_manager, **kwargs)


def scripted(*replies: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(replies))


def call_tool(name: str, args: dict[str, Any]) -> AIMessage:
    call = tool_call(name=name, args=args, id="toolu_test")
    return AIMessage(content="", tool_calls=[call], usage_metadata=USAGE)


def search_call(args: dict[str, Any]) -> AIMessage:
    return call_tool("search_documents", args)


def final_reply(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata=USAGE)


def jev_client(answer: Callable[[httpx2.Request], httpx2.Response]) -> TypeSafeClient:
    return TypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(answer),
        retry=RetryPolicy(max_retries=0),
        compression=False,
    )


def fake_jev(probability: Callable[[Any], float], states: list[Any] | None = None) -> Jev:
    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if states is not None:
            states.append(body["state"])
        noul = {"type": "noul", "noul": probability(body["state"])}
        usage = {"input_tokens": 100, "output_tokens": 0}
        payload = {"model": "jev-1.13.0", "answers": {"answer": noul}, "usage": usage}
        return httpx2.Response(200, json=payload)

    return Jev(jev_client(answer), budget_usd=1.0)
