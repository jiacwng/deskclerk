# A chat model that replays scripted replies and records every prompt it is sent, so the agent
# runs in tests without the paid API.

from collections.abc import Callable, Sequence
from typing import Any

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
