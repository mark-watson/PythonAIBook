"""Tool schemas, execution, and the agentic loop."""

from __future__ import annotations

from typing import Any, cast

import pytest

import litelm
from tests.conftest import FakeTransport, chat_body, tool_call_body


def weather(args: dict[str, Any]) -> str:
    return f"sunny and 22C in {args['location']}"


def weather_tool() -> litelm.Tool:
    return litelm.make_tool(
        "get_weather",
        "Get the current weather for a location",
        [("location", "string", "City name, e.g. Paris")],
        weather,
    )


def fail_tool(args: dict[str, Any]) -> str:
    raise RuntimeError(f"no weather for {args['location']}")


class TestMakeTool:
    def test_tuple_parameters_become_a_json_schema(self) -> None:
        schema = litelm.tool_schemas([weather_tool()])[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["parameters"] == {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Paris"}
            },
            "required": ["location"],
        }

    def test_optional_and_enum_parameters(self) -> None:
        tool = litelm.make_tool(
            "get_weather",
            "weather",
            [
                litelm.param("location", "string", "City name"),
                litelm.param(
                    "units",
                    "string",
                    "celsius or fahrenheit",
                    required=False,
                    enum=("celsius", "fahrenheit"),
                ),
            ],
            weather,
        )
        parameters = litelm.tool_schemas([tool])[0]["function"]["parameters"]
        assert parameters["required"] == ["location"]
        assert parameters["properties"]["units"]["enum"] == ["celsius", "fahrenheit"]

    def test_a_tool_without_a_handler_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="handler"):
            litelm.make_tool("t", "d", [])

    def test_a_non_callable_handler_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="handler must be callable"):
            litelm.Tool(
                name="t",
                description="d",
                parameters=[],
                func=cast(litelm.Handler, "nope"),
            )

    def test_a_bad_parameter_spec_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="parameter spec"):
            litelm.make_tool("t", "d", cast(Any, [("only", "two")]), weather)

    def test_tool_list_and_mapping_are_both_accepted(self) -> None:
        tool = weather_tool()
        assert litelm.tool_schemas([tool]) == litelm.tool_schemas({"get_weather": tool})

    def test_a_non_tool_in_the_list_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="Tool values"):
            litelm.normalize_tools(cast(Any, ["not a tool"]))


class TestExecuteToolCalls:
    def call(
        self, name: str, arguments: str = '{"location": "Paris"}', call_id: str = "c1"
    ) -> litelm.ToolCall:
        return litelm.parse_chat_response(
            chat_body(
                content=None, tool_calls=[tool_call_body(name, arguments, call_id)]
            ),
            "test-model",
        ).tool_calls[0]

    def test_happy_path(self) -> None:
        (result,) = litelm.execute_tool_calls(
            [weather_tool()], [self.call("get_weather")]
        )
        assert result == litelm.ToolResult(
            call_id="c1", name="get_weather", result="sunny and 22C in Paris"
        )

    def test_unknown_tool_is_reported_to_the_model(self) -> None:
        (result,) = litelm.execute_tool_calls([weather_tool()], [self.call("nope")])
        assert result.result == "Error: unknown tool: nope"

    def test_missing_required_argument_is_reported(self) -> None:
        (result,) = litelm.execute_tool_calls(
            [weather_tool()], [self.call("get_weather", "{}")]
        )
        assert result.result == (
            "Error: tool 'get_weather' missing required argument(s): location"
        )

    def test_malformed_json_arguments_are_reported(self) -> None:
        (result,) = litelm.execute_tool_calls(
            [weather_tool()], [self.call("get_weather", "{not json")]
        )
        assert "invalid JSON arguments" in result.result

    def test_a_handler_exception_becomes_a_result_string(self) -> None:
        tool = litelm.make_tool(
            "get_weather", "weather", [("location", "string", "City")], fail_tool
        )
        (result,) = litelm.execute_tool_calls([tool], [self.call("get_weather")])
        assert result.result.startswith("Error: tool 'get_weather' raised:")

    def test_none_result_becomes_an_empty_string(self) -> None:
        tool = litelm.make_tool("drop", "drop it", [], lambda args: None)
        (result,) = litelm.execute_tool_calls([tool], [self.call("drop", "{}")])
        assert result.result == ""

    def test_non_string_results_are_stringified(self) -> None:
        tool = litelm.make_tool("count", "count", [], lambda args: 7)
        (result,) = litelm.execute_tool_calls([tool], [self.call("count", "{}")])
        assert result.result == "7"

    def test_locally_built_calls_skip_json_validation(self) -> None:
        call = litelm.ToolCall(
            id="c9", name="get_weather", arguments={"location": "Rome"}
        )
        (result,) = litelm.execute_tool_calls([weather_tool()], [call])
        assert result.result == "sunny and 22C in Rome"


class TestChatWithTools:
    def test_runs_the_loop_until_the_model_answers(
        self, fake_transport: FakeTransport
    ) -> None:
        tool = weather_tool()
        fake_transport.json_results = [
            chat_body(
                content=None,
                tool_calls=[tool_call_body("get_weather", '{"location": "Paris"}')],
                finish_reason="tool_calls",
            ),
            chat_body(content="It is sunny and 22C in Paris."),
        ]

        response = litelm.chat_with_tools(
            "ollama/llama3.2:3b", "Weather in Paris?", [tool]
        )

        assert response.content == "It is sunny and 22C in Paris."
        assert len(fake_transport.requests) == 2
        second = fake_transport.requests[1]["payload"]["messages"]
        assert second[1]["role"] == "assistant"
        assert second[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert second[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny and 22C in Paris",
        }

    def test_a_name_to_tool_mapping_works_as_the_registry(
        self, fake_transport: FakeTransport
    ) -> None:
        """`Tools` is documented as a list *or* a mapping; the mapping path must
        resolve tools rather than reporting every call as unknown."""
        weather = weather_tool()
        fake_transport.json_results = [
            chat_body(
                content=None,
                tool_calls=[tool_call_body("get_weather", '{"location": "Paris"}')],
                finish_reason="tool_calls",
            ),
            chat_body(content="Sunny."),
        ]

        response = litelm.chat_with_tools(
            "ollama/llama3.2:3b", "Weather?", {"get_weather": weather}
        )

        assert response.content == "Sunny."
        tool_message = fake_transport.requests[1]["payload"]["messages"][2]
        assert tool_message["content"] == "sunny and 22C in Paris"

    def test_stops_at_max_iterations(self, fake_transport: FakeTransport) -> None:
        tool = weather_tool()
        fake_transport.json_results = [
            chat_body(
                content=None,
                tool_calls=[tool_call_body("get_weather", '{"location": "Paris"}')],
                finish_reason="tool_calls",
            )
        ]

        response = litelm.chat_with_tools(
            "ollama/llama3.2:3b", "Weather?", [tool], max_iterations=1
        )

        assert response.tool_calls
        assert len(fake_transport.requests) == 1

    def test_rejects_a_useless_iteration_budget(self) -> None:
        with pytest.raises(litelm.LitelmError, match="max_iterations"):
            litelm.chat_with_tools("ollama/llama3.2:3b", "hi", [], max_iterations=0)
