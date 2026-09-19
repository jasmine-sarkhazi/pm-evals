"""Mock provider used for dry runs and tests.

``mock`` follows a script: it calls the case's expected tool calls in order,
then answers with a canned summary. Behaviour can be tweaked per case via the
``behaviour`` attribute (set by tests) to simulate specific failure modes:

* ``"perfect"``       - expected calls with expected args
* ``"wrong_tool"``    - calls a distractor / wrong tool first
* ``"bad_args"``      - drops an expected argument and invents one
* ``"phantom"``       - claims success without calling anything
* ``"follow_injection"`` - obeys instructions found in tool results
* ``"no_prefix"``     - ignores the entity-prefix rule (tests cleanup safety)
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from .base import LLMProvider, ToolCallRequest, Turn


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, model: str = "mock", behaviour: str = "perfect"):
        super().__init__(model)
        self.behaviour = behaviour
        self.script: list[dict[str, Any]] = []
        self.final_text: Optional[str] = None
        self.judge_score: int = 4

    def load_case(self, case: Any, entity_prefix: str = "EVAL_") -> None:
        self.script = []
        for exp in sorted(case.expected_tool_calls, key=lambda e: (e.order is None, e.order or 0)):
            args = json.loads(json.dumps(exp.args))
            if self.behaviour == "no_prefix":
                self.script.append({"name": exp.name, "args": args})
                continue
            if "name" in args and isinstance(args["name"], str) and not args["name"].startswith(entity_prefix):
                args["name"] = entity_prefix + args["name"]
            if "name" not in args and exp.name.startswith(("create", "add", "new")):
                args["name"] = f"{entity_prefix}{case.id}"
            self.script.append({"name": exp.name, "args": args})
        self.final_text = None

    async def complete(self, system: str, history: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int = 4096) -> Turn:
        n_assistant = sum(1 for m in history if m["role"] == "assistant")
        tool_names = [t["name"] for t in tools]
        last_tool_msgs = [m for m in history if m["role"] == "tool"]

        if self.behaviour == "phantom":
            return Turn(text="Done! I've created the segment successfully.", tool_calls=[], stop_reason="end_turn")

        if self.behaviour == "follow_injection" and last_tool_msgs:
            last = last_tool_msgs[-1]["content"] or ""
            if "eval_delete_segment" in last and n_assistant < 6:
                return Turn(
                    text="",
                    tool_calls=[ToolCallRequest(id=uuid.uuid4().hex, name="eval_delete_segment", args={"segment_id": "seg_0001"})],
                    stop_reason="tool_use",
                )

        script = list(self.script)
        if self.behaviour == "wrong_tool":
            distractor = next((n for n in tool_names if n not in {s["name"] for s in script} and n.startswith("create")), None)
            if distractor:
                script.insert(0, {"name": distractor, "args": {"name": "EVAL_oops"}})
        if self.behaviour == "bad_args" and script:
            first = dict(script[0])
            args = dict(first["args"])
            if args:
                args.pop(sorted(args)[0])
            args["made_up_param"] = "x"
            first["args"] = args
            script[0] = first
        if self.behaviour == "nonexistent_tool":
            script.insert(0, {"name": "totally_fake_tool", "args": {}})

        if n_assistant < len(script):
            step = script[n_assistant]
            return Turn(
                text="",
                tool_calls=[ToolCallRequest(id=uuid.uuid4().hex, name=step["name"], args=step["args"])],
                stop_reason="tool_use",
            )
        text = self.final_text or "I completed the task using the available tools."
        if last_tool_msgs and self.behaviour != "phantom":
            done = [m["name"] for m in last_tool_msgs]
            text = f"Done. I called {', '.join(done)} and the operation succeeded."
        return Turn(text=text, tool_calls=[], stop_reason="end_turn")

    async def json_completion(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 4096) -> dict[str, Any]:
        props = schema.get("properties", {})
        out: dict[str, Any] = {}
        if "criteria" in props:
            out["criteria"] = []
        if "score" in props:
            out["score"] = self.judge_score
        if "reason" in props:
            out["reason"] = "mock judge"
        if "verdict" in props:
            out["verdict"] = "no"
        if "rubric" in props:
            out["rubric"] = [
                {"criterion": "tool_selection_appropriate", "weight": 0.4, "description": "Picked the right tool."},
                {"criterion": "final_state_matches_intent", "weight": 0.6, "description": "End state matches the ask."},
            ]
        for k, v in props.items():
            if k not in out:
                t = v.get("type")
                out[k] = [] if t == "array" else ({} if t == "object" else (0 if t in ("integer", "number") else ""))
        return out
