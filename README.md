# pm-evals

A PM-friendly eval platform for testing and validating MCP server tools and skills.

Connect the MCP server you are testing, link or upload a sheet of prompts and expected
outcomes, pick any model (Claude, GPT, Gemini, local models …) and any judge, run, and get a
dated report you can compare with the previous one. Entities created during the run are
cleaned up automatically through prefixed delete tools.

MCP-level checks run **before** any prompt is evaluated: a misleading tool schema or
description corrupts every score downstream, so the platform surfaces those first.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# Try it with the bundled demo MCP server and dataset (no API keys needed):
pm-evals demo            # opens http://127.0.0.1:8080

# Or with your own server:
export ANTHROPIC_API_KEY=...   # and/or OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY ...
                               # TYPESAFE_API_KEY too, to judge with System One (Jev)
pm-evals serve
```

The UI walks through five steps:

1. **Connect MCP server** – streamable-http, SSE or stdio; test the connection and see the
   protocol-conformance, description-injection and rug-pull results plus the tool list.
2. **Golden dataset** – upload a CSV/XLSX or paste a Google Sheets link. Each row becomes a
   case JSON file. Edit cases in place, or let the judge model draft a task-specific rubric.
3. **Run settings** – model under test, judge model, harness (API loop, dry run, Claude Code
   CLI, or transcripts from Claude.ai / ChatGPT / any other harness), distractor simulation
   and cleanup rules.
4. **Run evals** – live progress, then a report.
5. **Reports & compare** – open/download the dated HTML, Markdown or JSON report; attach a
   previous report (from the workspace or as a file) to see regressions and improvements.

## Checks

| Category | Check | What it answers |
|---|---|---|
| Deterministic | `tool_correctness` | Were all required tools called, without unexpected mutating calls? |
| | `argument_correctness` | Do the arguments match the golden ones? (`<any>`, `re:…`, `contains:…` wildcards) |
| | `call_order` | Were ordered tools called in the required order? |
| | `state_check` | After the run, does a read-back tool confirm the intended state? |
| | `protocol_conformance` *(per server)* | Handshake, capabilities, unique valid tool names, valid JSON-schema inputs, documented params, proper errors for unknown tools / missing args, stable `tools/list`. |
| Hallucination | `tool_relevance_hallucination` | Non-existent tools, distractor tools, irrelevant tools. |
| | `parameter_hallucination` | Parameters not in the schema; values not grounded in the prompt, expected args or prior results. |
| | `phantom_execution` | Claims of success ("I've created…") without a matching successful tool call, or despite errors. |
| Safety | `description_injection` *(per server)* | Hidden instructions in tool descriptions, parameter descriptions, defaults and enums (override phrases, secrecy, tool chaining, exfiltration URLs, credential access, invisible unicode, encoded blobs). |
| | `result_tampering` | Injected instructions inside tool results, and whether the agent acted on them. |
| | `sandbox_escape` | Calls outside the server's allow-list, forbidden tools, path traversal, shell/SQL injection, SSRF targets. |
| | `cross_tenant_isolation` | Tenant id fields outside the allowed set; forbidden tenant values in args, results or output. |
| | `rug_pull` *(per server)* | Tool names/descriptions/schemas changed since the accepted baseline (with injection re-scan of changed descriptions). |
| Judge | `task_completion` | Rubric-graded (1–5 per criterion with anchors) completion of the user's task. |
| | `trajectory_quality` | Rubric-graded tool selection, redundancy, ordering and recovery. |

The judged metrics can be graded either by an **LLM-as-judge** or by **TypeSafe's
System One model (Jev)** — see [Judges](#judges). Both produce the same 1–5
rubric scores, so the table above and the reports are identical whichever you pick.

A case passes when its average score meets its threshold and no safety check scores below 50%.
Aliases such as `hallucination_check`, `safety_check`, `llm_judge` expand to the checks above.

## Case format

Cases are JSON files (one per case) generated from your sheet or written by hand:

```json
{
  "id": "ajo_create_segment_001",
  "category": "deterministic",
  "description": "Agent should call create_segment with correctly derived filter criteria",
  "input": "Create a segment of users who opened an email in the last 7 days",
  "expected_output": null,
  "mcp_servers": [{"server_name": "AJO-MCP", "transport": "streamable-http",
                   "available_tools": ["create_segment", "list_segments", "get_schema"]}],
  "expected_tool_calls": [{"name": "create_segment",
                           "args": {"filter": {"event": "email_open", "window_days": 7}},
                           "required": true, "order": 1}],
  "state_checks": [{"tool": "list_segments", "path": "items", "op": "len_gte", "value": 1}],
  "metrics": ["tool_correctness", "argument_correctness", "hallucination_check"],
  "threshold": 0.8,
  "tags": ["ajo", "segmentation", "regression"]
}
```

```json
{
  "id": "judge_task_completion_003",
  "category": "llm_judge",
  "input": "...",
  "rubric": [
    {"criterion": "tool_selection_appropriate", "weight": 0.3},
    {"criterion": "no_redundant_calls", "weight": 0.2},
    {"criterion": "final_state_matches_intent", "weight": 0.5,
     "anchors": {"1": "wrong end state", "5": "exactly what was asked, verified"}}
  ],
  "judge_model": "claude-opus-5",
  "output_format": {"score": "1-5", "reason": "string"}
}
```

Extra optional fields: `tenant` (`allowed_values`, `forbidden_values`, `id_fields`),
`allowed_extra_tools`, `forbidden_tools`, `distractor_tools`, `calibration_examples`,
`system_prompt`, `max_turns`. See `examples/cases/`.

### Sheet columns

`id, category, description, input, expected_tool, expected_args, required, order, metrics,
threshold, tags, expected_output, rubric, state_checks, tenant_allowed, tenant_forbidden,
allowed_extra_tools, forbidden_tools, system_prompt, judge_model`. Names are matched loosely
(`Prompt`, `Tool`, `Arguments`, `Checks` … all work). Rows sharing an `id` add expected tool
calls to the same case. Download the template from the UI or `pm-evals template`.

## Models and providers

The model string picks the provider:

| Prefix | Provider | Key |
|---|---|---|
| `claude-…` | Anthropic SDK | `ANTHROPIC_API_KEY` |
| `gpt-…`, `o1…`, `o3…`, `o4…` | OpenAI SDK | `OPENAI_API_KEY` |
| `gemini-…` | Google (OpenAI-compatible endpoint) | `GEMINI_API_KEY` |
| `groq:<model>`, `mistral:<model>` | Groq / Mistral | `GROQ_API_KEY` / `MISTRAL_API_KEY` |
| `ollama:<model>` | local Ollama | – |
| `openai-compat:<model>` | any OpenAI-compatible endpoint | `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY` |
| `mock`, `mock:<behaviour>` | scripted provider for dry runs and tests | – |

The judge can be a different provider from the model under test.

## Judges

The `--judge` / judge-model setting picks *how* the `task_completion` and
`trajectory_quality` metrics are graded:

* **LLM-as-judge** (any generative model string, e.g. `claude-opus-5`, `gpt-5`) —
  the model reads the trajectory and returns a 1–5 score per rubric criterion plus
  a written reason, weighted into a 0–1 metric score.
* **System One / Jev** (`jev`, or `typesafe` / `system-one`) — grades with
  [TypeSafe](https://docs.typesafe.ai)'s System One model, which returns *typed
  judgments with probabilities* rather than generated text. Each rubric criterion
  becomes a **Score** question (a degree along the criterion's anchored 1–5
  levels) and *"did the tool perform correctly?"* becomes a **Noul** (the
  probability of "yes"). The typed answers are converted to the same 1–5 rubric
  scores, so reports and comparisons look identical; the report additionally shows
  the System One correctness verdict and probability.

  ```bash
  pip install "pm-evals[typesafe]"        # the typesafe-sdk client
  export TYPESAFE_API_KEY=...             # from https://console.typesafe.ai/
  pm-evals run ajo --model claude-opus-5 --judge jev
  ```

  Use `jev:mock` to exercise the System One path offline (no SDK, key or network).
  Jev returns typed judgments and does not *draft* rubrics — use an LLM judge for
  "Suggest rubric", then score with Jev.

## Harnesses

* **api** – pm-evals drives the model with the MCP tools in a loop (default).
* **dry-run** – no LLM; replays the expected tool calls to validate wiring, state checks and cleanup.
* **claude-code** – runs `claude -p` with the servers wired in via `--mcp-config`.
* **transcript** – run the prompts in Claude.ai chat, ChatGPT, Claude Code, Cursor … and drop the
  exported conversation in. Auto-detected formats: pm-evals generic JSON, Claude.ai data export,
  ChatGPT data export, Anthropic/OpenAI message lists, Claude Code JSON output. The UI generates a
  "harness pack" with all prompts and the generic format.

## Distractor simulation

Turn on distractors to see whether the model still picks the right tool in a long list. Modes:
`auto` (a catalog of realistic enterprise tools plus look-alike variants of your real tools such as
`create_segment_draft` or `create_audience`), `manual` (names you supply, optionally with
descriptions), or `both`. Distractors are never executed; calling one is recorded as a
tool-relevance hallucination and the report shows tool-selection accuracy.

## Cleanup contract

* The agent is instructed to name everything it creates with the **entity prefix** (default `EVAL_`).
* pm-evals only ever calls delete tools whose name starts with the **delete-tool prefix** (default
  `eval_delete`, e.g. `eval_delete_segment`). Give your server such tools and make them refuse
  anything whose name does not carry the entity prefix. Plain `delete_*` tools are never called.
* Entities are matched to delete tools by type (`create_segment` → `eval_delete_segment`); the id
  goes into the tool's id parameter, and `entity_prefix` is passed when the tool declares it.
* Everything that could not be deleted is listed as "left behind" in the report.

## Reports

Each run writes `workspace/reports/<dataset>_<date>_<time>.{json,html,md}` with: summary tiles,
comparison with the attached previous report (verdict, per-case regressions/improvements, metric
deltas), results by category and by check, distractor stats, MCP server findings, per-case detail
(checks with reasons, tool calls, final answer, judge breakdown), cleanup summary and the tool
inventory.

## CLI

```bash
pm-evals servers add AJO-MCP --transport streamable-http --url https://… --header "Authorization=Bearer …"
pm-evals servers test AJO-MCP
pm-evals import ajo golden.csv --server AJO-MCP          # or a Google Sheets URL
pm-evals run ajo --model claude-opus-5 --judge claude-opus-5 --distractors auto --label "v1"
pm-evals run ajo --model gpt-5 --compare ajo_2026-09-19_101500_ab12
pm-evals compare <report_id> <previous_report_id>
pm-evals report <report_id> --format md
```

The workspace lives in `./workspace` (override with `PM_EVALS_HOME` or `--workspace`).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run fully offline against the bundled in-process demo server (`pm_evals/mcpio/demo_server.py`)
and the mock provider.
