"""Distractor tool simulation.

PMs can flood the tool list with plausible-but-wrong tools to see whether the
model still picks the right one. Two sources:

* ``manual`` - names (and optional descriptions) the PM supplies.
* ``auto``   - a catalog of realistic enterprise tools plus *near-miss*
  variants of the real tools (``create_segment`` -> ``create_segment_draft``,
  ``create_audience`` ...), which are the hardest to distinguish.

Distractors are never executed. If the model calls one, the call is recorded
as a tool-relevance hallucination and the model is told the tool is
unavailable.
"""

from __future__ import annotations

import random
import re
from typing import Iterable

from ..models import DistractorConfig, DistractorTool

_CATALOG: list[tuple[str, str]] = [
    ("create_campaign", "Create a marketing campaign with a name, channel and start date."),
    ("update_campaign", "Update fields on an existing campaign."),
    ("list_campaigns", "List campaigns filtered by status."),
    ("create_journey", "Create a customer journey with entry conditions."),
    ("publish_journey", "Publish a journey so it starts processing customers."),
    ("create_audience_from_csv", "Upload a CSV and create a static audience from it."),
    ("export_audience", "Export an audience to a downloadable CSV."),
    ("get_email_template", "Fetch an email template by id."),
    ("create_email_template", "Create a reusable email template."),
    ("send_test_email", "Send a test rendering of an email template to an address."),
    ("schedule_email_send", "Schedule an email send to an audience at a time."),
    ("create_push_notification", "Create a mobile push notification message."),
    ("create_sms_message", "Create an SMS message."),
    ("get_delivery_report", "Return delivery metrics for a message."),
    ("get_open_rate", "Return the open rate for a message over a window."),
    ("get_click_rate", "Return click-through rate for a message."),
    ("list_events", "List tracked event types available in the data model."),
    ("query_events", "Query raw events with a filter expression."),
    ("create_event_trigger", "Create a trigger that fires on an event."),
    ("create_dashboard", "Create an analytics dashboard."),
    ("add_dashboard_widget", "Add a chart widget to a dashboard."),
    ("run_report", "Run a saved analytics report."),
    ("create_report", "Create a saved analytics report."),
    ("create_contact", "Create a CRM contact."),
    ("update_contact", "Update a CRM contact."),
    ("merge_contacts", "Merge duplicate contacts."),
    ("create_deal", "Create a sales deal in the pipeline."),
    ("update_deal_stage", "Move a deal to a new pipeline stage."),
    ("create_task", "Create a task for a teammate."),
    ("assign_task", "Assign a task to a user."),
    ("create_ticket", "Open a support ticket."),
    ("close_ticket", "Close a support ticket with a resolution."),
    ("create_invoice", "Create an invoice for a customer."),
    ("refund_payment", "Refund a payment."),
    ("create_subscription", "Create a recurring subscription."),
    ("cancel_subscription", "Cancel a subscription."),
    ("create_coupon", "Create a discount coupon."),
    ("create_product", "Create a catalog product."),
    ("update_inventory", "Adjust inventory levels for a SKU."),
    ("create_webhook", "Register a webhook endpoint."),
    ("rotate_api_key", "Rotate an integration API key."),
    ("create_user", "Create a user account."),
    ("deactivate_user", "Deactivate a user account."),
    ("assign_role", "Assign a role to a user."),
    ("create_team", "Create a team."),
    ("create_folder", "Create a folder in the asset library."),
    ("upload_asset", "Upload an image or file asset."),
    ("tag_asset", "Add tags to an asset."),
    ("create_landing_page", "Create a landing page from a template."),
    ("publish_landing_page", "Publish a landing page."),
    ("create_form", "Create a lead capture form."),
    ("create_survey", "Create a customer survey."),
    ("create_experiment", "Create an A/B experiment."),
    ("get_experiment_results", "Fetch results for an experiment."),
    ("create_alert", "Create a metric alert."),
    ("create_note", "Attach a note to a record."),
    ("search_records", "Full-text search across records."),
    ("translate_text", "Translate text to another language."),
    ("summarize_document", "Summarize a document."),
    ("create_calendar_event", "Create a calendar event."),
    ("send_slack_message", "Post a message to a Slack channel."),
]

_NEAR_MISS_SUFFIXES = ["_draft", "_v2", "_legacy", "_async", "_batch", "_preview"]
_SYNONYMS = {
    "create": ["add", "new", "register", "build"],
    "segment": ["audience", "cohort", "group", "list"],
    "list": ["get_all", "enumerate", "fetch"],
    "get": ["fetch", "read", "lookup"],
    "update": ["modify", "patch", "edit"],
    "delete": ["remove", "archive", "purge"],
    "schema": ["fields", "model", "structure"],
    "email": ["message", "mail"],
    "user": ["customer", "profile", "contact"],
}


def _near_misses(real_names: Iterable[str], rng: random.Random, per_tool: int = 2) -> list[DistractorTool]:
    out: list[DistractorTool] = []
    seen: set[str] = set(real_names)
    for name in list(real_names):
        candidates: list[str] = []
        for suf in _NEAR_MISS_SUFFIXES:
            candidates.append(f"{name}{suf}")
        parts = name.split("_")
        for i, part in enumerate(parts):
            for syn in _SYNONYMS.get(part, []):
                candidates.append("_".join(parts[:i] + [syn] + parts[i + 1 :]))
        rng.shuffle(candidates)
        added = 0
        for cand in candidates:
            if cand in seen or not re.fullmatch(r"[A-Za-z0-9_]+", cand):
                continue
            seen.add(cand)
            pretty = cand.replace("_", " ")
            out.append(
                DistractorTool(
                    name=cand,
                    description=f"{pretty.capitalize()}. Similar to {name} but for a different object type or workflow.",
                )
            )
            added += 1
            if added >= per_tool:
                break
    return out


def build_distractors(config: DistractorConfig, real_tool_names: Iterable[str]) -> list[DistractorTool]:
    real = [n for n in real_tool_names]
    real_set = set(real)
    rng = random.Random(config.seed)
    out: list[DistractorTool] = []
    seen: set[str] = set(real_set)

    def _add(tool: DistractorTool) -> None:
        if tool.name in seen:
            return
        seen.add(tool.name)
        out.append(tool)

    if config.mode in ("manual", "both"):
        for t in config.tools:
            desc = t.description or f"{t.name.replace('_', ' ').capitalize()}."
            _add(DistractorTool(name=t.name, description=desc, input_schema=t.input_schema))
    if config.mode in ("auto", "both"):
        catalog = list(_CATALOG)
        rng.shuffle(catalog)
        for name, desc in catalog[: max(0, config.count)]:
            _add(
                DistractorTool(
                    name=name,
                    description=desc,
                    input_schema={
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "id": {"type": "string"}},
                    },
                )
            )
        if config.near_miss:
            for t in _near_misses(real, rng):
                _add(t)
    return out
