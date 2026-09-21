"""TypeSafe "System One" judge (Jev).

An alternative to the LLM-as-judge. Instead of asking a *generative* model for a
1-5 rubric score wrapped in prose, this judge asks TypeSafe's System One model
(**Jev**) *typed* questions and reads back *typed* answers with probabilities:

* every rubric criterion becomes a :class:`Score` question (a degree along an
  ordered 1-5 dimension, anchored by the rubric's own anchors), and
* the overall "did the tool perform correctly?" verdict becomes a :class:`Noul`
  (the probability of "yes").

Because Jev returns probabilities rather than generated text, its judgments are
deterministic to read, calibratable and cheap to reason about. See the TypeSafe
skill / docs at https://docs.typesafe.ai for the primitives.

The real client comes from the optional ``typesafe-sdk`` package::

    pip install "pm-evals[typesafe]"     # and set TYPESAFE_API_KEY

A small in-process mock client (selected with the ``jev:mock`` model string)
lets the platform and its test-suite exercise the whole path offline, with no
key and no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import RubricItem

# Rubric criteria are graded on the same 1..5 scale the LLM judge uses, so the
# report's judge-breakdown table renders identically for either judge.
N_LEVELS = 5


# ---------------------------------------------------------------------------
# Question representation (our own, converted to the SDK's classes on the wire)
# ---------------------------------------------------------------------------


@dataclass
class Score:
    """Position along an ordered list of ``criteria`` levels (low -> high)."""

    instructions: str
    criteria: list[str]


@dataclass
class Noul:
    """Binary judgment whose probability of "yes" is itself the answer."""

    instructions: str
    criteria: Optional[str] = None


@dataclass
class Choice:
    """One option from an unordered ``criteria`` map (not used by the judge yet)."""

    instructions: str
    criteria: dict[str, Any]


# ---------------------------------------------------------------------------
# Score -> 1-5 normalisation
# ---------------------------------------------------------------------------


def _level_weights(probabilities: Any, n_levels: int) -> dict[int, float]:
    """Normalise a System One probability distribution to ``{1-based level: p}``.

    Accepts either a list (index 0 = level 1) or a dict keyed by level number
    or by level label position. Unknown shapes yield an empty mapping."""
    out: dict[int, float] = {}
    if isinstance(probabilities, dict):
        for k, v in probabilities.items():
            try:
                lvl = int(k)
            except (TypeError, ValueError):
                continue
            if 1 <= lvl <= n_levels:
                try:
                    out[lvl] = float(v)
                except (TypeError, ValueError):
                    continue
    elif isinstance(probabilities, (list, tuple)):
        for i, v in enumerate(probabilities[:n_levels]):
            try:
                out[i + 1] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def position_1_based(scored: Any, n_levels: int = N_LEVELS) -> float:
    """Best estimate of the 1-based level a ``Score`` result points at.

    Prefers the probability-weighted mean over the levels ("Score = a
    probability-weighted position on ordered levels"); falls back to the raw
    ``score`` field, detecting whether it is 0- or 1-based, then clamps to
    ``[1, n_levels]``."""
    weights = _level_weights(getattr(scored, "probabilities", None), n_levels)
    total = sum(weights.values())
    if total > 0:
        pos = sum(lvl * w for lvl, w in weights.items()) / total
        return max(1.0, min(float(n_levels), pos))
    raw = getattr(scored, "score", None)
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return 1.0
    # A value in [0, n-1) that is below 1 can only be 0-based; shift it up.
    if 0.0 <= s < 1.0:
        s += 1.0
    return max(1.0, min(float(n_levels), s))


def position_to_1_5(pos: float, n_levels: int = N_LEVELS) -> float:
    """Map a 1-based level position onto the shared 1-5 rubric scale."""
    n = max(2, int(n_levels))
    pos = max(1.0, min(float(n), float(pos)))
    return 1.0 + (pos - 1.0) * 4.0 / (n - 1)


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


class TypeSafeJevJudge:
    """Grade a run with TypeSafe's System One model (Jev).

    Exposes :meth:`grade`, which the ``task_completion`` / ``trajectory_quality``
    checks call instead of an LLM's ``json_completion``. ``kind == "system_one"``
    lets those checks detect this judge and take the typed path.
    """

    kind = "system_one"
    name = "typesafe"

    def __init__(self, model: str = "jev", api_key: Optional[str] = None, client: Any = None):
        self.model = model
        self._api_key = api_key
        self._client = client  # an injected client (e.g. the mock) is reused as-is

    # -- client lifecycle ---------------------------------------------------
    def _new_client(self) -> Any:
        try:
            from typesafe_sdk import AsyncTypeSafeClient
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "The TypeSafe SDK is not installed. Install it with "
                '`pip install "pm-evals[typesafe]"` and set TYPESAFE_API_KEY, '
                "or use the offline 'jev:mock' judge."
            ) from exc
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return AsyncTypeSafeClient(**kwargs)

    async def aclose(self) -> None:  # pragma: no cover - nothing persistent to close
        return None

    # -- question building --------------------------------------------------
    @staticmethod
    def _score_levels(item: RubricItem) -> list[str]:
        """Five ascending level labels for a criterion, using its anchors."""
        anchors = item.anchors or {}
        labels: list[str] = []
        for lvl in range(1, N_LEVELS + 1):
            text = anchors.get(str(lvl))
            if not text and lvl == 2 and anchors.get("1"):
                text = "slightly better than: " + anchors["1"]
            if not text and lvl == 4 and anchors.get("5"):
                text = "slightly worse than: " + anchors["5"]
            labels.append(f"{lvl} - {text}" if text else str(lvl))
        return labels

    def _build_questions(self, rubric: list[RubricItem], overall_instructions: str) -> tuple[dict[str, Any], list[str]]:
        questions: dict[str, Any] = {}
        keys: list[str] = []
        for i, item in enumerate(rubric):
            key = f"crit{i}"
            keys.append(key)
            instr = item.description or item.criterion.replace("_", " ")
            questions[key] = Score(
                instructions=f"Grade the run on: {instr}. Score the evidence in the trajectory, not the agent's tone.",
                criteria=self._score_levels(item),
            )
        questions["overall_correct"] = Noul(instructions=overall_instructions)
        return questions, keys

    @staticmethod
    def _to_sdk_questions(questions: dict[str, Any]) -> dict[str, Any]:
        from typesafe_sdk import Choice as SChoice  # pragma: no cover - optional dep
        from typesafe_sdk import Noul as SNoul
        from typesafe_sdk import Score as SScore

        out: dict[str, Any] = {}
        for key, q in questions.items():
            if isinstance(q, Score):
                out[key] = SScore(instructions=q.instructions, criteria=q.criteria)
            elif isinstance(q, Noul):
                out[key] = SNoul(instructions=q.instructions) if q.criteria is None else SNoul(instructions=q.instructions, criteria=q.criteria)
            elif isinstance(q, Choice):
                out[key] = SChoice(instructions=q.instructions, criteria=q.criteria)
        return out

    # -- grading ------------------------------------------------------------
    async def grade(self, state: dict[str, Any], rubric: list[RubricItem], overall_instructions: str) -> dict[str, Any]:
        """Return a rubric grading shaped like the LLM judge's output, plus the
        System One correctness verdict.

        ``{"criteria": [{criterion, score(1-5), reason, position, confidence}...],
           "score": overall 1-5, "correct": bool, "correct_probability": float,
           "reason": str}``.
        """
        if not rubric:
            rubric = [RubricItem(criterion="task_performed_correctly", weight=1.0)]
        questions, keys = self._build_questions(rubric, overall_instructions)

        if self._client is not None:
            resp = await self._client.system_one(state=state, questions=questions)
        else:
            client_cm = self._new_client()
            async with client_cm as client:  # pragma: no cover - real SDK path
                resp = await client.system_one(state=state, questions=self._to_sdk_questions(questions))

        return self._parse(resp, rubric, keys)

    def _parse(self, resp: Any, rubric: list[RubricItem], keys: list[str]) -> dict[str, Any]:
        scores = getattr(resp, "scores", {}) or {}
        nouls = getattr(resp, "nouls", {}) or {}

        criteria_out: list[dict[str, Any]] = []
        for item, key in zip(rubric, keys):
            scored = scores[key] if key in scores else None
            pos = position_1_based(scored, N_LEVELS) if scored is not None else 1.0
            score_1_5 = position_to_1_5(pos, N_LEVELS)
            confidence = getattr(scored, "confidence", None) if scored is not None else None
            reason = f"System One score {pos:.2f}/5"
            if confidence is not None and not (isinstance(confidence, float) and math.isnan(confidence)):
                reason += f" (confidence {float(confidence):.2f})"
            criteria_out.append(
                {
                    "criterion": item.criterion,
                    "score": round(score_1_5, 2),
                    "reason": reason,
                    "position": round(pos, 3),
                    "confidence": (round(float(confidence), 3) if confidence is not None else None),
                }
            )

        noul = nouls["overall_correct"] if "overall_correct" in nouls else None
        try:
            correct_p = float(getattr(noul, "noul", 0.0) or 0.0)
        except (TypeError, ValueError):
            correct_p = 0.0
        correct_p = max(0.0, min(1.0, correct_p))
        overall_1_5 = 1.0 + 4.0 * correct_p
        verdict = "correct" if correct_p >= 0.5 else "incorrect"
        return {
            "criteria": criteria_out,
            "score": round(overall_1_5, 2),
            "correct": correct_p >= 0.5,
            "correct_probability": round(correct_p, 3),
            "reason": f"System One (Jev) judged the run {verdict}: P(performed correctly) = {correct_p:.2f}.",
        }


# ---------------------------------------------------------------------------
# Offline mock client (model string ``jev:mock``)
# ---------------------------------------------------------------------------


@dataclass
class _Scored:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


@dataclass
class _Nouled:
    noul: float


@dataclass
class _Choiced:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class _Response:
    scores: dict[str, Any] = field(default_factory=dict)
    nouls: dict[str, Any] = field(default_factory=dict)
    choices: dict[str, Any] = field(default_factory=dict)


def _peaked(n_levels: int, pos: float) -> dict[str, float]:
    """A distribution whose probability-weighted mean is ``pos`` (1-based)."""
    pos = max(1.0, min(float(n_levels), float(pos)))
    lo = int(math.floor(pos))
    hi = min(n_levels, lo + 1)
    frac = pos - lo
    dist = {str(i): 0.0 for i in range(1, n_levels + 1)}
    if lo == hi:
        dist[str(lo)] = 1.0
    else:
        dist[str(lo)] = round(1.0 - frac, 4)
        dist[str(hi)] = round(frac, 4)
    return dist


class MockSystemOneClient:
    """Deterministic, offline stand-in for the System One client.

    ``behaviour="good"`` returns high scores and a strong "yes" on correctness;
    ``behaviour="bad"`` returns low scores and a strong "no". Used by the
    ``jev:mock`` judge model and by the test-suite so the typed-judge path runs
    with no SDK, key or network."""

    def __init__(self, behaviour: str = "good", level: Optional[float] = None, correct_p: Optional[float] = None):
        self.behaviour = behaviour
        self.level = level if level is not None else (4.5 if behaviour == "good" else 1.5)
        self.correct_p = correct_p if correct_p is not None else (0.92 if behaviour == "good" else 0.08)

    async def system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> _Response:
        resp = _Response()
        for key, q in questions.items():
            if isinstance(q, Score):
                n = len(q.criteria) or N_LEVELS
                pos = max(1.0, min(float(n), self.level))
                resp.scores[key] = _Scored(
                    score=pos,
                    legend={str(i + 1): c for i, c in enumerate(q.criteria)},
                    probabilities=_peaked(n, pos),
                    confidence=0.85 if self.behaviour == "good" else 0.6,
                )
            elif isinstance(q, Noul):
                resp.nouls[key] = _Nouled(noul=self.correct_p)
            elif isinstance(q, Choice):
                first = next(iter(q.criteria), "")
                resp.choices[key] = _Choiced(
                    choice=first,
                    probabilities={k: (1.0 if k == first else 0.0) for k in q.criteria},
                    confidence=0.9,
                )
        return resp
