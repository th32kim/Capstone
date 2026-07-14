"""Query constraints — temporal / person / place, parsed and applied POST-fusion (CLAUDE.md §8).

Parsing is deterministic and bounded: it pulls constraint phrases out of the query and returns
predicates over hydrated hits. Temporal phrases ("last Tuesday", "yesterday", "this morning")
map to a wall-clock window using the session start time; person/place constraints require an
entity of the right type to appear on the hit. Applied after RRF so they never distort the
fusion itself.

`now` is passed in (never read from the clock inside pure code) so runs stay deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MORNING = (time(5, 0), time(12, 0))
_AFTERNOON = (time(12, 0), time(17, 0))
_EVENING = (time(17, 0), time(23, 0))


@dataclass
class Constraints:
    time_window: tuple[datetime, datetime] | None = None
    persons: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)

    def active(self) -> bool:
        return bool(self.time_window or self.persons or self.places)


def parse_constraints(query: str, *, now: datetime, entities: list | None = None) -> Constraints:
    q = query.lower()
    c = Constraints()
    today = now.date()

    def day_window(d: date) -> tuple[datetime, datetime]:
        return datetime.combine(d, time.min), datetime.combine(d, time.max)

    if "yesterday" in q:
        c.time_window = day_window(today - timedelta(days=1)); c.raw.append("yesterday")
    elif "today" in q:
        c.time_window = day_window(today); c.raw.append("today")
    for i, wd in enumerate(_WEEKDAYS):
        if wd in q:
            delta = (today.weekday() - i) % 7 or 7  # most recent past occurrence
            d = today - timedelta(days=delta)
            c.time_window = day_window(d); c.raw.append(f"last {wd}")
            break
    for label, (lo, hi) in (("morning", _MORNING), ("afternoon", _AFTERNOON), ("evening", _EVENING)):
        if label in q:
            base = c.time_window[0].date() if c.time_window else today
            c.time_window = (datetime.combine(base, lo), datetime.combine(base, hi))
            c.raw.append(label)
            break

    # person / place from query NER (types passed in), else skip (no fabrication)
    for e in entities or []:
        if getattr(e, "type", "") == "PERSON":
            c.persons.append(e.text.lower())
        elif getattr(e, "type", "") in ("GPE", "LOC"):
            c.places.append(e.text.lower())
    return c


def apply_constraints(hits: list[dict], c: Constraints, *, session_start: datetime | None) -> list[dict]:
    """Filter hydrated hit dicts by the parsed constraints. Segment ts are seconds from session start."""
    if not c.active():
        return hits
    out = []
    for h in hits:
        if c.time_window and session_start is not None:
            seg_dt = session_start + timedelta(seconds=float(h["t_start"]))
            if not (c.time_window[0] <= seg_dt <= c.time_window[1]):
                continue
        ent_text = " ".join(h.get("entities", ())).lower()
        if c.persons and not any(p in ent_text for p in c.persons):
            continue
        if c.places and not any(p in ent_text for p in c.places):
            continue
        out.append(h)
    return out
