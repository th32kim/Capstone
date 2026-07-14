# CORPUS — the labelling protocol

Everything downstream of Tier-1 can be *built* without labels. Nothing downstream of Tier-1 can be
*evaluated* without them. The corpus is the real critical path.

## Layout

```
data/
  manifest.yaml          # one entry per clip — tracked in git
  corpus/                # the .mp4s themselves — gitignored (big, and possibly consent-sensitive)
  labels/<clip_id>.csv   # ground truth — tracked in git
  queries/queries.yaml   # the labelled query set, for retrieval eval
```

`data/labels/<clip_id>.csv`:
```csv
t_start,t_end,category,labeller,mode
14.5,41.0,conversation,andy,dense
88.2,96.0,reading_screen,andy,dense
```

## What counts as an "interesting" event

Agree this **before** anyone labels, and write it down here, or two labellers will produce two
different ground truths and the F1 number will be meaningless. A serviceable starting definition —
*a span you would want to be able to search for later* — with categories:

- `conversation` — someone is speaking to or near the wearer with content
- `reading_screen` / `reading_document` — the wearer is attending to text
- `demonstration` — someone is showing/doing something the wearer is watching
- `whiteboard` / `presentation`
- `notable_object` / `notable_place` — the wearer stops and looks at something specific
- `transaction` — an interaction with a person or a machine

Explicitly **not** interesting: walking, waiting, idle scanning, silence, empty corridors, a laptop
open on a desk with nobody speaking.

## The two labelling modes — this distinction decides whether your numbers mean anything

| Mode | How | Fast? | Can measure **precision**? | Can measure **recall**? |
|---|---|---|---|---|
| `assisted` | UI pre-seeds Tier-1's proposals; the human accepts / rejects / adjusts | yes | **yes** | **NO** |
| `dense` | the human labels every event from scratch, blind to the detector | no | yes | **yes** |

The reason is simple and it catches everyone: **in assisted mode you never see the events the
detector never proposed.** Recall computed from assisted labels is not a low estimate — it is a
meaningless one, and it will be *flatteringly* wrong.

**Requirement: at least one full clip labelled `dense`, held out, and used as the recall set.**
`eval/gating.py` refuses to compute recall from `assisted` labels and raises instead. That refusal
is a feature — leave it in.

## Inter-labeller agreement

Two people label at least one clip independently. Report **Cohen's κ** on the per-window binary
interesting/not-interesting series. **If κ < 0.60 the labels are not trustworthy and no F1 computed
against them should be reported**, no matter how good it looks. Fix the definition above, then
re-label.

## Consent

POV footage of other people. Get consent from anyone recorded, note it in `manifest.yaml`, and keep
the corpus out of git. This is a lifelogging project whose entire privacy argument is that raw media
never leaves the device — do not undercut it in the repo.

## Size

Enough to sweep thresholds without overfitting them. As a starting target:
**≥ 6 clips, ≥ 60 minutes total**, spanning: a conversation, reading a screen/signs, faces entering
and leaving, walking (the hard negative — this is what will generate false positives), a static quiet
scene, and a noisy environment. Hold out at least one clip entirely; sweep on the rest.

If you sweep thresholds on the same clips you report F1 on, the F1 is not a result. It is a
description of the sweep.
