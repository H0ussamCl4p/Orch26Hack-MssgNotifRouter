# Architecture

Message Notification Router for WhatsApp. This document describes the pipeline
module by module, then records the key **design decisions** and why they were
made. For setup and run commands see [the README](../README.md).

## Pipeline overview

```
messages.csv
     │
     ▼
  ingest ─────────────► load_all(): 12 input CSVs as named DataFrames
     │
     ▼   for each message_id
  ┌──────────────┬─────────────────┬──────────────────┐
  │  context     │  candidates     │  media (cache)   │
  │  build_      │  get_candidates │  load_extracted  │
  │  context()   │  best_evidence  │  (OCR/transcript)│
  └──────┬───────┴────────┬────────┴─────────┬────────┘
         └────────────┬────┴──────────────────┘
                      ▼
                 routing.route_one  ──►  Gemma 4 (action / type / reason / confidence)
                      │                   + deterministic evidence override
                      ▼
                 consistency.reconcile  ──►  confidence calibrated into [0.78, 0.91]
                      ▼
                 output.csv  (validated by validate.py)
```

Each expensive stage (media extraction, routing) is cached on disk, so re-runs
are near-instant and require no API calls.

## Modules

- **config.py** — single source of truth: paths (resolved relative to the repo
  root, no absolute paths), the allowed enums, the 6-column output contract, and
  `.env` loading. Secrets are only ever read from the environment.
- **ingest.py** — `load_all()` returns the 12 input CSVs as named DataFrames.
  `output.csv` is the write target, not an input, so it is not loaded.
- **context.py** — assembles a compact, JSON-serializable dossier per message:
  the user's quiet hours and engagement, the group/business relationship, a
  notification-load block, and a spoofing signal. Two sharp edges handled:
  - `in_dnd()` correctly treats do-not-disturb windows that wrap past midnight
    (`22:00-07:00`) as `[start,24:00) ∪ [00:00,end]` — a naive `start<=t<=end`
    would always be false (all 14 windows in the data wrap midnight).
  - `domain_mismatch` flags impersonation: sender domain ≠ official domain, or a
    young sender domain (<90d) on an old account (≥365d).
- **candidates.py** — deterministic evidence retrieval:
  - `get_candidates()` filters the receiving user's own history to the same
    counterparty (business/group/sender), broadening to the same conversation
    type only when a real relationship exists but is thin. Enriches each
    candidate with the user's past reaction (opened/replied/dismissed/muted/
    reported). Returns `first_contact` when there is no counterparty history.
  - `best_evidence()` selects the evidence IDs (see decisions below).
  - `diagnose()` reports retrieval health over all 110 messages.
- **media.py** — hybrid extraction with an incremental disk cache. Images →
  Gemma 4 vision (OCR text + visual semantics: document type, brand, QR/link/
  amount, urgency). Audio → faster-whisper locally. Extracted content is wrapped
  as untrusted data (see decisions). A successfully-extracted media is never
  re-processed; errors are isolated per file and retried on the next run.
- **routing.py** — `route_one()` assembles context + candidates + media into one
  prompt, calls the routing model, then hardens the output. Decisions are cached
  per message, version-guarded by `PROMPT_VERSION`.
- **consistency.py** — batch reconciliation. Currently calibrates confidence into
  the observed ground-truth band. A cheap, deterministic post-process (no LLM).
- **run.py** — pipeline entry point. Routes all messages concurrently
  (`ROUTING_WORKERS`), reconciles, writes `output.csv`. Per-message isolation:
  a failing message falls back to a placeholder and never sinks the batch.
- **validate.py** — enforces the output contract: 110 rows, exact columns/order,
  enum membership, confidence in [0,1], non-empty evidence, full id coverage.
- **evaluation.py** — scores the pipeline against the 30 labelled examples:
  accuracy + macro-F1 + confusion for action and message_type, evidence
  exact/precision/recall (with `none` as a first-class value), calibration
  (Brier + per-bucket accuracy), and an action×type coherence check.

## Design decisions

### No vector search — metadata filter instead
With only ~412 historical messages, a metadata filter (receiving user, then same
business/group/sender) narrows the pool to a handful of genuinely comparable
precedents. At this volume a deterministic filter is **more precise** than blind
vector similarity, needs no embedding model or index, and is fully reproducible.
Measured retrieval recall of the gold evidence is 86% — the right precedent is
almost always in the pool.

### Evidence chosen deterministically, not by the LLM
The ground truth cites a single, most **textually similar** precedent. We
measured candidate-selection strategies against the 30 labels:

| selector | top-1 == gold |
|---|---|
| metadata rank #1 | 36% |
| LLM's own pick | 39% |
| **lexical (token-overlap) top-1** | **54%** |

So `best_evidence()` picks the lexical top-1 over the candidate pool
(Jaccard token overlap, |A ∩ B| / |A ∪ B| — not vector search), overriding whatever the LLM returns.
Benefits: **reproducible**, and a hallucinated evidence ID is **impossible** (the
choice is drawn only from retrieved candidates). The LLM still produces
action/type/reason/confidence; evidence is decoupled from it.

### Precision/recall trade-off on evidence (exposed knob)
The gold cites one id in 25/28 cases, so citing one maximizes precision. Moving
from the LLM's multi-id output to a single lexical pick changed the numbers as
follows:

| | before (LLM, ~2 ids) | after (lexical top-1) |
|---|---|---|
| exact-match | 30% | **43%** |
| id precision | 44% | **52%** |
| id recall | 68% | 48% |
| false-positive ids | 27 | **14** |

We deliberately favour precision — a wrong evidence id is worse than an honest
`none`. The trade-off is exposed as `EVIDENCE_TOPK` (default 1); raising it
recovers recall at the cost of precision (`best_evidence(..., k=2)` had 71%
top-2 coverage).

### Confidence calibrated into [0.78, 0.91]
Every ground-truth confidence in `sample_messages.csv` falls within
[0.78, 0.91]. The router emits higher, less-calibrated values (0.9–0.98), so
`consistency.py` applies a **monotonic** linear rescale of the batch into that
band (with p5/p95 clipping so one outlier can't compress the rest). Relative
ordering — which messages the model was more sure about — is preserved, and the
absolute values match observed calibration. Verified: Brier 0.06, and accuracy
rises monotonically across confidence buckets (86→92→100%).

### Gemma 4 for vision and routing (free) — after RECITATION on gemini-*-flash
`gemini-3.6-flash` extracted 19/20 images well but blocked one benign poster with
`finish_reason=RECITATION` (it refuses to reproduce text it recognizes) and has a
low free-tier daily cap (20 req/day). `gemma-4-31b-it` (an open model served on
the same API) accepts images, has **no RECITATION filter**, and a much larger
free quota — so it became the default for both vision and routing. It supports
neither a system role nor JSON mode, so instructions are folded into the user
turn and JSON is parsed defensively (tolerant of a ```json fence). Cost: $0.

### Version-guarded caching
Media extraction and routing decisions are cached per item on disk. The routing
cache entries carry a `PROMPT_VERSION` and model tag; editing the prompt or
switching model bumps the version and transparently invalidates stale entries,
while unchanged runs re-spend no quota. Evidence is recomputed on every run
(deterministic, API-free), so improving `best_evidence()` needs no re-routing.

### `_untrusted` encapsulation against prompt injection
The dataset contains prompt-injection attempts embedded in media (e.g. a fake
bank screenshot instructing the reader). We do **not** strip them — the injection
itself is a scam signal — but encapsulate all extracted content under an
`_untrusted` wrapper with a disclaimer and an `injection_suspected` flag, and the
vision/routing prompts are hardened to treat media strictly as data, never as
instructions. `to_prompt_block()` renders media inside explicit
`<untrusted_media_content>` delimiters for the router.

## Reproducibility & hygiene

- No hardcoded secrets — keys come from the environment (`GEMINI_API_KEY`).
- Deterministic: `temperature=0` for all LLM calls, greedy transcription, fixed
  filtering/sorting. No unseeded randomness.
- Pinned dependencies (`requirements.txt`, `requirements-media.txt`).
- No absolute paths — everything resolves from the repo root.
- Only `dataset/` inputs are used; no organizer-only files.
