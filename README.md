# WhatsApp Message Notification Router

[![Hackathon](https://img.shields.io/badge/HackerRank-Orchestrate%20Hackathon%202026-2EC866?logo=hackerrank&logoColor=white)](https://github.com/H0ussamCl4p/Orch26Hack-MssgNotifRouter)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/H0ussamCl4p/Orch26Hack-MssgNotifRouter/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made by Choubik Houssam](https://img.shields.io/badge/Made%20by-Choubik%20Houssam-orange)](https://github.com/H0ussamCl4p)

Routes every incoming WhatsApp message to **notify**, **digest**, or **mute** —
personalized per user, with a message type, a short reason, a calibrated
confidence, and a historical evidence citation.

> **Result: #106 / 1983 globally · #1 in Morocco. Score 70.9 / 100.** Solo build, 24 hours.
> Scoring combined the agent's output, the code, and a 30-minute live technical
> interview.

Built for the HackerRank Orchestrate hackathon (Message Notification Router).

---

## The problem

WhatsApp is noisy: family chats, society notices, school updates, business
promotions, image posters, voice notes, and scams all arrive in one stream.
Every message must be routed **notify / digest / mute** with a type, a reason, a
calibrated confidence, and the historical message it was judged against. Inputs
are multimodal — text, image posters/screenshots, and voice notes.

## Pipeline

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

Media extraction and routing are cached on disk, so re-runs are near-instant and
API-free. Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Results

Measured against 30 labelled examples — **directional, not exact** (one message
≈ 3 accuracy points). Full breakdown: [docs/RESULTS.md](docs/RESULTS.md).

| Dimension | Score |
|---|---|
| action | 93% accuracy · macro-F1 0.93 |
| message_type | 87% accuracy · macro-F1 0.72 |
| evidence | exact-match 43% |
| calibration | Brier 0.06 |
| retrieval recall | 86% |
| cost | $0 |

## Three design decisions

**Deterministic evidence selection, not the LLM's pick.** The ground truth cites
a single most textually-similar precedent. Measured against the labels, a Jaccard
token-overlap top-1 matched gold **54%** of the time versus **39%** for the LLM's
own choice (and 36% for metadata rank). So `best_evidence()` selects the citation
deterministically over the retrieved candidate pool. A hallucinated evidence ID
becomes structurally impossible — the choice can only be an ID that was actually
retrieved.

**Metadata filter, not vector search.** With only ~412 historical messages, a
metadata filter (receiving user → same business/group/sender) narrows the pool to
a handful of genuinely comparable precedents — more precise than embedding
similarity at this volume, with no model or index to maintain, and fully
reproducible. Measured retrieval recall of the gold evidence is **86%**, so the
ceiling here is selection, not retrieval.

**Confidence rescaled into an observed band.** Every ground-truth confidence in
the sample falls within `[0.78, 0.91]`, but the router emits higher, less
calibrated values (0.90–0.98). `consistency.py` applies a single monotonic linear
rescale of the batch into that band (with p5/p95 clipping so one outlier can't
compress the rest). Relative ordering is preserved; the absolute values match
observed calibration (Brier 0.06).

## Limitations

Stated plainly, because the numbers are small and the design has real edges:

- **Retrieval recall caps at 86%.** ~14% of the gold evidence is not in the
  candidate pool at all, so no selection method — however clever — can recover it.
- **`spam` and `unknown` have n = 1** in the 30 labels. A single miss sends that
  class's F1 to 0.00, which is what drags macro-F1 to 0.72 against 87% accuracy.
  It's a small-sample artefact, not a systemic failure.
- **The retrieval layer is in-memory pandas.** It's the first thing that breaks at
  scale; ~412 history rows is comfortable, millions would not be.
- **The confidence band is static.** `[0.78, 0.91]` was fit to this sample and
  will drift silently if the underlying distribution changes — there is no
  re-calibration loop.

## Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # add ".[dev]" for pytest + ruff

# System dependency for voice-note decoding (Debian/Ubuntu/Kali):
sudo apt install -y ffmpeg
```

## Configure

```bash
cp .env.example .env
# edit .env — set GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
```

Optional overrides (`MEDIA_VISION_MODEL`, `MEDIA_WHISPER_SIZE`, `ROUTING_MODEL`,
`ROUTING_WORKERS`, `EVIDENCE_TOPK`, `OUTPUT_PATH`) are documented in
[.env.example](.env.example) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run

The dataset is not redistributed ([dataset/README.md](dataset/README.md)); place
it under `dataset/` first. Each stage is cached, so re-runs are near-instant.

```bash
python -m router.media --all      # 1. extract media into the disk cache
python -m router.run              # 2. route all messages -> results/output.csv
python -m router.validate         # 3. check the output contract
python -m router.evaluation       # 4. score against the 30 sample labels
```

Equivalent console scripts (`router-media`, `router-run`, `router-validate`,
`router-evaluation`) are installed with the package. Common tasks are wrapped in
the [Makefile](Makefile) (`make extract route validate eval test`).

## Test

No dataset and no network required — fixtures are built inline.

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest tests/ -v
```

## A note on how this was built

This solution was built with AI assistance (Claude Code) under my direction: I
owned the architecture and the trade-offs; the assistant wrote the
implementation under those constraints. The complete, timestamped development
transcript is preserved in [docs/transcript/](docs/transcript/).

---

<p align="center">
  <strong>HackerRank Orchestrate Hackathon 2026</strong> · v1.0.0<br>
  Made by <a href="https://github.com/H0ussamCl4p">Choubik Houssam</a>
</p>
