# Dataset

The dataset for this challenge is the property of **HackerRank** and is **not
redistributed** in this repository. Everything under `dataset/` except this file
is gitignored.

## What goes here

The pipeline reads the following files from this directory:

```
dataset/
├── messages.csv                  # the 110 messages to route
├── sample_messages.csv           # 30 solved examples (used by evaluation.py)
├── users.csv
├── groups.csv
├── group_members.csv
├── business_accounts.csv
├── user_business_history.csv
├── message_history.csv
├── message_events.csv
├── images.csv                    # image_id -> media file path
├── voice_notes.csv               # voice_note_id -> media file path
├── daily_notification_summary.csv
└── media/
    ├── images/
    └── audio/
```

## How to obtain it

The data was provided to participants of the **HackerRank Orchestrate** hackathon
(Message Notification Router challenge). If you have access to that challenge,
place the files above into this directory.

## Running without the dataset

The dataset is only needed to *run* the pipeline (`router.media`, `router.run`,
`router.evaluation`). The test suite (`pytest`) builds its fixtures inline and
needs no dataset, and `router.validate` can check any output CSV standalone —
so `pip install -e . && pytest` works on a fresh clone with no data present.
