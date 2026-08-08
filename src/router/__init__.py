"""WhatsApp message notification router.

A deterministic-retrieval + LLM pipeline that routes each incoming WhatsApp
message to notify / digest / mute with a message type, reason, calibrated
confidence, and historical evidence. See docs/ARCHITECTURE.md for the design.
"""

__version__ = "1.0.0"
