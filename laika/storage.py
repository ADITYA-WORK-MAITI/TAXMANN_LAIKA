"""Saving, loading and exporting conversations."""

import csv
import io
import json
from datetime import datetime
from pathlib import Path

from laika.config import settings

Conversation = list[dict[str, str]]


class ConversationStore:
    """Keeps named conversations in a JSON file."""

    def __init__(self, path: Path = settings.data_dir / "saved_conversations.json"):
        self.path = path

    def load_all(self) -> dict[str, Conversation]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, conversation: Conversation, name: str | None = None) -> str:
        name = name or datetime.now().strftime("Conversation %Y-%m-%d %H:%M:%S")
        conversations = self.load_all()
        conversations[name] = conversation
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(conversations, indent=2, ensure_ascii=False), encoding="utf-8")
        return name


def to_csv(conversation: Conversation) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Question", "Response"])
    for turn in conversation:
        writer.writerow([turn["question"], turn["response"]])
    return buffer.getvalue()
