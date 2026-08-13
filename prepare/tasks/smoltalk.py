"""SmolTalk —— 英文通用对话。nanochat 两个阶段都用它当主体。

https://huggingface.co/datasets/HuggingFaceTB/smoltalk
"""
from __future__ import annotations

from prepare.tasks._local import parquet_batches
from prepare.tasks.common import Task

ROLES = ("system", "user", "assistant")


class SmolTalk(Task):
    name = "SmolTalk"
    lang = "en"

    def conversations(self):
        for batch in parquet_batches("SmolTalk", ["messages"]):
            for row in batch.column(0).to_pylist():
                if not row:
                    continue
                turns = [(m.get("role", ""), m.get("content", "")) for m in row]
                turns = [(r, c) for r, c in turns if r in ROLES and c]
                if turns:
                    yield turns
