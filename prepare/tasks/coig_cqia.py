"""COIG-CQIA —— 中文指令对话(知乎长答、考试解析等)。

**这是 nanochat 没有的东西之一**(Summer = nanochat + 中文 + ReTok),对应它的
SmolTalk 在中文侧的位置。

注意它的答案很长(中位约 1126 token),编码时超过 seq_len 的会被整条丢弃 ——
v4 那次编码丢了 11,006 条。

https://huggingface.co/datasets/m-a-p/COIG-CQIA
"""
from __future__ import annotations

import json

from prepare.tasks._local import files
from prepare.tasks.common import Task


class COIGCQIA(Task):
    name = "COIG_CQIA"
    lang = "zh"

    def conversations(self):
        for path in files("COIG_CQIA"):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:                          # noqa: BLE001
                        continue
                    q = (r.get("instruction") or "").strip()
                    extra = (r.get("input") or "").strip()
                    a = (r.get("output") or "").strip()
                    if not q or not a:
                        continue
                    yield [("user", f"{q}\n{extra}" if extra else q),
                           ("assistant", a)]
