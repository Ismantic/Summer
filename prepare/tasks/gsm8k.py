"""GSM8K —— 小学数学应用题,教分步推理。

nanochat 在 midtrain 里放 1 个 epoch、SFT 里也放 1 个(新版单阶段是 4 个)。

https://huggingface.co/datasets/openai/gsm8k
"""
from __future__ import annotations

from prepare.tasks._local import parquet_batches
from prepare.tasks.common import Task


class GSM8K(Task):
    name = "GSM8K_Train"
    lang = "en"

    def conversations(self):
        for batch in parquet_batches("GSM8K_Train", ["question", "answer"]):
            for q, a in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if q and a:
                    yield [("user", q), ("assistant", a)]
