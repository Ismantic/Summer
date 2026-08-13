"""MMLU 的 auxiliary_train —— 英文多选题型。

nanochat 的注释说明了它为什么在混比里:这 100K 行抽自 ARC / MC_TEST / OBQA /
RACE,**教的是「看到选项就答字母」这个格式**,不是知识。

https://huggingface.co/datasets/cais/mmlu
"""
from __future__ import annotations

from prepare.tasks._local import parquet_batches
from prepare.tasks.common import Task, render_mc

LETTERS = ["A", "B", "C", "D"]


class MMLUAux(Task):
    name = "MMLU_AuxTrain"
    lang = "en"

    def __init__(self, stop=None, render=render_mc):
        super().__init__(stop=stop)
        self.render = render

    def conversations(self):
        # auxiliary_train 把 question/choices/answer 嵌在一个叫 `train` 的结构体列里
        for batch in parquet_batches("MMLU_AuxTrain", ["train"]):
            for row in batch.column(0).to_pylist():
                if row is None:
                    continue
                ch = row.get("choices") or []
                if not ch or row.get("answer") is None:
                    continue
                letters = LETTERS[:len(ch)]
                yield [("user", self.render(row.get("question", ""), letters, list(ch))),
                       ("assistant", letters[int(row["answer"])])]
