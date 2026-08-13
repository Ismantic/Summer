"""ARC(AI2 Reasoning Challenge)的 **train split** —— 科学常识多选。

nanochat 只在 **SFT** 阶段放它(ARC-Easy 2.3K + ARC-Challenge 1.1K),midtrain
里没有。我们原来两个阶段都没有 —— 这是 C 组里「SFT 数据」那一条的主要缺口。

**读数时要记住这件事**:它的 ARC 成绩有相当一部分是**同分布的格式训练**,
因为 train split 和评测用的 test split 同源同格式。所以拿 ARC 比高低时,
要把「格式会不会」和「知识有没有」分开看(`prepare/mc_eval.py` 的
`自由首选是字母` 就是为此)。**train 和 test 不重叠**,不是泄漏。

https://huggingface.co/datasets/allenai/ai2_arc
"""
from __future__ import annotations

from prepare.tasks._local import parquet_batches
from prepare.tasks.common import Task, render_mc


class ARC(Task):
    lang = "en"

    def __init__(self, subset: str = "ARC-Easy", stop=None, render=render_mc):
        super().__init__(stop=stop)
        assert subset in ("ARC-Easy", "ARC-Challenge"), f"未知 subset {subset}"
        self.subset = subset
        self.name = f"ARC_{'Easy' if subset == 'ARC-Easy' else 'Challenge'}_Train"
        self.render = render

    def conversations(self):
        for batch in parquet_batches(self.name, ["question", "choices", "answerKey"]):
            qs, chs, keys = (batch.column(i).to_pylist() for i in range(3))
            for q, ch, key in zip(qs, chs, keys):
                if not ch or not key:
                    continue
                letters = list(ch.get("label") or [])
                texts = list(ch.get("text") or [])
                # 少数行的 answerKey 是数字("1")而不是字母 —— 丢掉,不猜
                if key not in letters or len(letters) != len(texts):
                    continue
                yield [("user", self.render(q, letters, texts)),
                       ("assistant", key)]
