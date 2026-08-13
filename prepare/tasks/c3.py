"""C3(CLUE)—— 中文多选阅读理解,11869 条。

**中文侧对应 nanochat 的 MMLU_AuxTrain**:教「看到选项就答字母」这个格式。
起因是实测出来的洞 —— midtrain 里的多选题原本全是英文,于是那个行为只在英文
语境里学会了(中文格式跟随 0.39,英文 0.96)。

取的是正经 **train split**,和 C-Eval 评测集无重叠。

**answer 是答案原文,不是下标** —— 要反查 index。查不到就丢,不猜:猜错会教
模型把正确答案和错的字母绑在一起,而且不会报错。

https://huggingface.co/datasets/clue/clue
"""
from __future__ import annotations

from prepare.tasks._local import parquet_batches
from prepare.tasks.common import Task, render_mc

LETTERS = ["A", "B", "C", "D"]


class C3(Task):
    name = "C3_Train"
    lang = "zh"

    def __init__(self, stop=None, render=render_mc):
        super().__init__(stop=stop)
        self.render = render

    def conversations(self):
        cols = ["context", "question", "choice", "answer"]
        for batch in parquet_batches("C3_Train", cols):
            rows = [batch.column(i).to_pylist() for i in range(4)]
            for ctx, q, ch, ans in zip(*rows):
                if not ch or ans is None or ans not in ch:
                    continue
                ctx = "\n".join(ctx) if isinstance(ctx, (list, tuple)) else (ctx or "")
                ch = list(ch)
                letters = LETTERS[:len(ch)]
                q_text = f"{ctx}\n{q}" if ctx else q
                yield [("user", self.render(q_text, letters, ch)),
                       ("assistant", letters[ch.index(ans)])]
