"""GSM8K —— 小学数学应用题,教分步推理。

nanochat 在 midtrain 里放 1 个 epoch、SFT 里也放 1 个(新版单阶段是 4 个)。

https://huggingface.co/datasets/openai/gsm8k
"""
from __future__ import annotations

import re

from prepare.tasks._local import parquet_batches
from prepare.tasks.common import Task

#: GSM8K 的答案里嵌着计算器标注 `<<48/2=24>>`,那是数据集的内部标记,不是自然文本。
#:
#: **nanochat 把它解析成 python 工具调用**(`tasks/gsm8k.py`:按 `(<<[^>]+>>)`
#: 切开,`expr` 进 `<|python_start|>`、`result` 进 `<|output_start|>`),
#: 所以它的训练数据里这些标记**从不以字面出现**。
#:
#: 我们做不了工具调用(词表满了,加不进那四个 token),所以**直接删掉标记**。
#: 删了正好通顺:`48/2 = <<48/2=24>>24 clips` → `48/2 = 24 clips` ——
#: `>>` 后面那个数本来就是结果。
#:
#: 实测 98.6% 的答案含这个标记,不删就是在教模型吐数据集的内部标注。
CALC = re.compile(r"<<[^>]*>>")


class GSM8K(Task):
    name = "GSM8K_Train"
    lang = "en"

    def conversations(self):
        for batch in parquet_batches("GSM8K_Train", ["question", "answer"]):
            for q, a in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if q and a:
                    yield [("user", q), ("assistant", CALC.sub("", a))]
