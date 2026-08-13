"""Task 的基类 —— 一个 Task 就是一个「对话数据集」。

对齐 nanochat 的 `tasks/common.py`。每个数据集一个模块、接口统一,混比就是一个
**任务实例的列表**(同一个任务传两次 = 两个 epoch)。

## 和 nanochat 的一处偏离:流式,不可索引

它的 Task 是**可索引**的(`get_example(index)`,底层 `load_dataset` 随机访问),
我们是**流式**的(`__iter__`)。理由:`data/source.py` 是数据源的唯一真相来源,
数据由 `data/download.py` 落到本地 parquet;改用 `load_dataset` 会分叉出第二个
来源,以后「这份数据从哪来」就有两个答案了。

代价是 nanochat 的 `Task(start, stop, step)` 那种任意切片做不到。我们只保留
`stop=N`(取前 N 条)—— 它的混比里也只用到这一种(`SmolTalk(split="train",
stop=10_000)`)。

## 掩码在哪

Task 只产出 `[(role, content)]` 的轮次,**不做分词、不建掩码**。掩码由
`prepare/chat.py` 按 role 建(assistant 那段 mask=1),因为它要看 token 边界。
"""
from __future__ import annotations

from typing import Iterator


class Task:
    """流式任务。子类实现 `conversations()`,产出 `[(role, content)]`。"""

    #: 供人读的名字,进 mix.json 的账
    name: str = "Task"
    #: "en" | "zh" —— 只用于报账,不影响训练
    lang: str = "en"

    def __init__(self, stop: int | None = None):
        assert stop is None or stop > 0, f"stop 必须是正数,收到 {stop}"
        self.stop = stop

    def conversations(self) -> Iterator[list[tuple[str, str]]]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[list[tuple[str, str]]]:
        for i, conv in enumerate(self.conversations()):
            if self.stop is not None and i >= self.stop:
                return
            yield conv

    def __repr__(self) -> str:
        return f"{self.name}({'all' if self.stop is None else self.stop})"


class TaskMixture:
    """任务的混合。**传同一个任务多次就是多个 epoch** —— 和 nanochat 一样。

    产出顺序是**按任务依次**,不交错。行级别的打散在 `chat.py` 最后统一做
    (它对打包后的行做全局 shuffle),所以这里不需要再洗一遍。
    """

    def __init__(self, tasks: list[Task]):
        assert tasks, "混比不能是空的"
        self.tasks = tasks

    def __iter__(self):
        for t in self.tasks:
            for conv in t:
                yield t, conv

    def summary(self) -> str:
        from collections import Counter
        c = Counter(repr(t) for t in self.tasks)
        return "  ".join(f"{k}×{v}" if v > 1 else k for k, v in c.items())


def render_mc(question: str, letters: list[str], choices: list[str]) -> str:
    """逐字对齐 nanochat 的 `tasks/common.py:render_mc`。**不要「改进」它** ——
    换了格式就不能和它的数比,训练和评测也会对不上。

    字母放在选项**之后**(`={字母}`)是它有意的:让 prompt 里的字母和 assistant
    要输出的字母是同一个 token。
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query


def render_mc_ours(question: str, letters: list[str], choices: list[str]) -> str:
    """**我们早期自己用的渲染**,`{问题}\\n{A. 选项}`。

    留着是因为 v2~v5 的 midtrain 都用它训的,评测要能按同一套格式考
    (`prepare/mc_eval.py --render ours`)。**新数据不要再用它** —— 对齐基准是
    nanochat,`render_mc` 才是目标。
    """
    opts = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    return f"{question}\n{opts}"
