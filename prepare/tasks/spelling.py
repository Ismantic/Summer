"""拼写任务 —— 逐字母拆词,教字符级的组成感。

对齐 nanochat 的 `tasks/spellingbee.py`。它在 midtrain 里放
SimpleSpelling 200K + SpellingBee 80K,SFT 里各 300。

## 为什么一个 piece 词表的模型需要这个

分词器把词切成 piece,模型看不见字母。「strawberry 里有几个 r」**不是知识问题,
是表示问题** —— 除非训练里明确教过拆字母,否则模型没有途径知道一个 token 内部
长什么样。ReTok 换了词表之后只会更严重:切分边界和 Qwen3 原生的不一样,预训练里
学到的任何字母感都失效了。

## 两处有意的偏离

**1. 不做 Python 工具调用那一半。** nanochat 的 SpellingBee 答案是两段:先手工
数一遍,再用 `<|python_start|>'word'.count('x')<|python_end|>` +
`<|output_start|>N<|output_end|>` 验算一次。**我们的词表满了(81903),加不了这
四个 token**,加了会让所有已发布模型失配。所以只保留手工那一段,连同它的
`{序号}:{字母} hit! count=N` 格式和结尾的 `This gives us N.`。
**代价是学不到工具使用** —— 那是 nanochat 那条线的一个真实能力,我们没有。

**2. 只做英文。** 两个任务都基于一份 370K 的英文词表。汉字不是字母拼的,
笔画/拼音是另一回事 —— **不做类比,免得编一个 nanochat 没有的任务出来。**
"""
from __future__ import annotations

import random

from prepare.tasks._local import files
from prepare.tasks.common import Task

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# **逐字抄 nanochat 的模板表**(`tasks/spellingbee.py:54`)。用来做数据增广 ——
# 同一个问题的多种问法,免得模型只认一种句式。
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
]

_CACHE: list[str] | None = None


def _words() -> list[str]:
    """370K 英文词表。**不联网** —— 由 data/ 那一层下好放本地。"""
    global _CACHE
    if _CACHE is None:
        out: list[str] = []
        for path in files("EnglishWords"):
            with open(path, encoding="utf-8", errors="replace") as f:
                out.extend(w for w in (line.strip() for line in f) if w.isalpha())
        if not out:
            raise ValueError("EnglishWords 读出来是空的")
        _CACHE = out
    return _CACHE


class SimpleSpelling(Task):
    """`Spell the word: apple` → `apple:a,p,p,l,e`。逐字对齐 nanochat。"""

    name = "SimpleSpelling"
    lang = "en"

    def __init__(self, size: int = 200000, seed: int = 42):
        super().__init__(stop=size)
        self.size, self.seed = size, seed

    def conversations(self):
        words = list(_words())
        random.Random(self.seed).shuffle(words)    # 和 SpellingBee 用不同的词序
        for i in range(self.size):
            w = words[i % len(words)]
            yield [("user", f"Spell the word: {w}"),
                   ("assistant", f"{w}:{','.join(w)}")]


class SpellingBee(Task):
    """`How many r in strawberry?` → 手工逐字母数一遍。

    答案带**逐字母的过程**而不是直接给数字:直接给数字等于要求模型一步之内做完
    计数,那是它做不到的;写出过程才让计数变成可以逐 token 完成的事。
    """

    name = "SpellingBee"
    lang = "en"

    def __init__(self, size: int = 80000, seed: int = 0):
        super().__init__(stop=size)
        self.size, self.seed = size, seed

    def conversations(self):
        words = _words()
        for i in range(self.size):
            rng = random.Random(self.seed + i)
            w = rng.choice(words)
            # **90% 问词里真有的字母,10% 随机** —— 对齐 nanochat。全随机的话
            # 答案会有一大半是 0,模型学到的是「猜 0」。
            letter = rng.choice(w) if rng.random() < 0.9 else rng.choice(LETTERS)
            count = w.count(letter)

            tmpl = rng.choice(USER_MSG_TEMPLATES)
            if rng.random() < 0.3:          # 30% 概率整句小写(懒得按 shift)
                tmpl = tmpl.lower()
            q = rng.choice(["", "'", '"'])
            wq = rng.choice(["", "'", '"'])
            user = tmpl.format(letter=f"{q}{letter}{q}", word=f"{wq}{w}{wq}")
            if rng.random() < 0.5:          # 一半的人不打问号
                user += "?"

            text = (f"We are asked to find the number '{letter}' in the word "
                    f"'{w}'. Let me try a manual approach first.\n\n"
                    f"First spell the word out:\n{w}:{','.join(w)}\n\n"
                    f"Then count the occurrences of '{letter}':\n")
            run = 0
            for j, ch in enumerate(w, 1):
                # **序号和字母之间不能有空格** —— nanochat 特意说明:" a" 和 "a"
                # 是不同的 token,加空格会把这个格式教成另一个东西。
                if ch == letter:
                    run += 1
                    text += f"{j}:{ch} hit! count={run}\n"
                else:
                    text += f"{j}:{ch}\n"
            text += f"\nThis gives us {run}.\n\nMy final answer is:\n\n#### {count}"
            yield [("user", user), ("assistant", text)]
