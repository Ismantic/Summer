"""中文指令数据 —— 三个源,补的是**唯一条数**。

`Summer = nanochat + 中文 + ReTok`,而中文侧一直是供给不足:COIG-CQIA 全量只有
44,693 条,SmolTalk 有 460,341 条,差一个数量级。这个缺口在两个侧面各咬过一次:

    token 占比   照 nanochat 的行数配,中文只占 1.9%,只好把 SmolTalk 截到 10 万
    条数占比     单阶段里中文占 7.7% 条,中文停止率就只有 4%(英文 61%)

预演已证实**条数占比是 `<end>` 的杠杆**(7.7% → 24.9% 时中文停止率 4% → 31%),
但那是靠把 33,700 条重复 4 遍换来的,再加 epoch 就是过拟合。**要的是唯一条数。**

## 三个源

    Firefly_ZH     1,649,399 条   23 种任务类型,单轮,`kind`/`input`/`target`
    Magpie_ZH        200,000 条   自合成对话,质量高 —— 中文侧的 SmolTalk
    AlpacaGPT4_ZH     42,677 条   alpaca 格式,小而干净

## Firefly 要按 kind 筛,不能全量倒进去

它 23 种任务里有一大批**超短答案**的结构化任务(按 `target` 的字符中位):

    NLI 2   SentimentAnalyze 2   TextMatching 3   NER 5   MRC 6
    Couplet 10   Summary 20   KeywordRecognition 20   ClassicalChinese 23

全量倒进去,它们教的是「答完两个字就停」—— 和英文拼写题占了 47% 的条数是
**同一种病**,只是方向相反。刚在预演里吃过这个亏。

所以设了个门槛:**只留答案字符中位 ≥ 40 的 kind**(约 30 token)。
**这个阈值是我们定的,不是 nanochat 的** —— 它是纯英文项目,没有这个问题。
定 40 的理由:再低就进入「一两个词」的区间,那种长度学不到「答完一段再收尾」。

留下来的(条数 / 答案字符中位):

    BELLE 543,285/67   Cot 74,771/104   ProductDesc 70,000/88
    AncientPoem 69,950/56   OpenQA 69,843/183   MusicComment 50,000/231
    TextCorrection 50,000/44   Composition 50,000/582   Translation 50,000/54
    JinYongGeneration 49,990/991   LyricGeneration 49,985/347   Dictionary 30,895/59
    StoryGeneration 19,048/748   Program 974/144   ProseGeneration 658/1206

**超长的那几个(ProseGeneration/JinYong/StoryGeneration)会在 seq 1025 下被整条
丢弃**,留着不是浪费 —— `chat.py` 会照实报丢弃数,那个数本身是要看的。
"""
from __future__ import annotations

import json

from prepare.tasks._local import files, parquet_batches
from prepare.tasks.common import Task

#: 答案字符中位 ≥ 40 的 kind。理由见模块 docstring。
FIREFLY_KINDS = {
    "BELLE", "Cot", "ProductDesc", "AncientPoem", "OpenQA", "MusicComment",
    "TextCorrection", "Composition", "Translation", "JinYongGeneration",
    "LyricGeneration", "Dictionary", "StoryGeneration", "Program",
    "ProseGeneration",
}


class FireflyZH(Task):
    """110 万+ 中文指令。`kind` / `input` / `target`,单轮。"""

    name = "Firefly_ZH"
    lang = "zh"

    def __init__(self, stop=None, kinds: set[str] | None = None):
        super().__init__(stop=stop)
        # None = 用上面那份筛过的集合;传 set() 表示不筛(**会引入超短答案**)
        self.kinds = FIREFLY_KINDS if kinds is None else kinds

    def conversations(self):
        for path in files("Firefly_ZH"):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:                          # noqa: BLE001
                        continue
                    if self.kinds and d.get("kind") not in self.kinds:
                        continue
                    q = (d.get("input") or "").strip()
                    a = (d.get("target") or "").strip()
                    if q and a:
                        yield [("user", q), ("assistant", a)]


class MagpieZH(Task):
    """20 万条自合成中文对话。用 `conversations` 列(可能多轮),不用
    `instruction`/`response` —— 那两列只有第一轮。"""

    name = "Magpie_ZH"
    lang = "zh"
    ROLES = {"human": "user", "gpt": "assistant"}

    def conversations(self):
        for batch in parquet_batches("Magpie_ZH", ["conversations"]):
            for row in batch.column(0).to_pylist():
                if not row:
                    continue
                turns = [(self.ROLES.get(m.get("from", ""), m.get("from", "")),
                          m.get("value", "")) for m in row]
                turns = [(r, c) for r, c in turns if r in ("user", "assistant") and c]
                if turns:
                    yield turns


class AlpacaGPT4ZH(Task):
    """4.3 万条 alpaca 格式。**整个文件是一个 JSON 数组**,不是 jsonl。"""

    name = "AlpacaGPT4_ZH"
    lang = "zh"

    def conversations(self):
        for path in files("AlpacaGPT4_ZH"):
            with open(path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            for d in data:
                q = (d.get("instruction") or "").strip()
                extra = (d.get("input") or "").strip()
                a = (d.get("output") or "").strip()
                if q and a:
                    yield [("user", f"{q}\n{extra}" if extra else q),
                           ("assistant", a)]
