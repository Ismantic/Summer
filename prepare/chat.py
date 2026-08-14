"""对话数据的预编码 —— midtrain 与 SFT 两个阶段共用。

    python -m prepare.chat --mix midtrain --output output/midtrain_1024
    python -m prepare.chat --mix sft      --output output/sft_1024

产出 `<out>.pt`(int32 `[N, seq_len]`,和预训练同格式)加 `<out>.mask.pt`
(uint8 同形状,1 = 这个位置要算 loss)。`src/train.py --loss_mask <path>` 读它。

## 为什么需要这一层(而不是直接 SFT)

预训练语料全是纯文本,`<user>` / `<assistant>` / `<system>` 三个 token 从没作为
**输入**出现过。实测 Summer-0.5B-S1 的权重里,它们和 `<pad>` 的余弦相似度
**全是 +1.0000** —— 在模型眼里是同一个向量(普通 token 之间平均 0.285)。
它们的范数从 0.64 长到 1.66,只是因为 embedding 与 lm_head 绑权重、每步都在
把「不该预测的 token」的 logit 往下压,四个受到完全相同的推力。

所以直接上 SFT,等于让模型在一个 epoch 里同时学任务和学这三个分隔符。
nanochat 把这件事拆成 midtrain(先用对话格式的数据把模板喂熟,整段算 loss)
和 SFT(只在助手那部分算 loss),这里照它分。

## 打包:best-fit,只放完整对话,尾部补 pad

**每条序列里的对话都是完整的。** 早先的做法是把对话首尾相接后按 seq_len 切,
于是一条序列可能从半截对话开始、也可能在半截处结束 —— 那种上下文推理时永远
不会出现,等于拿模型没见过的分布去训它。

现在改成:攒满就换新序列,尾部用 pad 补齐并 mask=0 掩掉。

**尾部 padding 在因果注意力下是安全的。** 自写的模型只做 is_causal,不接受
attention_mask;但 pad 在序列末尾 —— 后面的 pad 看得见前面的真内容,前面的真
内容看不见后面的 pad(因果),而 pad 位置本身 mask=0 不参与 loss。会算错的是
**左侧或中间**填充,不是尾部。

**用 best-fit 而不是 first-fit**(与 nanochat 的 chat_sft.py 一致):每次从待放
池里挑**最大的能装下的**那一段,而不是按顺序装。first-fit 实测填充率 25.6%,
best-fit 能压到个位数 —— 差别就是四分之一的算力在训 pad 还是在训数据。

## 掩码

midtrain:整段算 loss(mask 全 1)。目的是让模型见过这个格式,不是精调答案。
SFT:只在助手回复 + 其后的 eos 上算 loss。用户那段不算 —— 否则模型会去学
「怎么生成问题」,那不是我们要的能力,而且会稀释真正的监督信号。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare.tokenizer import (PieceTokenizerWrapper,           # noqa: E402
                               resolve_assets)

# ------------------------------------------------------------------ 配比
#
# 混比 = **一个任务实例的列表**,每个数据集一个类,放在 `prepare/tasks/`。
# 对齐 nanochat 的 `TaskMixture` —— **同一个任务传两次就是两个 epoch**。
#
# ## 为什么从 token 权重改成按行数
#
# 原来按 token 权重分配,坑是**权重是上限不是保证**:C3 规划 0.10(10.6M)实得
# 只有 2.4M,因为全量就 11869 条、抽干了,缺口被 SmolTalk 顶上 —— 而账面上
# 看不出来,`planned_weights` 写着 0.10,你得去翻 `actual_tokens` 才发现。
# 按行数写就没有这回事:要多少条就是多少条,数据不够会直接少。
#
# ## 中文侧是我们多出来的那一半
#
# 对齐目标是**两阶段**那一版(`aa530cd^` 的 mid_train.py + chat_sft.py),
# 不是上游现在合并后的单阶段 —— 我们对照的 d20 就是两阶段训出来的。
#
# **逐条照搬会删掉中文。** nanochat 是纯英文的;Summer = nanochat + 中文 + ReTok,
# 所以中文侧配对应的源:COIG-CQIA 对 SmolTalk,C3 对 MMLU_AuxTrain。
# 汉字不是字母拼的,所以拼写那两个任务没有中文对应物,是纯英文的。
#
# ### 一处有意的偏离:SmolTalk 截到 10 万行(它用全量 460,341)
#
# 照搬全量的实测账(每条的 token 数是抽 800 条量的):
#
#   SmolTalk       460,341 行 × 804 = 370.2M   ← 一家占 87%
#   MMLU_AuxTrain   99,842 行 × 281 =  28.1M
#   SpellingBee      80,000 行 × 147 =  11.7M
#   SimpleSpelling  200,000 行 ×  32 =   6.4M
#   COIG_CQIA        44,694 行 × 137 =   6.1M ┐ 中文 8.3M
#   C3_Train         11,869 行 × 179 =   2.1M ┘ = **1.9%**
#   合计 426M
#
# 问题出在供给:**COIG-CQIA 全量只有 44,694 条,SmolTalk 有 460,341 条,差 10 倍。**
# 照搬会把中文从 v5 的 15% 压到 1.9%,而 v4/v5 刚证明中文侧是最脆的一环
# (打包方式一改,中文停止率 29% → 69%,英文几乎不动)。
#
# 截到 10 万之后:总量 136M、中文 6.0%、SmolTalk 占 59%。
#
# **这是数据供给的问题,不是混比的问题。** 真正的解法是补更大的中文指令集
# (Infinity-Instruct 中文子集、Magpie 中文、firefly 之类),把中文侧做到十万条
# 量级,那时候才能真按 nanochat 的行数对齐。在那之前这个 10 万是权衡。
#
# 顺带一个比 A 组更大的差距:**我们原来的 midtrain 只有 100M,对齐后 136M,
# 而全量对齐是 426M。** 按比例算 nanochat 是约 400M/11B = 3.6%,我们 13B 底座
# 对应该有 470M —— 一直偏小了 4 倍多。
#
# ## seq_len 仍是 1024,这是已知限制
#
# COIG-CQIA 的回答很长(中位 1126 token),1024 下只有 66% 能整条装下。
# 2048 能把中文提到近 3 倍,但**预训练全程只见过 1024**,位置 1024-2048 的 RoPE
# 从没被训过 —— 用 2048 做后训练等于让模型顺便学「长位置怎么编码」,
# 正是在特殊 token 上吃过的亏。**这是已知限制,不是待修的 bug。**
def MIDTRAIN_TASKS():
    from prepare.tasks.c3 import C3
    from prepare.tasks.coig_cqia import COIGCQIA
    from prepare.tasks.gsm8k import GSM8K
    from prepare.tasks.identity import Identity
    from prepare.tasks.mmlu import MMLUAux
    from prepare.tasks.smoltalk import SmolTalk
    from prepare.tasks.spelling import SimpleSpelling, SpellingBee
    return [
        # **这里偏离 nanochat:它用全量 460,341 行,我们截到 10 万。** 理由见下。
        SmolTalk(stop=100000),         # en 通用对话  370M → 80M
        MMLUAux(),                     # en 多选题型  nanochat: 100K 行  28.1M
        GSM8K(),                       # 数学/分步推理  全量 7,473 行  1.2M
        Identity(1000), Identity(1000),  # 身份 ×2 个 epoch,和 nanochat 一样
        SimpleSpelling(200000),        # 拼写  nanochat: 200K 行  6.4M
        SpellingBee(80000),            # 数字母  nanochat: 80K 行  11.7M
        # ---- 以下是 nanochat 没有的中文侧 ----
        COIGCQIA(),                    # zh 通用对话,对应 SmolTalk  全量 44,694 行
        C3(),                          # zh 多选题型,对应 MMLU_AuxTrain  11,869 行
    ]


# SFT:nanochat 的 23K 行 —— **和 midtrain 是不同的配方**,以短答案题型为主
# (ARC + GSM8K),SmolTalk 只取前 1 万条。
def SFT_TASKS():
    from prepare.tasks.arc import ARC
    from prepare.tasks.c3 import C3
    from prepare.tasks.coig_cqia import COIGCQIA
    from prepare.tasks.gsm8k import GSM8K
    from prepare.tasks.identity import Identity
    from prepare.tasks.smoltalk import SmolTalk
    from prepare.tasks.spelling import SimpleSpelling, SpellingBee
    return [
        ARC("ARC-Easy"),               # nanochat: 2.3K 行
        ARC("ARC-Challenge"),          # nanochat: 1.1K 行
        GSM8K(),                       # nanochat: 8K 行
        SmolTalk(stop=10000),          # nanochat: 前 10K 行
        Identity(1000),                # nanochat: 1K 行
        SimpleSpelling(300), SpellingBee(300),
        # ---- 中文侧,按 en 的比例配 ----
        COIGCQIA(stop=10000),          # 对应 SmolTalk 那 10K
        C3(stop=3400),                 # 对应 ARC 的 2.3K+1.1K
    ]

# **掺进预训练的那份。** 与 midtrain 同样的配方,但产出的 shard 直接和纯文本
# shard 并列喂给 S2 —— 让 <user> / <assistant> 跟着 10B token 一起训,而不是
# 等预训练结束后靠 34.8M 的 midtrain 去补。
#
# 实测:S0 训完 11.8B token 后这三个 token 的余弦是 **+1.0000**(完全同一个
# 向量),因为预训练语料里它们出现 0 次;34.8M 的 midtrain 只把它推到 0.97。
# 掺进预训练能给它们约 250M token 的曝光,是 midtrain 的 7 倍。
#
# **整段算 loss,不掩码** —— 和 midtrain 同理,这一步是让模型见熟格式,不是精调
# 答案。所以产出的 shard 格式与纯文本完全一致(int32 [N, seq_len],无 mask 文件),
# 训练时直接并进 --train_data 的逗号列表。
# ---------------------------------------------------------------- 设计 B
#
# **单阶段后训练**,对齐上游现在的 `chat_sft.py`(commit 1ddaad1 之后)。
#
# 两套设计不能混着抄,列清楚:
#
#            旧版(= d20,我们的对照)          新版(设计 B)
#   预训练   连续流,无 BOS 对齐               **BOS + best-fit**
#   midtrain 连续流,全速 lr,**整段算 loss**   没有
#   SFT      best-fit,lr 2%,assistant-only   best-fit,**lr 0.8**,assistant-only
#
# 打包和掩码不用另写:`do_pack` 和掩码那两处的条件都是「不在 midtrain /
# chat_pretrain 里就走 best-fit + 只算 assistant」,正好是设计 B 要的。
#
# ## 混比:旧 midtrain 那套,但 MMLU 不跟着加到 3 个 epoch
#
# 上游新版是 MMLU ×3、GSM8K ×4(`--mmlu-epochs` / `--gsm8k-epochs` 的默认值)。
# GSM8K ×4 照抄(它小,1.2M × 4 = 4.8M);**MMLU 只放 1 个 epoch**,偏离。
#
# 理由和 SmolTalk 截到 10 万是同一个:MMLU_AuxTrain 是**英文**多选,×3 就是
# 84M,而中文多选(C3)全量只有 2.1M。我们已经量到英文格式跟随 0.96 早就饱和、
# 中文才 0.78 —— 再加 3 倍英文多选只会把这个差距拉得更开,而不是补短板。
# 真正该做的是补中文多选的量,那是数据问题。
def CHAT_TASKS(mmlu_epochs: int = 3, smoltalk: int | None = 172000,
               firefly: int = 35000):
    from prepare.tasks.arc import ARC
    from prepare.tasks.c3 import C3
    from prepare.tasks.coig_cqia import COIGCQIA
    from prepare.tasks.gsm8k import GSM8K
    from prepare.tasks.identity import Identity
    from prepare.tasks.mmlu import MMLUAux
    from prepare.tasks.smoltalk import SmolTalk
    from prepare.tasks.spelling import SimpleSpelling, SpellingBee
    from prepare.tasks.zh_instruct import AlpacaGPT4ZH, FireflyZH, MagpieZH
    return [
        # **读 17.2 万条,实得约 10 万完整对话。** 不切分之后 SmolTalk 有 41.6%
        # 装不下 seq 1024(它是唯一的多轮源),所以往后多读来补齐 —— 池子有 46 万,
        # 够。**代价是「读得更靠后」,不是「数据变少」。**
        SmolTalk(stop=smoltalk),       # None = 全量 460,341(= d20)
        # **×3 是上游新版 chat_sft 的默认(--mmlu-epochs);但 d20 的 midtrain 只有 ×1。**
        # 而 MMLU-aux 的来源是「ARC / MC_TEST / OBQA / RACE」—— 它不只帮 mmlu,
        # **也直接帮 arc_easy**。所以拿 ×3 的成绩和 d20 比,两个胜负都不干净。
        # `mix=chat_mmlu1` 就是为了拿干净结论的对照(见 MIXES)。
        *[MMLUAux() for _ in range(mmlu_epochs)],
        ARC("ARC-Easy"), ARC("ARC-Challenge"),  # **d20 的 SFT 里有 ARC train**
        *[GSM8K() for _ in range(4)],  # ×4,照抄 --gsm8k-epochs
        Identity(1000), Identity(1000),
        SimpleSpelling(200000),
        SpellingBee(80000),
        # ---- 中文侧,nanochat 没有 ----
        #
        # **按条数配到和英文对半,而且全是唯一条目、不重复。**
        #
        # `<end>` 是**每条对话出现一次**的信号,所以它的行为由**条数占比**决定,
        # 不是 token 占比 —— 预演实测:中文占 7.7% 条时中文停止率只有 4%
        # (英文 61%),提到 24.9% 就回到 31%。而 v4/v5 那个有 69~73% 的阶段
        # (sft3)中文按条数超过 50%。
        #
        # 之前只有 COIG(33,700 条存活)时,只能靠 ×4 重复凑占比,那是在过拟合。
        # 补了三个源之后唯一条数约 145 万,**重复没必要了**:
        #
        #   Firefly(筛过 15 个 kind)  1,179,399   丢弃 1.8%   token 中位 114
        #   Magpie                       200,000   丢弃 2.1%          441
        #   AlpacaGPT4                    42,677   丢弃 0.0%          119
        #   COIG(存活)                   33,700   丢弃 24.6%         215
        #
        # 新源在 seq 1025 下几乎不丢,正好补上 COIG 的短板(它丢掉四分之一)。
        #
        # **配到对半的理由**:预训练语料本身是中英 50:50(`SCRATCH_WEIGHTS`),
        # 后训练偏到一边会浪费底座的一半;而且 v4/v5 唯一成功的那个阶段就是对半。
        MagpieZH(),                    # 中文侧的 SmolTalk:自合成对话,答案偏长
        FireflyZH(stop=firefly),       # 广度:15 种任务类型,答案偏短
        AlpacaGPT4ZH(),
        COIGCQIA(),                    # 保留 —— 知乎长答那类,别的源没有
        C3(),                          # zh 多选题型,对应 MMLU_AuxTrain
    ]


MIXES = {"midtrain": MIDTRAIN_TASKS, "sft": SFT_TASKS,
         "chat_pretrain": MIDTRAIN_TASKS,
         "chat": CHAT_TASKS,                      # 设计 B:单阶段
         # **对照组:MMLU 只 1 个 epoch,和 d20 的 midtrain 一致。**
         # 其余一切和 `chat` 逐项相同 —— 单变量,既能拿干净的对 d20 胜负,
         # 也能量出 MMLU ×3 到底值多少点。
         "chat_mmlu1": lambda: CHAT_TASKS(mmlu_epochs=1),
         # **v8:SmolTalk 放到全量,中文按比例跟上,保持中文约 35%。**
         #
         # 起因是把 v7 和 d20 的**条数构成**逐项对了一遍:
         #
         #             d20(871K 条)      v7(870K 条)
         #   SmolTalk    470K = 54%       100K = 11.5%   ← 差 4.7 倍
         #   拼写        281K = 32%       280K = 32%      一样
         #   MMLU-aux    100K = 11.5%     100K = 11.5%    一样
         #
         # **拼写和 MMLU 的占比和它一模一样,唯独真实对话少 4.7 倍。**
         # 而 SmolTalk 正是「长答案 + <end>」的主要来源 —— 这解释了停止率卡在
         # 67~71%:模型见过的「答完一整段再收尾」太少。
         #
         # 那个 100K 上限是当初中文只有 33,700 条时设的(不截中文会被压到 1.9%)。
         # 补了中文供给之后那个约束没有了。
         #
         # **两处一起动是有意的对照设计,不是混淆**:SmolTalk 拉满的同时把中文
         # 按比例提上去,**让中文条数占比保持不变**,这样变的只有「真实对话占比」。
         "chat_full": lambda: CHAT_TASKS(mmlu_epochs=1, smoltalk=None,
                                         firefly=228000)}


# ------------------------------------------------------------------ 编码
def encode_turns(tok, turns, seq_len, split=False):
    """→ [(ids, mask), ...]。`split=False`(默认)时**只产出完整对话**,装不下就丢。

    mask=1 的位置要算 loss(只在助手回复及其 eos 上)。

    与 tokenizer.apply_chat_template 产出的序列**必须逐 token 相同** ——
    推理时走的是那个函数,训练和推理的格式对不上,学到的东西就用不上。
    这里自己拼是因为要同时产出掩码,拼法逐行对齐那边的实现。

    ## 为什么切段而不是丢弃

    seq_len 1024 下 SmolTalk 有 **40.9%** 的对话装不下(COIG-CQIA 22.5%) ——
    它是多轮长回复。整条丢掉等于扔掉四成数据,而且扔掉的恰好是信息最多的
    那些长对话,剩下的全是短问答,模型会学得又短又浅。

    切段的粒度是**轮次边界**:攒满就切,下一段重新以 <bos> 开头。绝不截在一轮
    中间 —— 截在助手回复中途的话,模型会学到「说到一半就停」。
    单轮本身就超长的(极少),那一轮整个丢掉。
    """
    segs = []
    ids: list[int] = []
    mask: list[int] = []

    def start_seg(system_content=None):
        nonlocal ids, mask
        ids, mask = [tok.bos_token_id], [0]
        if system_content:
            s = tok.encode(system_content, add_special_tokens=False)
            ids.extend(s); mask.extend([0] * len(s))
            # system 段用 <assistant> 之外的分隔已无可用 token(81902 已改作
        # 回答结束),而对话数据里几乎没有 system 消息 —— 直接把系统内容并进
        # 第一条用户消息,不引入新标记。
        ids.append(tok.user_token_id); mask.append(0)

    sys_content = turns[0][1] if turns and turns[0][0] == "system" else None
    body = turns[1:] if sys_content else turns
    start_seg(sys_content)

    # 一轮 = 用户 + 助手。按轮成对处理,这样切点一定落在轮次边界上。
    i = 0
    while i < len(body):
        role, content = body[i]
        turn_ids: list[int] = []
        turn_mask: list[int] = []
        if role == "user":
            turn_ids.append(tok.user_token_id); turn_mask.append(0)
            b = tok.encode(content, add_special_tokens=False)
            turn_ids.extend(b); turn_mask.extend([0] * len(b))
            # 紧跟的助手回复一起算进这一轮
            if i + 1 < len(body) and body[i + 1][0] == "assistant":
                b2 = tok.encode(body[i + 1][1], add_special_tokens=False)
                turn_ids.append(tok.assistant_token_id); turn_mask.append(0)
                turn_ids.extend(b2); turn_mask.extend([1] * len(b2))
                # **专用的回答结束符,不是 <eos>。** <eos> 在预训练里是文档
                # 分隔符(0.094%,S0 全程约 1100 万次),学到的含义是「后面还有
                # 下一篇」;拿它表示「到此为止」是在跟先验打架。
                # nanochat 用专用的 <|assistant_end|>,同理。见 DATA_FORMAT.md。
                turn_ids.append(tok.end_token_id); turn_mask.append(1)
                i += 2
            else:
                i += 1
        else:
            i += 1
            continue

        # **前缀不一定只有 bos** —— 有 system 时是 bos + 内容 + system_token。
        # 早先写成 len(turn_ids)+1,于是带 system 的样本会拼出超过 seq_len 的段,
        # 一路漏到 np.stack 才报「shape 不一致」。
        prefix_len = 1 + (len(tok.encode(sys_content, add_special_tokens=False)) + 1
                          if sys_content else 0)
        if prefix_len + len(turn_ids) > seq_len:
            continue                      # 单轮就超长,这一轮丢掉
        if len(ids) + len(turn_ids) > seq_len:
            if not split:
                # **对齐 nanochat:不切分。** 它的 chat_sft.py 只放「整条装得下」
                # 的对话,装不下就留在缓冲区等下一行,注释写着
                # 「pad the remainder instead of cropping — This ensures we
                # never discard any tokens」。**从不产生残片。**
                #
                # 我们原来切段,理由是「整条丢掉等于扔掉四成数据」(seq 1024 下
                # SmolTalk 有 40.9% 装不下)。但切出来的第二段开头是个引用了
                # 不存在前文的用户提问 —— **那种上下文推理时永远不会出现**,
                # 等于拿模型没见过的分布去训它。
                return []
            if any(mask):                 # 上一段有监督内容才留
                segs.append((ids, mask))
            start_seg(sys_content)
        ids.extend(turn_ids); mask.extend(turn_mask)

    if any(mask):
        segs.append((ids, mask))
    return segs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mix", choices=list(MIXES), required=True)
    p.add_argument("--output", required=True, help="不带扩展名,会写 .pt 和 .mask.pt")
    p.add_argument("--total_tokens", type=int, default=400_000_000)
    p.add_argument("--seq_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer_model", default=None)
    p.add_argument("--cn_dict", default=None)
    a = p.parse_args()

    # 词表在 PieceTokenizer 仓库的 save/ 下,本仓库不留副本 ——
    # resolve_assets() 反查(见 CLAUDE.md)。Wrapper 要的是**目录**。
    piece = a.tokenizer_model or resolve_assets()[0]
    tok = PieceTokenizerWrapper(os.path.dirname(os.path.abspath(str(piece))))

    tasks = MIXES[a.mix]()
    rng = random.Random(a.seed)
    seq_len = a.seq_length
    print(f"[chat] mix={a.mix}  seq_len={seq_len}  {len(tasks)} 个任务")
    print(f"  **按行数混,不按 token 权重** —— 有多少条就是多少条,不做上限裁剪")
    if a.total_tokens:
        print(f"  --total_tokens {a.total_tokens:,} 被忽略(按行数混之后它没有意义)")

    # 每个源单独打包 —— 混在一起打包会让某个源的对话被别的源截断,
    # 而且 shuffle 之后配比对不上账。
    all_ids: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    actual: dict[str, int] = {}
    rows_seen: dict[str, int] = {}
    padded = 0
    for task in tasks:
        name = task.name
        buf_i: list[int] = []
        buf_m: list[int] = []
        got = dropped = split = 0
        pad_id = tok.pad_token_id

        pool: list[tuple[list[int], list[int]]] = []

        def pack_stream(items):
            """**midtrain 用:连续流、切满、不 padding。**

            对齐 nanochat 的 `scripts/mid_train.py` —— 它用 deque 缓冲把对话首尾
            相接,凑够 `device_batch_size * max_seq_len + 1` 就切一批,和它预训练
            的 dataloader 同款。**一条对话可能被切断在中间**,那是有意的。

            与 SFT 的 pack() 相反,理由见 do_pack。
            """
            nonlocal buf_i, buf_m
            for ii, mm in items:
                buf_i.extend(ii); buf_m.extend(mm)
                while len(buf_i) >= seq_len:
                    all_ids.append(np.asarray(buf_i[:seq_len], dtype=np.int32))
                    all_mask.append(np.asarray(buf_m[:seq_len], dtype=np.uint8))
                    del buf_i[:seq_len], buf_m[:seq_len]
            items.clear()

        def do_pack(items, final=False):
            """**按阶段选打包方式,这是 nanochat 分开处理的地方。**

            midtrain 走 pack_stream(连续流、切满、无 padding),SFT 走 pack
            (best-fit、补 pad)。理由是它自己在两处写清楚的:midtrain 阶段
            数据管够,算力才宝贵,所以截断填满;SFT 阶段数据稀缺,一条都不能丢,
            所以宁可浪费算力补 pad。

            **早先两个阶段都用了 best-fit,那是错的。** 我把 SFT 的做法误推广
            到了 midtrain,代价是 5.4% 的算力在训尾部 padding。nanochat 的
            mid_train.py 用的是 deque 连续流缓冲(和它预训练的 dataloader 同款),
            grep 不到任何 mask —— 整段算 loss、也不留 pad。
            """
            if a.mix in ("midtrain", "chat_pretrain"):
                pack_stream(items)
                return
            pack(items, final=final)

        def pack(items, final=False):
            """**best-fit**:每行反复挑「最大的还装得下的」,装不下才 pad。

            与 nanochat 的 chat_sft.py 一致。first-fit(按顺序装)实测填充率
            25.6%,best-fit 能压到个位数 —— 差的是四分之一算力训 pad 还是训数据。
            """
            items.sort(key=lambda x: -len(x[0]))
            used = [False] * len(items)
            n_left = len(items)
            while n_left:
                row_i: list[int] = []
                row_m: list[int] = []
                progressed = True
                while progressed:
                    progressed = False
                    for k, (ii, mm) in enumerate(items):
                        if used[k]:
                            continue
                        if len(row_i) + len(ii) <= seq_len:
                            row_i.extend(ii); row_m.extend(mm)
                            used[k] = True; n_left -= 1
                            progressed = True
                            break            # 已按长度降序,第一个装得下的就是最大的
                if not row_i:
                    break
                _emit(row_i, row_m)
            items[:] = [it for k, it in enumerate(items) if not used[k]]

        def _emit(row_i, row_m):
            nonlocal padded
            if len(row_i) > seq_len:
                raise AssertionError(
                    f"打包出了长度 {len(row_i)} 的序列,超过 seq_len={seq_len}")
            n_pad = seq_len - len(row_i)
            padded += n_pad
            # **补 BOS 不补 <pad>,对齐 nanochat**(chat_sft.py:`row.extend(
            # [bos_token] * remaining)`)。两者都 mask=0、都在行尾,数值上等价;
            # 对齐是为了少一处「我们和它不一样」的地方。
            all_ids.append(np.asarray(row_i + [tok.bos_token_id] * n_pad, dtype=np.int32))
            all_mask.append(np.asarray(row_m + [0] * n_pad, dtype=np.uint8))

        rows = 0
        for turns in task:
            rows += 1
            segs = encode_turns(tok, turns, seq_len)
            if not segs:
                dropped += 1
                continue
            if len(segs) > 1:
                split += 1
            for ids, mask in segs:
                if a.mix in ("midtrain", "chat_pretrain"):
                    mask = [1] * len(ids)  # 整段算 loss:教格式,不精调答案
                pool.append((ids, mask))
                got += len(ids)
            # 池子攒够了就打包一批,不用等全部读完(那要占几个 G 内存)
            if len(pool) >= 4000:
                do_pack(pool)
        if pool:                       # 收尾:池子里剩下的也打包掉
            do_pack(pool, final=True)
        # **连续流不足一行的尾巴直接丢**,不补 pad —— nanochat 的 deque 也是这样。
        # 补了就破坏「零填充、100% 被监督」那个性质,而那正是 v4 的收益来源。
        # 每个任务最多丢 seq_len-1 个 token。
        # **同一个任务可能出现多次(= 多个 epoch),要累加不能覆盖。**
        actual[name] = actual.get(name, 0) + got
        rows_seen[name] = rows_seen.get(name, 0) + rows
        print(f"  {name:20s} {rows:>7,} 条 → {got:>12,} token"
              f"(切成多段 {split:,},整条丢弃 {dropped:,})")

    if not all_ids:
        print("!! 一条都没编出来"); return 1

    idx = list(range(len(all_ids)))
    rng.shuffle(idx)
    arr = np.stack([all_ids[i] for i in idx])
    msk = np.stack([all_mask[i] for i in idx])

    import torch
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    torch.save(torch.from_numpy(arr), a.output + ".pt")
    torch.save(torch.from_numpy(msk), a.output + ".mask.pt")
    ratio = msk.sum() / msk.size
    meta = {"mix": a.mix, "seq_length": seq_len, "chunks": int(arr.shape[0]),
            "total_tokens": int(arr.size), "supervised_ratio": float(ratio),
            # **不再有 planned_weights** —— 按行数混就没有「规划 vs 实得」的落差
            # 这回事了。`tasks` 记的是混比本身(同一个名字出现两次 = 两个 epoch),
            # `rows` 和 `actual_tokens` 记实得。
            "tasks": [repr(t) for t in tasks],
            "rows": rows_seen, "actual_tokens": actual}
    with open(a.output + ".mix.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n写出 {arr.shape[0]:,} 条 × {seq_len} = {arr.size:,} token")
    print(f"  被监督的比例 {ratio * 100:.1f}%  |  尾部填充 {padded / arr.size * 100:.1f}%")
    print(f"  {a.output}.pt / .mask.pt / .mix.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
