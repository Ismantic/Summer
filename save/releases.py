"""已发布到 Hugging Face 的东西。**这个文件是发布状态的唯一记录。**

    python save/releases.py              # 列出来
    python save/releases.py --verify     # 与 HF 上的实际文件核对 sha256

发布是不可逆的(别人可能已经下走了),所以这里记的每一项都要能对得上账:
本地哪个 checkpoint、哪些文件、sha256 是多少。
"""
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Release:
    repo_id: str
    local: str                      # 相对仓库根,产出这份发布的 checkpoint
    stage: str
    note: str = ""
    # 发布目录(export 的产物)。核对 sha256 用这个,不用 local ——
    # local 是训练产出的 checkpoint,里面的词表还是旧名 piece.model;
    # 发布目录里才是上游名。两者内容相同,文件名不同。
    # 留空则按 save/releases/<repo 名> 推。
    release: str = ""
    # 关键文件的 sha256。核对用 —— 发布之后本地和线上必须一致,
    # 不一致说明有一边被动过。
    sha256: dict[str, str] = field(default_factory=dict)

    def local_dir(self) -> Path:
        return ROOT / self.local

    def release_dir(self) -> Path:
        return ROOT / (self.release
                       or f"save/releases/{self.repo_id.split('/')[-1]}")


RELEASES = [
    Release(
        repo_id="Ismantic/Summer-0.5B-S0",
        local="output/summer05b_s0b",
        stage="S0(从零预训练,纯单语,2026-08-27 重做)",
        note="**替换了 2026-08 的原始 S0 发布**——重新对齐 nanochat 的实际训练"
             "配方:英文单源 FineWebEdu(原来是五源混合)、中文多源但筛过简繁"
             "(丢弃两个实测 51.8%/55.7% 繁体主导的源)、BOS 对齐 best-fit 打包"
             "(bos_bestfit,逐行核对过和 nanochat 源码算法一致;原来是无 BOS 的"
             "连续流)、seq_len 2048(原来是 1024,nanochat 从初始提交起就是"
             "2048)。14.6B token,英中 70:30,55,694 步。"
             "\n"
             "**每一次输入都必须以 <bos> 开头**——这个模型训练时从没见过不带"
             "BOS 的序列,不加会生成退化的重复内容(自己踩过这个坑:评测脚本"
             "漏加 BOS,似然协议 arc_easy 从 0.54 掉到接近随机,WMT22 5-shot 直接"
             "复读崩溃)。`example_load.py`/`example_vllm.py` 已经处理。"
             "\n"
             "对旧版:字母 MC 四项全面超过(旧版三项低于随机,新版四项都在"
             "随机线之上);似然协议 arc_easy/arc_challenge 明显更好,"
             "mmlu/ceval/gsm8k 基本持平。WMT22 5-shot 两版都接近零"
             "(纯单语,预期内)。没有发现任何指标倒退。",
        sha256={
            "model.safetensors":
                "ea4d8ca333b633e43b9afa1138904f0e72a57a5819642ecb816ed6eb751049cc",
            "Summer-Tokenizer.pt":
                "b9b81cefcaa5d47cd3aa6e653dda0a80f90b7863b3cfff790dfc07c662dda50f",
        },
    ),
    Release(
        repo_id="Ismantic/Summer-0.5B-S1",
        local="output/summer05b_s1b",
        stage="S1(在 S0 上用中英平行语料退火,2026-08-27 重做)",
        note="**替换了 2026-08 的原始 S1 发布**,和新版 S0 同一批重做:同样对齐"
             "nanochat 的 bos_bestfit 打包 + seq_len 2048。从 S0 的**最终权重**"
             "重新开始(不是从中间 checkpoint 续——A/B 测过带完整优化器状态续"
             "训 vs 权重清零重开,100 步内 loss 逐步几乎完全重合,两者等价),"
             "跑 5,103 步、1.34B token,含约 30% 中英平行语料,退火数据同样是"
             "bos_bestfit 打包(退火阶段如果打包方式和预训练不一致,会在训练"
             "时引入分布外偏移,是同一类问题的另一种表现)。"
             "\n"
             "**每一次输入都必须以 <bos> 开头**,同新版 S0。"
             "\n"
             "WMT22 5-shot zh-en BLEU 8.99 / COMET 0.6883,en-zh 28.36 / 0.7736"
             "——和旧版 S1(8.99 / 0.6855,27.29 / 0.7743)基本打平,新数据配方"
             "没有牺牲翻译能力。",
        sha256={
            "model.safetensors":
                "9533953cc0955d35444fd57c50bd4d5a6de018d0a228b0fe81c526295c5bf921",
            "Summer-Tokenizer.pt":
                "b9b81cefcaa5d47cd3aa6e653dda0a80f90b7863b3cfff790dfc07c662dda50f",
        },
    ),
    Release(
        repo_id="Ismantic/Qwen3-1.7B-Base-ReTok",
        local="output/phase2_ckpt_v18_tie",
        stage="Phase 2 (LoRA tie-safe)",
        note="v18。WMT22 zh-en BLEU 20.46 / COMET 0.7933。"
             "1,577,147,392 参数,310 个张量(tie,没有 lm_head.weight)。",
        sha256={
            "model.safetensors":
                "b8676b8410d661bf9879612786c8980aae72250ecb87cb375a604e724133162a",
            # 词表用上游名(与 PieceTokenizer 仓库 save/ 下同名)。
            # 2026-07-27 之前线上叫 piece.model,内容相同 —— 那次改名把旧名删了。
            "Summer-Tokenizer.pt":
                "b9b81cefcaa5d47cd3aa6e653dda0a80f90b7863b3cfff790dfc07c662dda50f",
        },
    ),
]


def cmd_list() -> None:
    for r in RELEASES:
        d = r.local_dir()
        print(f"\n{r.repo_id}")
        print(f"  https://huggingface.co/{r.repo_id}")
        print(f"  阶段    {r.stage}")
        rd = r.release_dir()
        print(f"  源 ckpt {r.local}  {'(在)' if d.exists() else '(**不在**)'}")
        print(f"  发布目录 {rd.relative_to(ROOT)}  "
              f"{'(在)' if rd.exists() else '(**不在** —— 先 make -C save export)'}")
        if r.note:
            print(f"  说明    {r.note}")


def cmd_verify() -> int:
    import hashlib

    from huggingface_hub import HfApi

    api = HfApi()
    ok = True
    for r in RELEASES:
        print(f"\n=== {r.repo_id} ===")
        try:
            info = api.model_info(r.repo_id, files_metadata=True)
        except Exception as e:                                 # noqa: BLE001
            print(f"  取不到 HF 信息:{type(e).__name__}: {e}")
            ok = False
            continue
        remote = {s.rfilename: getattr(s.lfs, "sha256", None) if s.lfs else None
                  for s in info.siblings}
        for name, expect in r.sha256.items():
            got_remote = remote.get(name)
            # 与线上比就该拿发布目录比 —— 文件名一致
            local_f = r.release_dir() / name
            got_local = (hashlib.sha256(local_f.read_bytes()).hexdigest()
                         if local_f.exists() else None)
            same_r = got_remote == expect
            same_l = got_local == expect if got_local else None
            ok &= same_r and (same_l is not False)
            print(f"  {name}")
            print(f"    登记 {expect[:16]}…")
            print(f"    HF   {(got_remote or '取不到')[:16]}…  "
                  f"{'ok' if same_r else '**不符**'}")
            print(f"    本地 {(got_local or '文件不在')[:16]}…  "
                  f"{'ok' if same_l else ('**不符**' if same_l is False else '跳过')}")
    print("\n" + ("核对通过" if ok else "**有不符项** —— 本地或线上被动过"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="与 HF 核对 sha256")
    a = p.parse_args()
    if a.verify:
        return cmd_verify()
    cmd_list()
    return 0


if __name__ == "__main__":
    sys.exit(main())
