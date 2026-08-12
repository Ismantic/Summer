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
        local="output/summer05b_s0",
        stage="S0(从零预训练,纯单语)",
        note="随机初始化从零训:Qwen3-0.6B-Base 架构 + 自训 81903 piece 词表 = "
             "524,336,128 参数,单语 12B token(中英 50:50)跑 45,149 步。"
             "\n"
             "**5-shot 翻译基本为零**(WMT22 zh-en BLEU 0.54 / COMET 0.4638,"
             "en-zh 3.97 / 0.5872):模型完全无视 few-shot 示例,zh→en 时连输出"
             "语言都不对。语言模型学到了,in-context learning 没有 —— 这不是"
             "缺陷,是 13B token 这个量级的实话。它的用途是当 S1 的对照,"
             "以及当后续 midtrain/SFT 的起点。",
        sha256={
            "model.safetensors":
                "d7c081f09d27487588dd88c11e5ef8734a11ca02ae9d7427fbee36b5c3d95a63",
            "Summer-Tokenizer.pt":
                "b9b81cefcaa5d47cd3aa6e653dda0a80f90b7863b3cfff790dfc07c662dda50f",
        },
    ),
    Release(
        repo_id="Ismantic/Summer-0.5B-S1",
        local="output/summer05b_s1",
        stage="S1(在 S0 上用中英平行语料退火)",
        note="从 S0 的 step 40,000 分叉,用含 30% 中英平行语料的 1.2B token 跑完"
             "退火段(40,000 → 45,149)。与 S0 是受控对照:同起点、同超参、"
             "同学习率时间表,**只差数据**。"
             "\n"
             "WMT22 5-shot zh-en BLEU 8.99 / COMET 0.6855,en-zh 27.29 / 0.7743"
             "—— 相对 S0 跳了一个量级(0.54 / 3.97),而且是定性跨越:S0 无视"
             "示例、输出语言都不对,S1 开始真的在翻译。这 1.2B token 里的"
             "平行语料是 in-context learning 出现的直接原因。",
        sha256={
            "model.safetensors":
                "d465bf052d71d07a236507b1145fdcaa87fa1d4c1bfa6d11f644de35e7dd3a79",
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
