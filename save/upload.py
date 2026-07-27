"""把发布目录传到 Hugging Face。

    python -m save.upload --dry-run      # 先看要传什么(默认就该先跑这个)
    python -m save.upload --code-only    # 只传代码和模型卡,不碰权重
    python -m save.upload                # 全部(含 3.0GB 权重)

## 先想清楚要不要传权重

`--code-only` 只跳过 `model.safetensors`(3GB),词表和代码都传 ——
推理代码、模型卡、配置。**权重没变时一定用它**:

- 3.0GB 重传一遍纯属浪费
- **覆盖已发布权重本身就是不必要的风险** —— 别人可能已经下走了,
  再传一次哪怕内容相同,也会产生新的 commit 和新的 LFS 指针

判断权重变没变:`python save/releases.py --verify` 做三方 sha256 核对
(登记值 / HF 线上 / 本地)。全 ok 就说明权重不用动。

## 发布是不可逆的

传完立刻:

1. `python save/releases.py --verify` 确认线上与本地一致
2. 如果权重变了,把新的 sha256 更新进 `releases.py`

需要目标 namespace 的写权限(`huggingface-cli login`)。

**注意代理**:hf-mirror 是只读镜像,上传必须走官方源。这个脚本不动
`HF_ENDPOINT`,但本机 shell 里如果设了 `HF_ENDPOINT=https://hf-mirror.com`
要先 unset —— 否则会往镜像推,失败得莫名其妙。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from save.releases import RELEASES                              # noqa: E402

DEFAULT_DIR = Path(__file__).resolve().parent / "releases"

# 要同时写 "__pycache__/**" —— "**/__pycache__/**" 匹配不到**顶层**的
# __pycache__/。BERTc 就因为这个把 .pyc 传上去过一次。
IGNORE = ["__pycache__/**", "**/__pycache__/**", "**/*.pyc", ".cache/**"]

# --code-only 时跳过的**只有 3GB 权重**。
#
# 词表(Summer-Tokenizer.pt / .dict.txt)加起来 7MB,而且**必须与加载器期望的
# 文件名保持同步** —— 改了名却不推,线上就会出现「模型卡写 Summer-Tokenizer.pt
# 而目录里是 piece.model」的不一致。所以词表跟代码一起走,不算 WEIGHTS。
WEIGHTS = ["model.safetensors"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*",
                    help="要传的 repo 名(不含 namespace),默认全部已导出的")
    ap.add_argument("--release-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--code-only", action="store_true",
                    help="只传代码 / 模型卡 / 配置,跳过权重和词表")
    ap.add_argument("--dry-run", action="store_true", help="只打印,不上传")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    if os.environ.get("HF_ENDPOINT", "").rstrip("/").endswith("hf-mirror.com"):
        print("HF_ENDPOINT 指向 hf-mirror(只读镜像),上传会失败。先 unset。")
        return 1

    by_name = {r.repo_id.split("/")[-1]: r for r in RELEASES}
    names = a.names or [n for n in by_name if (a.release_dir / n).exists()]
    if not names:
        print(f"{a.release_dir} 下没有已导出的目录。先跑:\n"
              f"  make -C save export")
        return 1
    unknown = [n for n in names if n not in by_name]
    if unknown:
        print(f"releases.py 里没登记:{unknown}")
        return 1
    missing = [n for n in names if not (a.release_dir / n).exists()]
    if missing:
        print(f"没导出:{missing}")
        return 1

    skip = set(WEIGHTS) if a.code_only else set()
    for name in names:
        folder = a.release_dir / name
        repo_id = by_name[name].repo_id
        files = [f for f in sorted(folder.iterdir())
                 if f.is_file() and f.name not in skip and f.suffix != ".pyc"]
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"\n  {repo_id}  ←  {folder}"
              f"  ({size:.1f} MB, {len(files)} 个文件)"
              + ("  [私有]" if a.private else ""))
        for f in files:
            print(f"      {f.name}")
        if skip:
            print(f"    跳过:{sorted(skip)}")
        if a.dry_run:
            continue

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=a.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(folder),
            ignore_patterns=IGNORE + (WEIGHTS if a.code_only else []),
            commit_message=("Update inference code and model card"
                            if a.code_only else f"Upload {name}"))
        print(f"    → https://huggingface.co/{repo_id}")

    if a.dry_run:
        print("\n--dry-run:什么都没传。去掉这个参数才会真上传。")
    elif a.code_only:
        print("\n--code-only:权重未动,只更新了代码和模型卡。")
    else:
        print("\n传完了。立刻跑 `python save/releases.py --verify` 核对,"
              "权重变了的话把新 sha256 更新进 releases.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
