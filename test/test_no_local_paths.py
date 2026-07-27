"""被 git 跟踪的文件里不许出现本机绝对路径。

## 为什么要有这一项

这个仓库有 GitHub remote,也往 HF 发布模型。`/home/<用户名>/...` 这种路径
一旦被推上去,泄露的是**用户名**,而且 git 历史里删不掉 —— 只能改写历史强推。
2026-07-27 就发现已推的 `main` 里有 88 个文件带用户名,以及 HF 上的
`training_lineage.md` 整篇都是本机路径。

绝对路径对读者**没有任何信息量** —— 别人机器上的目录结构和你的不一样。
要举例就写 `~/...`,要能跑就走 `local.mk` / 环境变量 / `Path.home()`。

## 为什么不写死用户名

写死用户名本身就是泄露。这里从 `$HOME` / `$USER` 现取,所以换台机器也管用,
而且这个文件里不含任何本机信息。

    python test/test_no_local_paths.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def main() -> int:
    home = os.path.expanduser("~").rstrip("/")
    user = os.path.basename(home)
    # $HOME 之外也查 /home/<user> 和 /Users/<user> —— 有些路径是从别的机器
    # 抄来的,$HOME 匹配不上。
    needles = {home, f"/home/{user}", f"/Users/{user}"}
    # 用户名单独出现也算(比如出现在 URL、机器名里)。太短的名字会误报,
    # 所以要求它前后不是别的单词字符。
    pat = re.compile("|".join(
        [re.escape(n) for n in sorted(needles, key=len, reverse=True)]
        + ([rf"(?<!\w){re.escape(user)}(?!\w)"] if len(user) >= 4 else [])))

    hits = []
    for rel in tracked_files():
        p = ROOT / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue                       # 二进制,跳过
        for i, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                hits.append((rel, i, line.strip()[:100]))

    print(f"扫了 {len(tracked_files())} 个被跟踪的文件")
    print(f"找的是 $HOME({home})、/home/<user>、/Users/<user>、单独的用户名\n")
    if not hits:
        print("没有本机路径泄露 ok")
        return 0

    print(f"**{len(hits)} 处泄露本机信息** —— 这些会被推到 GitHub / HF:\n")
    for rel, i, line in hits:
        print(f"  {rel}:{i}")
        print(f"      {line}")
    print("\n改法:举例用 `~/...`;要能跑就走 local.mk / 环境变量 / Path.home()。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
