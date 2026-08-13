"""按 data/source.py 的登记下载语料与评测资产。

    python data/download.py --list           # 看每个源下了没
    python data/download.py --all            # 下全部
    python data/download.py --pretrain       # 只下预训练语料
    python data/download.py FineWebEdu CC_EN # 只下指定的几个
    python data/download.py --all --dry      # 只看要下什么,不动手

落地位置由 `SUMMER_DATA_ROOT` 控制,默认 `data/downloads/`。
走官方源:`--endpoint ""`,或 `HF_ENDPOINT= python data/download.py ...`。

**代理**:hf-mirror 走不通代理,而 GitHub 反过来需要代理。这里在 import 阶段
就把代理环境变量摘掉(与 BERTc 的 data/download.py 同一处理)。

`n_parts` 的截断在客户端做 —— `allow_patterns` 只能筛路径、不能限个数,所以
先列全表、按 glob 筛、再取前 N 个。取「前 N 个」而不是随机 N 个是有意的:
换机器重下能拿到同一批文件。
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import source                                        # noqa: E402

# hf-mirror 走不通代理。在任何网络调用之前摘掉。
SAVED_PROXY = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
for _k in SAVED_PROXY:
    del os.environ[_k]

# **用镜像就必须关掉 xet,否则镜像等于没用。**
#
# xet 是 HF 新的分块传输后端。它只从 HF 官方的 `cas-bridge.xethub.hf.co` 取
# 数据块 —— `HF_ENDPOINT` 只影响元数据请求,blob 该走 xet 还是走 xet。于是
# 镜像被绕过,而且这条链路还刚好是最慢的。
#
# 2026-07-27 实测(同一批文件,同一台机器):
#
#     开 xet    0.5 MB/s   —— 还伴随 cas-bridge 的 Read timed out
#     关 xet  117   MB/s   —— 234 倍
#
# 这个差别不会报错,只表现为「下载特别慢」——213GB 按 0.5MB/s 要跑 5 天。
# 走官方源(`--endpoint ""`)时不关,那种情况下 xet 是真的更快。
if "hf-mirror.com" in os.environ.get("HF_ENDPOINT", source.HF_ENDPOINT):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _glob_to_re(pat: str) -> re.Pattern:
    """glob → regex。`*` 不跨 `/`,`**` 跨 `/`。"""
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i:i + 2] == "**":
                out.append(".*")
                i += 2
                if pat[i:i + 1] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _repo_type(src: source.Source) -> str:
    return src.repo_type


def probe_one(src: source.Source, dry: bool) -> bool:
    """只下第一个匹配文件,然后真读一遍 —— 验 glob / fmt / text_field 对不对。

    全量下载动辄上百 GB,而真正会写错的是路径、格式、字段这三样,一个文件就
    能暴露。`compileall` 和 dry-run 都看不出这类错(见 `_attic/MANIFEST.md`
    里那个 `sys` 没 import 的例子)。
    """
    from huggingface_hub import HfApi, hf_hub_download

    endpoint = source.HF_ENDPOINT or None
    rtype = _repo_type(src)
    api = HfApi(endpoint=endpoint)
    try:
        files = sorted(api.list_repo_files(src.repo_id, repo_type=rtype))
    except Exception as e:                                     # noqa: BLE001
        print(f"  !! 取不到文件表:{type(e).__name__}: {e}")
        return False

    pats = [_glob_to_re(p) for p in (src.allow_patterns or [src.part_glob])]
    matched = [f for f in files if any(p.match(f) for p in pats)]
    if not matched:
        print(f"  !! glob 匹配不到任何文件:{src.allow_patterns or src.part_glob}")
        return False
    first = matched[0]
    print(f"  探针文件 {first}  (共匹配 {len(matched)} 个)")
    if dry:
        return True

    dest = src.dir()
    try:
        hf_hub_download(repo_id=src.repo_id, repo_type=rtype, filename=first,
                        local_dir=str(dest), endpoint=endpoint)
    except Exception as e:                                     # noqa: BLE001
        print(f"  !! 下载失败:{type(e).__name__}: {str(e)[:120]}")
        return False

    if src.repo_type == "model":
        print(f"  ok  {(dest / first).stat().st_size / 1e6:.0f}MB")
        return True

    # 真读一遍:格式对不对、字段在不在
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from prepare.encode_corpus import iter_text
        n = chars = 0
        for txt in iter_text(src.fmt, [str(dest / first)], src.text_field):
            chars += len(txt)
            n += 1
            if n >= 20:
                break
        if n == 0:
            print(f"  !! fmt={src.fmt} field={src.text_field} 读不出任何文本")
            return False
        print(f"  ok  fmt={src.fmt} field={src.text_field} "
              f"→ 读出 {n} 篇,平均 {chars // n} 字符")
        return True
    except Exception as e:                                     # noqa: BLE001
        print(f"  !! 读取失败:{type(e).__name__}: {str(e)[:120]}")
        return False


def download_one(src: source.Source, workers: int, dry: bool) -> bool:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    dest = src.dir()
    endpoint = source.HF_ENDPOINT or None
    rtype = _repo_type(src)

    if src.kind == "url":
        # 普通 URL,不走 HF。给拼写任务的英文词表用 —— 那份表不在 HF 上。
        # **repo_id 存 URL,part_glob 存落地文件名。**
        import urllib.request
        target = dest / src.part_glob
        if target.exists():
            print(f"  已有 {target}")
            return True
        print(f"  下载 {src.repo_id} → {target}")
        if not dry:
            dest.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(src.repo_id, tmp)
            tmp.rename(target)          # 下完才改名,断了不会留半个文件冒充成品
        return True

    if src.kind == "hf-snapshot":
        print(f"  整仓下载 {src.repo_id} → {dest}")
        if not dry:
            snapshot_download(repo_id=src.repo_id, repo_type=rtype,
                              local_dir=str(dest), endpoint=endpoint)
        return True

    api = HfApi(endpoint=endpoint)
    print(f"  列 {src.repo_id} 文件表 ...")
    try:
        files = sorted(api.list_repo_files(src.repo_id, repo_type=rtype))
    except Exception as e:                                     # noqa: BLE001
        print(f"  !! 取不到文件表:{type(e).__name__}: {e}")
        return False

    pats = [_glob_to_re(p) for p in (src.allow_patterns or [src.part_glob])]
    matched = [f for f in files if any(p.match(f) for p in pats)]
    picked = matched
    if src.n_parts is not None:
        picked = matched[:src.n_parts]
        # 要 N 个但上游只有 M<N 个 —— 说出来。默默给少了会让人以为池子够大。
        if len(matched) < src.n_parts:
            print(f"  注意:n_parts={src.n_parts} 但上游只有 {len(matched)} 个匹配文件,"
                  f"实际取全部 {len(matched)} 个")
    if not picked:
        print(f"  !! 没有文件匹配 {src.allow_patterns or src.part_glob}")
        return False

    todo = [f for f in picked if not (dest / f).exists()]
    print(f"  选中 {len(picked)} 个,已有 {len(picked) - len(todo)},待下 {len(todo)}")
    if dry or not todo:
        return True

    def one(path: str):
        try:
            hf_hub_download(repo_id=src.repo_id, repo_type=rtype,
                            filename=path, local_dir=str(dest),
                            endpoint=endpoint)
            return path, None
        except Exception as e:                                 # noqa: BLE001
            return path, f"{type(e).__name__}: {e}"

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (path, err) in enumerate(pool.map(one, todo), 1):
            if err:
                failed.append((path, err))
                print(f"    [{i}/{len(todo)}] 失败 {path} — {err[:90]}")
            elif i % 10 == 0 or i == len(todo):
                print(f"    [{i}/{len(todo)}] ok")
    if failed:
        print(f"  !! {len(failed)} 个失败,重跑本命令即可续传")
    return not failed


def cmd_list() -> None:
    print(f"落地根 {source.DATA_ROOT}")
    print(f"HF 端点 {source.HF_ENDPOINT or '(官方源)'}\n")
    print(f"{'源':<17}{'lang':<6}{'已有/需要':>12}   {'repo_id'}")
    for src in source.ALL_SOURCES.values():
        need = src.n_parts if src.n_parts is not None else "全量"
        have = src.present()
        mark = "ok" if (src.n_parts and have >= src.n_parts) or (
            src.n_parts is None and have) else "  "
        print(f"{src.name:<17}{src.lang:<6}{f'{have}/{need}':>12} {mark} {src.repo_id}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", help="源名,留空配合 --all/--pretrain")
    p.add_argument("--list", action="store_true", help="只看状态")
    p.add_argument("--all", action="store_true")
    p.add_argument("--pretrain", action="store_true", help="只下预训练语料")
    p.add_argument("--dry", action="store_true", help="只看要下什么")
    p.add_argument("--probe", action="store_true",
                   help="每个源只下第一个文件并真读一遍,验 glob/fmt/field")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--endpoint", default=None,
                   help='覆盖 HF_ENDPOINT;传 "" 走官方源')
    a = p.parse_args()

    if a.endpoint is not None:
        source.HF_ENDPOINT = a.endpoint

    if a.list or not (a.names or a.all or a.pretrain or a.probe):
        cmd_list()
        return 0

    if a.probe and not a.names:
        a.all = True

    if a.all:
        todo = list(source.ALL_SOURCES.values())
    elif a.pretrain:
        todo = list(source.PRETRAIN_SOURCES.values())
    else:
        todo = [source.get(n) for n in a.names]

    ok, bad = True, []
    for src in todo:
        print(f"\n=== {src.name}  ({src.repo_id}) ===")
        if src.note and not a.probe:
            print(f"  {src.note}")
        good = probe_one(src, a.dry) if a.probe else download_one(src, a.workers, a.dry)
        if not good:
            bad.append(src.name)
        ok &= good
    if a.probe:
        print(f"\n探针结果:{len(todo) - len(bad)}/{len(todo)} 通过"
              + (f",失败 {bad}" if bad else ""))
    else:
        print("\n" + ("全部完成" if ok else "有失败项 —— 重跑本命令续传"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
