#!/usr/bin/env bash
# 装 Summer 唯一的 C++ 依赖:PieceTokenizer(piece 分词器 + 81903 词表)。
# 仓库不在本地就从 GitHub clone —— 整个 Summer 只依赖 Hugging Face 和 GitHub。
#
#   bash prepare/install_deps.sh              # clone + 编译 + 装 + 校验
#   bash prepare/install_deps.sh clone        # 只 clone,不编译
#   bash prepare/install_deps.sh --verify     # 不装,只跑行为校验
#
# 仓库默认 clone 到 SUMMER_DEPS_DIR(默认 <仓库>/deps),已有就 git pull。
# 想用本机既有的 checkout:SUMMER_DEPS_DIR=~/src bash prepare/install_deps.sh
#
# ## 词表不留副本
#
# 81903 词表和中文分词词典都在 PieceTokenizer 仓库里,本仓库**不存副本**:
#
#   deps/PieceTokenizer/save/Summer-Tokenizer.pt        piece 模型
#   deps/PieceTokenizer/save/Summer-Tokenizer.dict.txt  中文分词词典
#
# 2026-07-27 用 sha256 核过:这两个文件与 v18 checkpoint 里的 piece.model /
# dict.txt 逐字节相同,也与 HF 上 Ismantic/Qwen3-1.7B-Base-ReTok 里的相同。
# 所以指向 clone 就能逐位复现 v18。
#
# ## 装完必须校验
#
# **PieceTokenizer 的编码一旦变了,81903 词表和已发布模型的 embedding 就对不上,
# 但代码不会报错** —— 只会悄悄训出/推出垃圾结果。所以:
#
#   1. 重建之前先 `python test/capture_baseline.py` 抓基线
#   2. 重建之后 `python test/test_tokenizer.py` 比对
#
# 顺序反了就失去意义:基线是用来发现「重建把行为改了」的。
#
# 需要 cmake、C++17 编译器、git。用 uv pip,这个 venv 里没有 pip。
set -euo pipefail

PY=${SUMMER_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
PY=${PY:-python3}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR=${SUMMER_DEPS_DIR:-$REPO_ROOT/deps}

PIECE_REPO=$DEPS_DIR/PieceTokenizer
PIECE_URL=https://github.com/Ismantic/PieceTokenizer.git

TARGET="${1:-all}"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

git_clone_or_pull() {
    local url=$1 dst=$2
    if [[ -d "$dst/.git" ]]; then
        echo "  已有 $dst,git pull"
        git -C "$dst" pull --ff-only
    else
        mkdir -p "$(dirname "$dst")"
        git clone --depth 1 "$url" "$dst"
    fi
    git -C "$dst" log -1 --format='  %h %ci  %s'
}

verify() {
    log "校验分词器行为"
    if [[ -f "$REPO_ROOT/test/test_tokenizer.py" ]]; then
        "$PY" "$REPO_ROOT/test/test_tokenizer.py"
    else
        echo "  test/test_tokenizer.py 不在,跳过"
    fi
}

if [[ "$TARGET" == "--verify" ]]; then
    verify
    exit 0
fi

log "PieceTokenizer"
git_clone_or_pull "$PIECE_URL" "$PIECE_REPO"

if [[ "$TARGET" == "clone" ]]; then
    echo "  只 clone,不编译(--> bash prepare/install_deps.sh 装全套)"
    exit 0
fi

log "编译并安装 PieceTokenizer"
command -v cmake >/dev/null || { echo "缺 cmake"; exit 1; }
uv pip install --python "$PY" -e "$PIECE_REPO"

log "确认词表文件在位"
for f in Summer-Tokenizer.pt Summer-Tokenizer.dict.txt; do
    if [[ -f "$PIECE_REPO/save/$f" ]]; then
        printf '  %-30s %s\n' "$f" "$(sha256sum "$PIECE_REPO/save/$f" | cut -c1-16)…"
    else
        echo "  !! 缺 $PIECE_REPO/save/$f —— 词表不在,后面全都跑不了"
        exit 1
    fi
done

verify
log "完成"
