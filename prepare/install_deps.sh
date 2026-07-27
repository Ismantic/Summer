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

# **必须是 editable 装法。** `prepare/tokenizer.py:resolve_assets()` 靠
# `piece_tokenizer.__file__` 反查 clone 里的 save/ 找词表 —— 要是把 .so 拷进
# site-packages,反查就断了,而 import 照样成功、只在读词表时才报错。
#
# pybind11 要装在目标 venv 里:上游 CMakeLists 是 `find_package(pybind11 REQUIRED)`,
# 找不到就 CMake Error,而 setup.py 把它吞成一句「.so 不存在或不是普通文件」——
# 看着像构建产物没生成,其实是缺依赖。
uv pip install --python "$PY" pybind11 >/dev/null

# 上游 setup.py 传的是 `-DPYTHON_EXECUTABLE`,**新版 CMake 的 FindPython 不认
# 这个名字**(要 `Python_EXECUTABLE`),于是它去挑系统 python。系统 python 和
# 目标 venv 同版本时撞对了、看不出来;不同版本就产出错 ABI tag 的 .so,
# setuptools 找不到它要的那个名字,报错同样是「.so 不存在」。
# 所以这里先试标准装法,ABI 不匹配就自己 cmake 再挂 .pth —— 等价于 editable。
if ! uv pip install --python "$PY" --no-build-isolation -e "$PIECE_REPO" 2>/dev/null; then
    ABI=$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')
    echo "  标准 editable 装法失败(大概是 CMake 挑了系统 python)"
    echo "  改为手动 cmake,目标 ABI $ABI"
    BUILD=$(mktemp -d)
    cmake -S "$PIECE_REPO" -B "$BUILD" -DBUILD_PYTHON=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$("$PY" -c 'import sys;print(sys.executable)')" \
        -Dpybind11_DIR="$("$PY" -c 'import pybind11;print(pybind11.get_cmake_dir())')" \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$BUILD/out" >/dev/null
    cmake --build "$BUILD" --target piece_tokenizer -j >/dev/null
    [[ -f "$BUILD/out/piece_tokenizer$ABI" ]] || {
        echo "  !! 还是没产出 piece_tokenizer$ABI —— 手动查 $BUILD"; exit 1; }
    cp "$BUILD/out/piece_tokenizer$ABI" "$PIECE_REPO/"
    # .pth 把 clone 加进 sys.path —— 与 editable 同效,resolve_assets() 能反查
    SP=$("$PY" -c 'import site;print(site.getsitepackages()[0])')
    echo "$PIECE_REPO" > "$SP/piece_tokenizer_clone.pth"
    rm -rf "$BUILD"
    echo "  装好了:$PIECE_REPO/piece_tokenizer$ABI + $SP/piece_tokenizer_clone.pth"
fi

# 反查必须真的通 —— 这是「装好了」的判据,import 成功不是。
"$PY" -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from prepare.tokenizer import resolve_assets
m, d = resolve_assets(); print(f'  反查词表 ok: {m}')" || {
    echo "  !! import 成功但反查词表失败 —— 大概装成了非 editable"; exit 1; }

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
