"""Summer 数据源注册表 + 路径解析。

**这个仓库的目标是:只靠 Hugging Face 和 GitHub 就能从零重做 ReTok。**
所以这里登记的每个源都必须有公开出处,不允许指向本机的既有目录 —— 一旦允许
「本地已有就跳过下载」,别人克隆下来跑不通,而自己永远发现不了。

下载落在 `SUMMER_DATA_ROOT`(默认仓库内 `data/downloads/`),gitignore。
本机若已有语料,用符号链接指过去,**不要把路径写回代码**。

**需要 HF 登录**:`BAAI/CCI3-HQ` 是 gated 数据集,得先在它的 HF 页面上接受条款,
再 `huggingface-cli login`(或设 `HF_TOKEN`)。其余十个源都是公开直下。

## 与 v18 实跑的关系

v18(当前 SOTA)读的是 本机 a6000 那台机器上的本地副本,那批文件多数已经过
格式转换(parquet → jsonl 等),且两个源换过上游。下表的 `repo_id` 是**公开
等价物**,不保证与 v18 当初吃进去的字节完全相同。

这不影响复现 v18 —— v18 的 checkpoint 已经存在并发布,`test/test_reproduce_sota.py`
钉的是 checkpoint 的行为。本表管的是**将来从零重建语料**。

溯源依据(2026-07-27 逐个核过):

| 源 | 依据 | 强度 |
|---|---|---|
| Gutenberg | 本地 parquet 的 blob sha256 与 HF LFS 元数据逐位相同 | **确证** |
| FineWebEdu | 第一条记录与 HF `sample-10BT` 逐字段相同,含 `id=<urn:uuid:0d8a309d-…>` | **确证** |
| Wikipedia_EN | 第一条是 `id=12 / Anarchism`,即 `wikimedia/wikipedia` 20231101.en 的首条 | **确证** |
| CCI3-HQ / CN_FineWeb_Edu / Cosmopedia | HF 上的文件路径与本地 glob 完全一致 | **确证** |
| SkyPile | schema 一致(单 `text` 列);首条不同 —— 本地 42 个 parquet 是从上游 437 个 jsonl 转换来的,顺序对不上,无法逐位比 | 强推定 |
| Wikipedia_CN | **原始出处不明**。本地字段是 `id/text/source`、`source: wikipedia.zh2307`,不是 wikimedia 格式,像是从某个聚合语料里切出来的 | 已替换 |
| CC_EN / CC_CN | **原始不可得**。v18 读的 `c2-en`/`c2-cn` 是自己跑 pipeline 处理 CC-MAIN-2023-50 的产物 —— 数据内嵌 `file_path` 指向已不存在的 `/mnt/data/now/DataPipeline`。对照 FineWebEdu 的 `file_path: s3://commoncrawl/…` 才是上游原版,这反证了 c2 是本地产物 | 已替换 |

替换的三个(Wikipedia_EN/CN 换 finewiki,CC_EN/CC_CN 换 fineweb/fineweb-2)是
**有意的**:目标是「从零能重建一份等价语料」,不是「逐字节还原 v18 那批文件」。
后者已经不可能 —— 那两个源的生产代码不存在了。

**注意 `CC_EN`/`CC_CN` 在 v18 代码里叫 `C4_EN`/`C4_CN`,那是 v17 的遗留名字。**
v17 确实用 `allenai/c4`(`en.noblocklist/c4-train.*.json.gz`),v18 换成了
CC-MAIN-2023-50 的切片却没改变量名。看代码的人会以为还是 C4 —— 属于「改错了
不报错」,这里一并正名。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 下载落地根。默认在仓库内,换机器不用改任何东西。
DATA_ROOT = Path(os.environ.get("SUMMER_DATA_ROOT",
                                str(REPO_ROOT / "data" / "downloads")))
# clone 的 C++ 依赖(PieceTokenizer)。跟 prepare/install_deps.sh 用同一个变量。
DEPS_DIR = Path(os.environ.get("SUMMER_DEPS_DIR", str(REPO_ROOT / "deps")))
# HF 镜像。置空走官方源。
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


@dataclass
class Source:
    """一个可下载的数据源。

    kind:
      hf           按 allow_patterns 选文件、按 n_parts 截断
      hf-snapshot  整仓下载(小仓库)

    repo_type 单独指定 "dataset" / "model",与 kind 正交。
    """
    name: str
    kind: str
    repo_id: str
    subdir: str                       # 相对 DATA_ROOT
    part_glob: str                    # 相对 dir(),用来数 present()
    # HF 上的仓库类型。**不要从 kind 推断** —— 一个模型仓库照样可以整仓下载
    # (kind="hf-snapshot"),按数据集去请求会 404。
    repo_type: str = "dataset"
    n_parts: int | None = None        # None = 全量
    allow_patterns: list[str] = field(default_factory=list)
    lang: str = ""                    # en / zh / ""
    # 下面两个给 prepare/encode_corpus.py 用:怎么读、读哪个字段。
    # 注意这是**上游 HF 原始格式**;v18 当初读的本地副本多数已转成 jsonl。
    fmt: str = "parquet"              # parquet / jsonl / jsonl_gz
    text_field: str = "text"
    note: str = ""

    def dir(self) -> Path:
        return DATA_ROOT / self.subdir

    def parts(self) -> list[Path]:
        return sorted(self.dir().glob(self.part_glob))

    def present(self) -> int:
        return len(self.parts())


# ---------------------------------------------------------------- 预训练语料
#
# v18 实跑的两段混合(权重见 prepare/encode_corpus.py):
#   main   1B token   EN 0.663 / CN 0.336   8 源
#   anneal 200M token EN 0.60  / CN 0.40    6 源
#
# ## n_parts 怎么定的
#
# **池子 ≠ 消耗量。** 编码器从池子里流式采样,只取到权重对应的 token 数就停。
# 各源实际吃掉多少:
#
#   FineWebEdu     0.316×1B + 0.20×200M = 356M      Wikipedia_CN  0.063×1B + 0.10×200M =  83M
#   Wikipedia_EN   0.189×1B + 0.15×200M = 219M      CC_EN         0.053×1B             =  53M
#   SkyPile        0.126×1B             = 126M      Cosmopedia            0.25×200M    =  50M
#   CCI3-HQ        0.126×1B + 0.05×200M = 136M      CN_FineWeb_Edu        0.25×200M    =  50M
#   Gutenberg      0.105×1B             = 105M      CC_CN         0.021×1B             =  21M
#
# 消耗量都在 1 亿 token 上下,而上游单个文件动辄几亿 token —— 比如 SkyPile 一个
# jsonl 就有约 319M token,光它一个就够喂满 126M 的份额。
#
# 所以 n_parts 取的是**池子规模**,对齐 v18 当初本地有多少可采样,不是对齐消耗量。
# 池子大只影响采样多样性。**想省磁盘就把 n_parts 调小** —— 只要池子 ≥ 消耗量,
# 训练照跑,区别是抽样的来源集中一些。
#
# SkyPile 的份数是按**字符量**折算的,不是按文件数:本地 42 个 parquet(压缩)
# = 1241 万篇 / 约 16.6G 字符;上游 437 个 jsonl(不压缩)平均 1.52GB,取前 47 个
# ≈ 53GB ≈ 16.7G 字符。按份数配会差一个数量级,因为两边格式不同。

PRETRAIN_SOURCES = {
    # ---------------- EN ----------------
    "FineWebEdu": Source(
        name="FineWebEdu", kind="hf", repo_id="HuggingFaceFW/fineweb-edu",
        subdir="fineweb-edu", part_glob="sample/10BT/*.parquet", n_parts=20,
        allow_patterns=["sample/10BT/*.parquet"], lang="en",
        note="英文教育向网页。只用 sample/10BT 子集(全量 3038 文件)。"
             "v18 读的是转成 jsonl 的 sample-10bt-*.json。",
    ),
    "Wikipedia_EN": Source(
        name="Wikipedia_EN", kind="hf", repo_id="HuggingFaceFW/finewiki",
        subdir="finewiki", part_glob="data/enwiki/*.parquet", n_parts=None,
        allow_patterns=["data/enwiki/*.parquet"], lang="en",
        note="英文维基,15 parquet / 37.7GB。与 BERTc 用同一个源。"
             "v18 用的是 wikimedia/wikipedia 20231101.en 转的 jsonl。",
    ),
    "Gutenberg": Source(
        name="Gutenberg", kind="hf",
        repo_id="BEE-spoke-data/gutenberg-en-v1-clean",
        subdir="gutenberg", part_glob="data/train-*.parquet", n_parts=None,
        allow_patterns=["data/train-*.parquet"], lang="en",
        note="英文公版书,7 parquet / 9930 篇。**sha256 确证**:本地 "
             "train-00000-of-00007.parquet 的 blob sha256 与 HF LFS 元数据相同。",
    ),
    "CC_EN": Source(
        name="CC_EN", kind="hf", repo_id="HuggingFaceFW/fineweb",
        subdir="fineweb", part_glob="data/CC-MAIN-2023-50/*.parquet", n_parts=10,
        allow_patterns=["data/CC-MAIN-2023-50/*.parquet"], lang="en",
        note="CommonCrawl 2023-50 英文清洗切片(全量 350 parquet)。"
             "**v18 代码里叫 C4_EN,是 v17 的遗留名 —— 那批数据不是 C4。**",
    ),
    "Cosmopedia": Source(
        name="Cosmopedia", kind="hf", repo_id="HuggingFaceTB/cosmopedia-v2",
        subdir="cosmopedia-v2", part_glob="cosmopedia-v2/train-*.parquet",
        n_parts=20, allow_patterns=["cosmopedia-v2/train-*.parquet"], lang="en",
        note="合成教科书/故事(全量 104 parquet / 114GB)。只 anneal 段用,权重 0.25。",
    ),

    # ---------------- CN ----------------
    "SkyPile": Source(
        name="SkyPile", kind="hf", repo_id="Skywork/SkyPile-150B",
        subdir="SkyPile", part_glob="data/*.jsonl", n_parts=47,
        allow_patterns=["data/*.jsonl"], lang="zh", fmt="jsonl",
        note="中文网页(全量 437 jsonl / 665GB)。47 个 ≈ 53GB ≈ 16.7G 字符,"
             "与 v18 本地那 42 个 parquet(16.6G 字符)量级相当 —— 按字符折算,"
             "不是按份数。实际只吃 126M token,一个文件就够;下 47 个是为了"
             "对齐当初的采样多样性,省磁盘可以调小。",
    ),
    "Wikipedia_CN": Source(
        name="Wikipedia_CN", kind="hf", repo_id="HuggingFaceFW/finewiki",
        subdir="finewiki", part_glob="data/zhwiki/*.parquet", n_parts=None,
        allow_patterns=["data/zhwiki/*.parquet"], lang="zh",
        note="中文维基,5 parquet / 5.5GB。与 BERTc 用同一个源。"
             "v18 那份池子只有 ~150M token,所以 main 段特意降了权(0.063)。",
    ),
    "CCI3-HQ": Source(
        name="CCI3-HQ", kind="hf", repo_id="BAAI/CCI3-HQ",
        subdir="CCI3-HQ", part_glob="data/part_*.jsonl", n_parts=10,
        allow_patterns=["data/part_*.jsonl"], lang="zh", fmt="jsonl",
        note="中文高质量网页(全量 679 文件)。HF 路径与 v18 本地 glob 完全一致。"
             "**这个源是 gated 的** —— 要先在 HF 页面上接受条款,再 "
             "`huggingface-cli login`(或设 HF_TOKEN),否则下载会 401。",
    ),
    "CC_CN": Source(
        name="CC_CN", kind="hf", repo_id="HuggingFaceFW/fineweb-2",
        subdir="fineweb-2", part_glob="data/cmn_Hani/train/*.parquet", n_parts=4,
        allow_patterns=["data/cmn_Hani/train/*.parquet"], lang="zh",
        note="CommonCrawl 中文清洗切片。**v18 代码里叫 C4_CN,同样是遗留名。**",
    ),
    "CN_FineWeb_Edu": Source(
        name="CN_FineWeb_Edu", kind="hf",
        repo_id="opencsg/Fineweb-Edu-Chinese-V2.2",
        subdir="Chinese-FineWeb-Edu-V2.2", part_glob="4_5/*.parquet", n_parts=200,
        allow_patterns=["4_5/*.parquet"], lang="zh",
        note="中文教育向。只用打分 4_5 的子集(全量 9745)。"
             "HF 路径与 v18 本地 glob 完全一致。只 anneal 段用,权重 0.25。",
    ),
}


# ---------------------------------------------------------------- 评测资产
#
# WMT22/23 测试集不在这里 —— `prepare/_translate_common.py:load_pair()` 走
# sacrebleu 自带的下载器(`sacrebleu.get_source_file`),不需要本仓库管。
# mono 六个 benchmark 由 `data/prefetch_eval_datasets.py` 从 HF 拉。

EVAL_SOURCES = {
    "qwen_base": Source(
        name="qwen_base", kind="hf-snapshot", repo_id="Qwen/Qwen3-1.7B-Base",
        repo_type="model",
        subdir="Qwen3-1.7B-Base", part_glob="*.safetensors", n_parts=None,
        note="ReTok 的起点,也是所有对照的基线模型(~3.4GB)。"
             "`make -C prepare retok` 和评测 base 都要它。"
             "**这是必需输入,不是可选** —— 之前没登记是个疏漏。",
    ),
    "retok_model": Source(
        name="retok_model", kind="hf-snapshot",
        repo_id="Ismantic/Qwen3-1.7B-Base-ReTok",
        repo_type="model",
        subdir="Qwen3-1.7B-Base-ReTok", part_glob="*.safetensors", n_parts=None,
        note="**已发布的成品**(~3.1GB)。想直接用而不重训的人只要这个。"
             "登记在这里而不是让文档写 `huggingface-cli download` —— 后者不清"
             "代理,在本机走不通 hf-mirror(LocalEntryNotFoundError)。"
             "`data/download.py` 在 import 阶段就摘掉代理。",
    ),
    "comet": Source(
        name="comet", kind="hf-snapshot", repo_id="Unbabel/wmt22-comet-da",
        repo_type="model",
        subdir="comet-wmt22-da", part_glob="checkpoints/*.ckpt", n_parts=None,
        note="COMET 打分模型(~400MB)。prepare/translate.py --compute_comet 要用。",
    ),
}


ALL_SOURCES = {**PRETRAIN_SOURCES, **EVAL_SOURCES}


def get(name: str) -> Source:
    if name not in ALL_SOURCES:
        raise KeyError(f"未登记的数据源:{name}。已登记:{sorted(ALL_SOURCES)}")
    return ALL_SOURCES[name]


def status() -> list[tuple[str, int, int | None, Path]]:
    """(名字, 已下载份数, 需要份数, 落地目录)。"""
    return [(s.name, s.present(), s.n_parts, s.dir()) for s in ALL_SOURCES.values()]
