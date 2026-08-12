"""PieceTokenizer 的 HuggingFace 风格外壳。

## dict.txt 是必需的,不是可选的

没有中文分词词典时,**中文的 token id 会变** —— 不只是慢。2026-07-27 实测
(同一 piece.model,带/不带 dict):

    中文 22 字     0.169ms → 1.060ms   6.3×    id 序列不同(长度都是 22)
    中文 ×20       3.289ms → 21.125ms  6.4×    id 序列不同(长度都是 440)
    英文           0.110ms → 0.110ms   1.0×    完全相同(dict 只管中文切分)

**round-trip 正确会掩盖这个问题**:两种模式 decode 都能还原原文,看起来没事,
但喂给模型的 id 已经不是训练时那套了。所以这里缺 dict 直接报错,不静默降级
(旧版本是 `else: load(model_file)`,不报错 —— 那是个真隐患)。

确实不需要 dict 的场合(比如只查 vocab_size)传 `require_dict=False`。

## 词表从哪来

本仓库**不存词表副本**。规范位置是 clone 的 PieceTokenizer:

    deps/PieceTokenizer/save/Summer-Tokenizer.pt
    deps/PieceTokenizer/save/Summer-Tokenizer.dict.txt

用 `resolve_assets()` 反查(经 `piece_tokenizer.__file__`)。checkpoint 目录里
另有一份 piece.model + dict.txt,与上面逐字节相同(sha256 已核),训练/评测时
直接从 checkpoint 目录加载即可。
"""
import os
import json
import piece_tokenizer as pt

PIECE_MODEL_NAME = "Summer-Tokenizer.pt"
PIECE_DICT_NAME = "Summer-Tokenizer.dict.txt"


def resolve_assets():
    """从已安装的 piece_tokenizer 反查词表文件,返回 (piece_model, cn_dict)。

    本仓库不留副本,所以要经 `piece_tokenizer.__file__` 找到 clone 的仓库根,
    再取 `save/` 下那两个文件。找不到就报错 —— 报错好过静默用错词表。

    `SUMMER_PIECE_MODEL` / `SUMMER_PIECE_DICT` 可以覆盖(与 BERTc 的
    `BERTC_PIECE_MODEL` 同一套做法)。
    """
    env_m = os.environ.get("SUMMER_PIECE_MODEL")
    env_d = os.environ.get("SUMMER_PIECE_DICT")
    if env_m and env_d:
        return env_m, env_d

    root = os.path.dirname(os.path.abspath(pt.__file__))
    tried = []
    for cand in (root, os.path.dirname(root)):
        save = os.path.join(cand, "save")
        model, cn_dict = (os.path.join(save, PIECE_MODEL_NAME),
                          os.path.join(save, PIECE_DICT_NAME))
        tried.append(save)
        if os.path.exists(model) and os.path.exists(cn_dict):
            return model, cn_dict
    raise FileNotFoundError(
        f"找不到 {PIECE_MODEL_NAME} / {PIECE_DICT_NAME}。找过:{tried}\n"
        f"  先跑 `bash prepare/install_deps.sh` clone 并安装 PieceTokenizer。")


# checkpoint 目录里词表文件的名字。**上游名优先。**
#
# `Summer-Tokenizer.pt` / `.dict.txt` 与 PieceTokenizer 仓库 `save/` 下的文件
# 同名 —— 这样从任何一个 checkpoint 都能一眼看出词表出自哪里,不用靠 sha256
# 去反推(BERTc 一直是这么做的,它的发布包里就叫 `BERTc-Tokenizer.pt`)。
#
# `piece.model` / `dict.txt` 是改造之前的名字。已发布的
# `Ismantic/Qwen3-1.7B-Base-ReTok` 和 v18 的 checkpoint 用的都是旧名,所以
# **保留为回退**,不然那些目录一个都加载不了。新产出统一用上游名。
_MODEL_NAMES = (PIECE_MODEL_NAME, "piece.model", "piece_mt.model")
_DICT_NAMES = (PIECE_DICT_NAME, "dict.txt")


def _first_in(model_dir, names):
    for n in names:
        p = os.path.join(model_dir, n)
        if os.path.exists(p):
            return p
    return None


def has_piece_vocab(model_dir) -> bool:
    """这个目录是 piece 词表的模型吗?

    评测入口靠它决定用 PieceTokenizerWrapper 还是 AutoTokenizer。**不要各自
    硬编码文件名** —— 新产出用上游名 `Summer-Tokenizer.pt`,旧的用
    `piece.model`,漏一个就会静默退回 AutoTokenizer,而 AutoTokenizer 对这个
    词表走不通,结果是错的。
    """
    return _first_in(model_dir, _MODEL_NAMES) is not None


class PieceTokenizerWrapper:
    def __init__(self, model_dir, require_dict=True):
        """从模型目录加载。

        词表文件按 `_MODEL_NAMES` / `_DICT_NAMES` 的顺序找 —— 上游名优先,
        旧名回退。
        """
        self._tok = pt.Tokenizer()

        model_file = _first_in(model_dir, _MODEL_NAMES)
        if model_file is None:
            raise FileNotFoundError(
                f"{model_dir} 里找不到词表。试过:{list(_MODEL_NAMES)}")

        cn_dict = _first_in(model_dir, _DICT_NAMES)
        if cn_dict is not None:
            self._tok.load(model_file, cn_dict)
        elif require_dict:
            raise FileNotFoundError(
                f"{model_dir} 里没有中文分词词典(试过 {list(_DICT_NAMES)})。"
                f"缺了它中文的 token id 会变(不只是慢),而且 decode 照样能"
                f"还原原文、不会报错。\n"
                f"  从 checkpoint 或 PieceTokenizer 的 save/{PIECE_DICT_NAME} "
                f"拷一份过来;确实不需要就传 require_dict=False。")
        else:
            self._tok.load(model_file)
        self.piece_model_path = model_file
        self.cn_dict_path = cn_dict

        # Load token mapping
        mapping_file = os.path.join(model_dir, "token_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file) as f:
                mapping = json.load(f)
            self.pad_token_id = mapping["pad_id"]
            self.bos_token_id = mapping["bos_id"]
            self.eos_token_id = mapping["eos_id"]
            self.user_token_id = mapping.get("user_id")
            self.assistant_token_id = mapping.get("assistant_id")
            self.system_token_id = mapping.get("system_id")
            # **81902 改作 `<end>`:回答结束的专用停止符。**
            #
            # 注意**词表里那个 piece 的字面名字仍然是 `<system>`** —— 改名要重建
            # 分词器,不值得。`<end>` 是代码和文档里的名字,指的就是 81902。
            #
            # 对话数据里几乎没有 system 消息,而「回答结束」急需一个没有历史
            # 包袱的 token。<eos> 在预训练里是文档分隔符(占 0.094%,S0 全程约
            # 1100 万次),学到的含义是「一篇结束,后面还有下一篇」—— 拿它表示
            # 「到此为止」是在跟先验打架,实测 SFT 后贪心只有 50-55% 能停下。
            #
            # nanochat 用专用的 <|assistant_end|>,它的 <|bos|> 只管文档边界,
            # 两者永不混用。我们词表已满(81903)加不了新 token,所以复用 81902
            # —— 它在预训练里出现 **0 次**,和新加一个等效。
            self.end_token_id = self.system_token_id
        else:
            # Fallback to piece_to_id lookups
            self.bos_token_id = self._tok.piece_to_id("<s>")
            self.eos_token_id = self._tok.piece_to_id("</s>")
            self.pad_token_id = self._tok.piece_to_id("<pad>")
            self.user_token_id = self._tok.piece_to_id("<user>")
            self.assistant_token_id = self._tok.piece_to_id("<assistant>")
            self.system_token_id = self._tok.piece_to_id("<system>")
            self.end_token_id = self.system_token_id
            if self.pad_token_id < 0:
                self.pad_token_id = 0

    @property
    def vocab_size(self):
        return self._tok.vocab_size()

    @property
    def stop_token_ids(self):
        """生成时该停下的 token 集合。

        **同时认 `<end>` 和 `<eos>`,是有意的向后兼容。** 新的对话格式用专用的
        `<end>`(81902);但 Summer-0.5B-S0 / -S1 和 ReTok 那条线的已发布模型是
        用 `<eos>` 训的,它们不认识 `<end>`。推理侧统一用这个集合,一份代码同时
        伺候两代模型 —— 多认一个停止符不会让新模型提前停(它不会去生成 `<eos>`,
        那个 token 在它的对话数据里出现 0 次)。
        """
        ids = [self.eos_token_id]
        end = getattr(self, "end_token_id", None)
        if end is not None and end != self.eos_token_id:
            ids.append(end)
        return ids

    def encode(self, text, add_special_tokens=False):
        ids = self._tok.encode_as_ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            special = {self.bos_token_id, self.eos_token_id, self.pad_token_id,
                       self.user_token_id, self.assistant_token_id, self.system_token_id}
            ids = [i for i in ids if i not in special]
        try:
            return self._tok.decode(ids)
        except UnicodeDecodeError:
            # Model emitted byte-fallback piece(s) that don't form valid UTF-8.
            # Per-piece fallback: keep ids that decode cleanly, drop the rest.
            parts = []
            for i in ids:
                try:
                    parts.append(self._tok.id_to_piece(i))
                except UnicodeDecodeError:
                    continue
            return "".join(parts).replace("▁", " ")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        """Build chat-formatted token sequence from messages."""
        ids = []

        # Check for system message
        start = 0
        if messages and messages[0]["role"] == "system":
            ids.append(self.bos_token_id)
            ids.extend(self._tok.encode_as_ids(messages[0]["content"]))
            ids.append(self.system_token_id)
            start = 1
        else:
            ids.append(self.bos_token_id)

        for msg in messages[start:]:
            if msg["role"] == "user":
                ids.append(self.user_token_id)
                ids.extend(self._tok.encode_as_ids(msg["content"]))
            elif msg["role"] == "assistant":
                ids.append(self.assistant_token_id)
                ids.extend(self._tok.encode_as_ids(msg["content"]))
                # **回答结束用 answer_end,不用 eos** —— 见 __init__ 里的理由。
                ids.append(self.end_token_id)

        if add_generation_prompt:
            ids.append(self.assistant_token_id)

        if tokenize:
            return ids
        else:
            # Return as string (rarely needed)
            return self._tok.decode(ids)

    def save_pretrained(self, output_dir):
        """Save tokenizer files to directory (for checkpoint saving)."""
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        # Copy piece.model
        # 从自己加载时的实际路径拷,不猜文件名
        if self.piece_model_path:
            shutil.copy2(self.piece_model_path,
                         os.path.join(output_dir, PIECE_MODEL_NAME))
        if self.cn_dict_path:
            shutil.copy2(self.cn_dict_path,
                         os.path.join(output_dir, PIECE_DICT_NAME))
        # Save mapping
        mapping = {
            "bos_id": self.bos_token_id,
            "eos_id": self.eos_token_id,
            "pad_id": self.pad_token_id,
            "user_id": self.user_token_id,
            "assistant_id": self.assistant_token_id,
            "system_id": self.system_token_id,
            # 同一个 id 的第二个名字:回答结束(专用停止符,见 tokenizer.py)
            "end_id": self.system_token_id,
        }
        with open(os.path.join(output_dir, "token_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)
