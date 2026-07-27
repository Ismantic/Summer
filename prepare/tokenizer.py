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
    """
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


class PieceTokenizerWrapper:
    def __init__(self, model_dir, require_dict=True):
        """从模型目录加载(需含 piece.model / dict.txt / token_mapping.json)。"""
        self._tok = pt.Tokenizer()

        # Find the .model file
        model_file = os.path.join(model_dir, "piece.model")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "piece_mt.model")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No piece model found in {model_dir}")

        cn_dict = os.path.join(model_dir, "dict.txt")
        if os.path.exists(cn_dict):
            self._tok.load(model_file, cn_dict)
        elif require_dict:
            raise FileNotFoundError(
                f"{model_dir} 里没有 dict.txt。缺了它中文的 token id 会变"
                f"(不只是慢),而且 decode 照样能还原原文、不会报错。\n"
                f"  从 checkpoint 或 deps/PieceTokenizer/save/{PIECE_DICT_NAME} "
                f"拷一份过来;确实不需要就传 require_dict=False。")
        else:
            self._tok.load(model_file)

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
        else:
            # Fallback to piece_to_id lookups
            self.bos_token_id = self._tok.piece_to_id("<s>")
            self.eos_token_id = self._tok.piece_to_id("</s>")
            self.pad_token_id = self._tok.piece_to_id("<pad>")
            self.user_token_id = self._tok.piece_to_id("<user>")
            self.assistant_token_id = self._tok.piece_to_id("<assistant>")
            self.system_token_id = self._tok.piece_to_id("<system>")
            if self.pad_token_id < 0:
                self.pad_token_id = 0

    @property
    def vocab_size(self):
        return self._tok.vocab_size()

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
                ids.append(self.eos_token_id)

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
        src = os.path.join(os.path.dirname(output_dir), "piece.model")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, "piece.model"))
        # Save mapping
        mapping = {
            "bos_id": self.bos_token_id,
            "eos_id": self.eos_token_id,
            "pad_id": self.pad_token_id,
            "user_id": self.user_token_id,
            "assistant_id": self.assistant_token_id,
            "system_id": self.system_token_id,
        }
        with open(os.path.join(output_dir, "token_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)
