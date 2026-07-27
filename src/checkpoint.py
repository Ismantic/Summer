"""safetensors 读写。**只依赖 torch。**

格式很简单,自己实现比拉一个依赖划算:

    [0:8]      u64 小端,头部 JSON 的字节数 N
    [8:8+N]    JSON:{名字: {dtype, shape, data_offsets:[起, 止]}, "__metadata__": {...}}
    [8+N:]     张量数据区,data_offsets 是相对这里的偏移

## 为什么不用 safetensors 库

`src/` 只依赖 torch —— 这条约束的意义是让读者能看见每一步在做什么。
safetensors 的格式本身只有二十来行代码,藏进依赖里反而看不见了。

## 不能改错的地方

**读进来的 key 必须与 HF 的 Qwen3ForCausalLM 完全一致。** 已发布的
`Ismantic/Qwen3-1.7B-Base-ReTok` 是标准 HF 命名,key 对不上不会报错 ——
`load_state_dict(strict=False)` 会静默跳过,模型照样随机初始化跑起来、
loss 照样降,只是那份权重白搭了。所以这里默认 `strict=True`。

**tie_word_embeddings 的模型没有 `lm_head.weight`。** v18 的 checkpoint 就是
这样:310 个张量,只有 `model.embed_tokens.weight` 和 `model.norm.weight` 加
28 层 ×11。加载时要把 lm_head 绑回 embed,不能指望文件里有。
"""
import json
import struct
from pathlib import Path

import torch

# safetensors 的 dtype 名 → torch dtype
_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}
_NAMES = {v: k for k, v in _DTYPES.items()}


def read_header(path) -> tuple[dict, int]:
    """返回 (头部 dict, 数据区起始偏移)。不读张量,几微秒。"""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def load_safetensors(path, device="cpu") -> dict[str, torch.Tensor]:
    """读成 {名字: 张量}。

    按 data_offsets 顺序 seek+read,每个张量一份可写内存。
    不用 mmap:`torch.frombuffer` 不接受只读缓冲区,套一层 `bytearray` 反正
    要拷一份,mmap 的零拷贝就没了 —— 还得处理 memoryview 挂着导出引用、
    `mm.close()` 抛 BufferError 的麻烦。
    """
    path = Path(path)
    hdr, base = read_header(path)
    hdr.pop("__metadata__", None)

    # 按文件内偏移排序后顺序读,避免来回 seek
    items = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
    out = {}
    with open(path, "rb") as f:
        for name, spec in items:
            lo, hi = spec["data_offsets"]
            f.seek(base + lo)
            buf = bytearray(hi - lo)
            if f.readinto(buf) != hi - lo:
                raise EOFError(f"{path}: 读 {name} 时文件提前结束")
            t = torch.frombuffer(buf, dtype=_DTYPES[spec["dtype"]])
            out[name] = t.reshape(spec["shape"]).to(device)
    return out


def save_safetensors(tensors: dict[str, torch.Tensor], path, metadata=None) -> None:
    """写出。张量按名字排序,同样的输入产生同样的字节。

    共享存储的张量(比如 tie 的 embed / lm_head)要**调用方自己去重** ——
    这里不猜。写两份不会报错,只是文件大一倍、加载时可能对不上 tie 的预期。
    """
    hdr, blobs, off = {}, [], 0
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        if t.dtype not in _NAMES:
            raise TypeError(f"{name}: 不支持的 dtype {t.dtype}")
        b = t.view(torch.uint8).reshape(-1).numpy().tobytes()
        hdr[name] = {"dtype": _NAMES[t.dtype], "shape": list(t.shape),
                     "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    # 头部按 8 字节对齐,张量数据区才能对齐
    pad = (-len(raw)) % 8
    raw += b" " * pad

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


def load_sharded(model_dir, device="cpu") -> dict[str, torch.Tensor]:
    """加载一个模型目录:单文件 `model.safetensors`,或分片 + index.json。"""
    model_dir = Path(model_dir)
    single = model_dir / "model.safetensors"
    if single.exists():
        return load_safetensors(single, device)

    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        out = {}
        for s in shards:
            out.update(load_safetensors(model_dir / s, device))
        return out
    raise FileNotFoundError(f"{model_dir} 下既没有 model.safetensors 也没有 index.json")
