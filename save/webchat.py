"""给发布的 chat 模型(v8/Summer-Chat)搭一个本机 WebService,自己在浏览器里聊。

    python -m save.webchat --model_path output/summer05b_v8
    python -m save.webchat --model_path output/summer05b_v8 --port 8899

打开 http://<这台机器的局域网 IP>:8899 就能用——FastAPI 里 host=0.0.0.0,
同一局域网的其他设备也能连,不止 localhost。

## 为什么不能用 `vllm serve`

`vllm serve` 是 OpenAI 兼容服务,自己把文本转成 token —— 但这个词表是
81903 的 piece 词表,不是 HF 标准格式,vLLM 认不了。所以这里自己写一个
薄薄的 FastAPI 服务:用 `PieceTokenizerWrapper` 编码/解码,vLLM 只管
拿 token id 推理(`skip_tokenizer_init=True`,和 `example_vllm.py`
同一个路子)。

## 默认参数:temperature=0.6, repetition_penalty=1.15

**不是贪心。** 贪心(temperature=0)只是跑分协议,为了和历史数字可比;
真正部署给人用,`repetition_penalty=1.15` 是实测的甜蜜点(中文长回答成功率
2%→25~30%,`prepare/stoprate.py` 的说明里有完整推导),nanochat 自己的
`chat_cli.py`(交互式)也不用贪心,默认 temperature=0.6。这里跟它对齐。

## BOS 约定

`bos_bestfit` 打包训出来的模型(S0B 起,v8 也是)每一行都以 `<bos>` 开头。
`tokenizer.py` 的 `apply_chat_template` 已经处理了这个 —— 每次请求把
完整对话历史重新走一遍 `apply_chat_template`,不用自己拼 BOS。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INDEX_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Summer-Chat</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif;
    background: #f5f5f7; color: #1a1a1a; display: flex; flex-direction: column;
    height: 100vh;
  }
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1a; color: #e8e8e8; }
    .bubble.user { background: #2b5fd9 !important; }
    .bubble.assistant { background: #2a2a2e !important; color: #e8e8e8 !important; }
    header { background: #202023 !important; border-color: #333 !important; }
    #input { background: #2a2a2e !important; color: #e8e8e8 !important; border-color: #444 !important; }
  }
  header {
    padding: 14px 20px; background: #fff; border-bottom: 1px solid #e0e0e0;
    font-weight: 600; display: flex; align-items: center; gap: 10px;
  }
  header .tag { font-weight: 400; font-size: 12px; color: #888; }
  #log { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .row { display: flex; }
  .row.user { justify-content: flex-end; }
  .bubble {
    max-width: 72%; padding: 10px 14px; border-radius: 14px; line-height: 1.5;
    white-space: pre-wrap; word-break: break-word; font-size: 14.5px;
  }
  .bubble.user { background: #3b78ff; color: #fff; border-bottom-right-radius: 4px; }
  .bubble.assistant { background: #fff; border: 1px solid #e5e5e5; border-bottom-left-radius: 4px; }
  .meta { font-size: 11px; color: #999; margin-top: 4px; }
  form { display: flex; gap: 8px; padding: 14px 20px; border-top: 1px solid #e0e0e0; }
  #input {
    flex: 1; padding: 10px 14px; border-radius: 20px; border: 1px solid #ccc;
    font-size: 14.5px; outline: none;
  }
  button {
    padding: 10px 20px; border-radius: 20px; border: none; background: #3b78ff;
    color: #fff; font-size: 14.5px; cursor: pointer;
  }
  button:disabled { background: #aaa; cursor: default; }
</style>
</head>
<body>
<header>Summer-Chat <span class="tag" id="modeltag"></span></header>
<div id="log"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="说点什么…" />
  <button id="send">发送</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let history = [];

fetch('/info').then(r => r.json()).then(d => {
  document.getElementById('modeltag').textContent = d.model;
});

function addBubble(role, text) {
  const row = document.createElement('div');
  row.className = 'row ' + role;
  const b = document.createElement('div');
  b.className = 'bubble ' + role;
  b.textContent = text;
  row.appendChild(b);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return b;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendBtn.disabled = true;
  addBubble('user', text);
  history.push({role: 'user', content: text});
  const bubble = addBubble('assistant', '…');
  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: history}),
    });
    const data = await resp.json();
    bubble.textContent = data.reply;
    history.push({role: 'assistant', content: data.reply});
  } catch (err) {
    bubble.textContent = '(出错了: ' + err + ')';
  }
  sendBtn.disabled = false;
  input.focus();
});
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--temperature", type=float, default=0.6,
                   help="0.6 对齐 nanochat chat_cli.py 的交互默认值,不是贪心")
    p.add_argument("--repetition_penalty", type=float, default=1.15,
                   help="实测甜蜜点,见 prepare/stoprate.py 的说明")
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    args = p.parse_args()

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    import uvicorn

    from prepare.tokenizer import PieceTokenizerWrapper

    print(f"Loading tokenizer from {args.model_path}...")
    tok = PieceTokenizerWrapper(args.model_path)

    print(f"Loading vLLM engine from {args.model_path}...")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model_path, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              skip_tokenizer_init=True, trust_remote_code=True,
              enforce_eager=False)

    class ChatRequest(BaseModel):
        messages: list[dict]

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/info")
    def info():
        return {"model": Path(args.model_path).name}

    @app.post("/chat")
    def chat(req: ChatRequest):
        ids = tok.apply_chat_template(req.messages, tokenize=True,
                                      add_generation_prompt=True)
        sampling = SamplingParams(
            temperature=args.temperature, max_tokens=args.max_new_tokens,
            stop_token_ids=tok.stop_token_ids,
            repetition_penalty=args.repetition_penalty)
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling,
                           use_tqdm=False)
        gen = list(out[0].outputs[0].token_ids)
        reply = tok.decode(gen, skip_special_tokens=True)
        return {"reply": reply}

    print(f"\n打开浏览器访问:  http://<这台机器的局域网 IP>:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
