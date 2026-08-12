# 文档索引

| | |
|---|---|
| [`WHY.md`](WHY.md) | **先看这个。** 各处设计选择的理由,以及改错了不报错的地方 |
| [`PRETRAIN.md`](PRETRAIN.md) | 从零重做一遍的全流程 |
| [`DATA_FORMAT.md`](DATA_FORMAT.md) | **数据格式约定** —— 特殊 token 各自什么含义、对话怎么打包、掩码怎么算。改错了不报错的那一类 |
| [`POSTTRAIN.md`](POSTTRAIN.md) | **后训练日志** —— midtrain/SFT 每一版改了什么、量到了什么,以及对齐 nanochat 还差哪些项 |
| [`eval/pipeline.md`](eval/pipeline.md) | 评测栈 |
| [`reports/summer-0.5b-pretrain.md`](reports/summer-0.5b-pretrain.md) | **Summer-0.5B 底座怎么训出来的** —— 随机初始化 → S0 单语 11.8B token → S1 平行语料退火。下游结果在 Interpreter 那边 |
| [`reports/`](reports/) | 其余是历史实验记录(**写于四层改造之前**,路径已过期) |

分工:层 README 讲「这一层解决什么问题」,`make help` 讲怎么跑,
`WHY.md` 讲为什么。**同一件事只写在一处。**

原始日志和机器产出的指标放 `output/` 和 `eval_results/`,不放这里。
