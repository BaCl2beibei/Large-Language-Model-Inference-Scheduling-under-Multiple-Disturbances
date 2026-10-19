# Llumnix 单机扰动实验工具

此模块独立运行在 `/workspace/llumnix-ray` 内，不修改 Llumnix、Ray 或 supervisor 配置。

## 常用命令

```bash
cd /workspace/llumnix-ray
source .venv/bin/activate

# 只读检查
python -m disturbance_exp doctor

# 离线单元测试
PYTHONPATH=. python -m unittest discover -s disturbance_exp/tests

# 校验场景、生成固定随机种子的请求轨迹并预演事件；不会发请求或启动扰动器
python -m disturbance_exp run --scenario disturbance_exp/scenarios/mixed.yaml --dry-run

# 少量真实请求的单机连通性验证
python -m disturbance_exp run --scenario disturbance_exp/scenarios/feasibility_baseline.yaml --execute

# 本机 HTTP 代理延迟验证
python -m disturbance_exp run --scenario disturbance_exp/scenarios/feasibility_network.yaml --execute

# GPU 场景还需显式启用；显存安全门未通过时会记录跳过原因
python -m disturbance_exp run --scenario disturbance_exp/scenarios/gpu_load.yaml --execute --allow-gpu-load

# 从已有结果重建图表和报告
python -m disturbance_exp report --result-dir disturbance_exp/results/实验目录
```

结果目录在 `disturbance_exp/results/`，包括 `manifest.json`、`preflight.json`、请求轨迹、逐请求 CSV、主机指标 CSV、事件日志、汇总 JSON、Markdown 报告和可用图表。每个结果目录使用新的 UTC 时间戳命名，不会覆盖旧结果。

## 扰动和指标口径

- 负载轨迹由固定随机种子生成；支持固定间隔、泊松到达、突发到达、ShareGPT JSON/JSONL/Parquet 及分段 QPS。
- 通过本地 tokenizer 计算请求输入 Token 数，并限制输入和输出 Token 总和不超过场景中的 `max_model_len`。
- `/generate_benchmark` 的 `per_token_latency` 第一项作为服务端 TTFT；仅当逐 Token 延迟条目数与 `num_output_tokens_cf` 一致且输出不少于两个 Token 时才计算 TPOT。
- 兼容 `/generate` 非流式响应，并可选择逐个解析该接口的 NUL 分隔流式 JSON；流式测量标为客户端 TTFT/TPOT，和服务端指标分开。
- `total_latency_ms` 是客户端端到端时间，代理延迟由客户端观测；因此服务端 TTFT 与客户端时延可以分别查看。
- CPU 负载由最多两个独立低优先级子进程产生，默认每个进程占空比不超过 50%。
- GPU 负载由小矩阵乘法产生，限制申请显存且要求 NVML 显存余量达到安全门槛；不得用大显存申请逼近 OOM。
- 网络代理仅监听回环地址。它对 API 请求做延迟或连接级失败模拟；失败值不是 TCP packet loss。
- `fault` 仅模拟 API 层失败，不停止后端、Worker、Ray Head、API Server 或 supervisor。
- 事件结束、异常和 Ctrl+C 路径都会清理本实验自己的 CPU/GPU 进程与代理，并执行服务健康复查。

## 当前限制

Network Namespace/veth/`tc netem` 需要 `CAP_NET_ADMIN`，环境不具备时报告 `UNSUPPORTED`；本版本只实现回环 HTTP 代理。

真实 Worker/实例故障恢复、KV Cache 内部块/抢占指标、跨节点 RayRPC 扰动和调度/迁移指标需要多实例 Llumnix 环境。当前不会伪造这些数据。单机长短请求混合只能制造请求级 KV Cache 压力，不能证明跨实例 KV 状态迁移。

当前容器的 `/workspace` 未挂载持久卷；实例 recycle/destroy 后结果和代码可能丢失。请将需要保留的代码/结果同步到外部 Git 或持久卷。

当前分析器按单次运行生成汇总和时间图；多次重复运行的置信区间、跨场景自动对比与 QPS 等级校准尚未实现。ShareGPT 与 Parquet 读取已提供代码路径，但本轮可行性测试使用短固定提示词，没有实际跑 ShareGPT 负载。
