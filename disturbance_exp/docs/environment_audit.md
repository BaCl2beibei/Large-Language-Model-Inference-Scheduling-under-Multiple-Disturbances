# 环境只读检查记录

检查日期：2026-10-20（Asia/Shanghai）  
目标：`/workspace/llumnix-ray`  
Git HEAD：`3fb6c03`

## 仓库和运行服务

- `disturbance_exp/` 检查前不存在。
- 现有 Git 工作区有修改的 `llumnix/version.py`，以及未跟踪的 `datasets/`、`model/`。开发时保留这些已有内容。
- supervisor 服务 `llumnix-v0` 为 `RUNNING`，PID 5107。
- `GET /health` 返回 HTTP 200；`GET /is_ready` 返回 `true`、HTTP 200。
- 服务日志由 supervisor 定向到 `/dev/stdout`，所以 `supervisorctl tail` 不提供可读日志文件。`/var/log/portal/llumnix-v0.log` 存在；检查时没有匹配到 `ERROR`、`Traceback`、`Exception` 或 `CRITICAL` 行。
- 服务实现位于 `llumnix/entrypoints/vllm/api_server.py`。`POST /generate_benchmark` 接受 `prompt` 与 SamplingParams 字段，返回 `generated_text`、`num_output_tokens_cf`、`per_token_latency`。时间数组按每个生成步骤记录毫秒间隔；第一项是 TTFT 候选值，后续项用于 TPOT。官方 `benchmark/benchmark_serving.py` 使用相同 endpoint 和 `prompt/max_tokens` 请求字段。
- `/generate` 非流式返回 `text` 数组；流式模式以 NUL 分隔 JSON，不是 SSE。

## 计算和依赖

- 主机有 36 个逻辑 CPU；内存约 62 GiB 总量、48 GiB 可用。
- GPU：NVIDIA GeForce RTX 3060，12 GiB；检查时已用约 9.4 GiB、空闲约 2.3 GiB，GPU 利用率为 0%。
- 服务虚拟环境为 Python 3.10.22。`aiohttp`、PyYAML、psutil、pynvml、PyTorch、Ray、vLLM、Transformers、NumPy、Matplotlib 和 PyArrow 可导入；`pytest` 不可导入。本项目自带的 `.venv` 没有 pip 模块，但服务所需库可导入，因此不安装或升级任何依赖，单元测试使用标准库 `unittest`。
- `pynvml` 可用；GPU 指标优先从 NVML 读取。

## 扰动能力和边界

- `tc`、`ip` 可用，`stress-ng` 不可用。
- 有效 capability 集合 `CapEff=0xa80405fb` 未包含 `CAP_NET_ADMIN`；`ip netns list` 为空。`lo` 当前只有默认 `noqueue` qdisc。未尝试创建 namespace/veth 或修改网络规则。
- 仅执行回环 HTTP 代理模式。GPU 注入器设置 3 GiB 最低显存余量，因此本次 2.3 GiB 的空闲量不足以启动 GPU 计算扰动。
- `/workspace` 没有持久卷挂载，recycle/destroy 后的代码和结果不保证保留。
- 检查期间没有停止、重启或修改 Llumnix/supervisor 服务，也没有清理主机网络规则。
