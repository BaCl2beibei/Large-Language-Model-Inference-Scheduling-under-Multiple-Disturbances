# 单机可行性验证结果

测试日期：2026-10-20（Asia/Shanghai）  
机器：RTX 3060 12 GiB，36 个逻辑 CPU   
服务：`llumnix-v0`，`127.0.0.1:8080`

## 结果

| 场景 | 请求结果 | 观测结果 | 说明 |
|---|---:|---|---|
| `/generate_benchmark` 基线 | 1/1 成功 | E2E 104.0 ms；服务端 TTFT 80.2 ms；TPOT 8.2 ms | 前后健康检查通过 |
| 本机网络代理 | 1/1 成功 | E2E 175.7 ms；服务端 TTFT 73.9 ms；TPOT 8.3 ms | 设置请求/响应各 40 ms；本次端到端增加约 71.7 ms |
| CPU 轻扰动 | 3/3 成功 | 1 worker、25% 占空比运行 5 秒；系统 CPU 平均 3.58% | worker 自动停止，无遗留子进程 |
| GPU 扰动安全门 | 3/3 成功 | 空闲显存 2314 MiB，低于 3072 MiB 门槛 | 按设计跳过 GPU 计算，没有启动 CUDA 扰动进程 |
| `/generate` 非流式 | 1/1 成功 | E2E 122.8 ms | 不提供服务端逐 Token 指标，按空值记录 |
| `/generate` NUL 流式 | 1/1 成功 | E2E 103.5 ms；客户端 TTFT 85.9 ms；TPOT 8.7 ms | NUL 分隔 JSON 解析成功 |

上表每个服务场景只有 1～3 个请求，用于验证请求格式、采集链路、代理延迟和资源清理。样本量不足以作性能比较、容量校准或因果结论。

离线验证：`compileall` 通过；标准库 `unittest` 共 12 项通过；正常、负载、扰动及基线/代理场景的 dry-run 通过。两个 `/generate` 兼容场景直接以单请求方式完成了执行验证。

## 安全和恢复

- 所有执行场景的实验后 `/health` 和 `/is_ready` 检查通过。
- supervisor 服务持续为 `RUNNING`；未重启或修改 Llumnix、Ray、supervisor。
- 没有安装或更新依赖。
- `GPU` 注入因显存安全门跳过；`Network Namespace` 因缺少 `CAP_NET_ADMIN` 不支持。
- 网络代理只监听回环地址。延迟模拟是 API 代理等待，不表示 TCP packet loss。

## 结果目录

每次运行的原始 CSV、事件 JSONL、汇总 JSON 和报告均保存在 `disturbance_exp/results/`，其中可行性目录名称以 `feasibility_` 开头。
