# feasibility_generate_stream 实验报告

- 状态：`complete`
- 单机运行时长：8 秒
- 请求成功率：1.000
- 成功吞吐：0.125 请求/秒
- 输出 Token 吞吐：0.375 Token/秒
- 服务端 TTFT：n/a 毫秒
- 服务端 TPOT：n/a 毫秒
- 客户端流式 TTFT：mean 85.875, P50 85.875, P95 85.875, P99 85.875 毫秒
- 客户端流式 TPOT：mean 8.694, P50 8.694, P95 8.694, P99 8.694 毫秒
- 客户端端到端时延：mean 103.500, P50 103.500, P95 103.500, P99 103.500 毫秒
- SLO 违约率：0.000
- GPU：NVIDIA GeForce RTX 3060；空闲显存 2314 MiB

## 扰动事件

- 无

## 图表

- `throughput_timeline.png`
- `gpu_utilization.png`

## 解释范围

网络代理造成的等待只表示 API 代理层额外时延，不代表 TCP packet loss 或跨节点网络故障。没有服务端可观测数据时，相关指标保持为空。
