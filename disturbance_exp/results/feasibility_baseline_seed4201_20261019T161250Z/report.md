# feasibility_baseline 实验报告

- 状态：`complete`
- 单机运行时长：20 秒
- 请求成功率：1.000
- 成功吞吐：0.050 请求/秒
- 输出 Token 吞吐：0.150 Token/秒
- 服务端 TTFT：mean 80.249, P50 80.249, P95 80.249, P99 80.249 毫秒
- 服务端 TPOT：mean 8.245, P50 8.245, P95 8.245, P99 8.245 毫秒
- 客户端端到端时延：mean 103.994, P50 103.994, P95 103.994, P99 103.994 毫秒
- SLO 违约率：0.000
- GPU：NVIDIA GeForce RTX 3060；空闲显存 2314 MiB

## 扰动事件

- 无

## 图表

- `ttft_timeline.png`
- `tpot_timeline.png`
- `throughput_timeline.png`
- `gpu_utilization.png`

## 解释范围

网络代理造成的等待只表示 API 代理层额外时延，不代表 TCP packet loss 或跨节点网络故障。没有服务端可观测数据时，相关指标保持为空。
