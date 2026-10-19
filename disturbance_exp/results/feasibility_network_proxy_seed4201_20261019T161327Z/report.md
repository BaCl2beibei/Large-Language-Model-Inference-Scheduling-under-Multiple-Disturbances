# feasibility_network_proxy 实验报告

- 状态：`complete`
- 单机运行时长：20 秒
- 请求成功率：1.000
- 成功吞吐：0.050 请求/秒
- 输出 Token 吞吐：0.150 Token/秒
- 服务端 TTFT：mean 73.873, P50 73.873, P95 73.873, P99 73.873 毫秒
- 服务端 TPOT：mean 8.269, P50 8.269, P95 8.269, P99 8.269 毫秒
- 客户端端到端时延：mean 175.712, P50 175.712, P95 175.712, P99 175.712 毫秒
- SLO 违约率：0.000
- GPU：NVIDIA GeForce RTX 3060；空闲显存 2314 MiB

## 扰动事件

- network_proxy @ 0.008031s：active — {"active": true, "bind_host": "127.0.0.1", "port": 36691, "connections": 0, "injected_failures": 0, "classification": "proxy_api_failure_not_packet_loss"}
- network_proxy @ 10.010264s：stopped — duration elapsed

## 图表

- `ttft_timeline.png`
- `tpot_timeline.png`
- `throughput_timeline.png`
- `gpu_utilization.png`
- `disturbance_timeline.png`

## 解释范围

网络代理造成的等待只表示 API 代理层额外时延，不代表 TCP packet loss 或跨节点网络故障。没有服务端可观测数据时，相关指标保持为空。
