# cpu_load_single_gpu 实验报告

- 状态：`complete`
- 单机运行时长：12 秒
- 请求成功率：1.000
- 成功吞吐：0.250 请求/秒
- 输出 Token 吞吐：1.000 Token/秒
- 服务端 TTFT：mean 89.062, P50 87.972, P95 91.361, P99 91.662 毫秒
- 服务端 TPOT：mean 8.645, P50 8.617, P95 8.838, P99 8.857 毫秒
- 客户端端到端时延：mean 118.735, P50 118.342, P95 121.208, P99 121.463 毫秒
- SLO 违约率：0.000
- GPU：NVIDIA GeForce RTX 3060；空闲显存 2314 MiB

## 扰动事件

- cpu_compute @ 0.005366s：active — {"active": true, "workers": 1, "pids": [9248]}
- cpu_compute @ 5.026489s：stopped — duration elapsed

## 图表

- `ttft_timeline.png`
- `tpot_timeline.png`
- `throughput_timeline.png`
- `gpu_utilization.png`
- `disturbance_timeline.png`

## 解释范围

网络代理造成的等待只表示 API 代理层额外时延，不代表 TCP packet loss 或跨节点网络故障。没有服务端可观测数据时，相关指标保持为空。
