# gpu_load_single_gpu 实验报告

- 状态：`complete`
- 单机运行时长：12 秒
- 请求成功率：1.000
- 成功吞吐：0.250 请求/秒
- 输出 Token 吞吐：1.000 Token/秒
- 服务端 TTFT：mean 94.191, P50 97.186, P95 99.730, P99 99.956 毫秒
- 服务端 TPOT：mean 9.432, P50 9.734, P95 9.748, P99 9.749 毫秒
- 客户端端到端时延：mean 126.258, P50 127.757, P95 132.543, P99 132.968 毫秒
- SLO 违约率：0.000
- GPU：NVIDIA GeForce RTX 3060；空闲显存 2314 MiB

## 扰动事件

- gpu_compute @ 1.00232s：skipped — {"supported": false, "reason": "only 2314 MiB free; require 3072 MiB safety reserve", "free_mib": 2314}

## 图表

- `ttft_timeline.png`
- `tpot_timeline.png`
- `throughput_timeline.png`
- `gpu_utilization.png`
- `disturbance_timeline.png`

## 解释范围

网络代理造成的等待只表示 API 代理层额外时延，不代表 TCP packet loss 或跨节点网络故障。没有服务端可观测数据时，相关指标保持为空。
