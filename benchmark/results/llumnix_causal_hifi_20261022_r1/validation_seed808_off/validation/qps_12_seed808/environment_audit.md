# 四卡基准环境审计

采集时间：2026-10-22T07:07:59.020760+00:00

## 硬件

- GPU 0: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-f3cc4337-82f1-115b-9058-58cde88a4124`
- GPU 1: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-8ec028dd-cd3d-55c3-daf2-066e44491ddd`
- GPU 2: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-55fbd150-b302-9b67-3a0d-425855d9f47d`
- GPU 3: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-de41c419-b360-1592-a753-f718bdd5b0ee`
- CPU：Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz，80 个逻辑 CPU
- 内存：125 GiB，总可用 114 GiB
- 工作目录盘：31 GiB 可用
- GPU 拓扑：启动前以 `nvidia-smi topo -m` 核对；GPU 3 到 GPU 0–2 经 SYS/NUMA 互联，其余 GPU 对之间为 PHB。

## 软件

- Python：`3.10.12`
- PyTorch：`{'version': '2.4.0+cu121', 'cuda': '12.1', 'cuda_available': True, 'device_count': 4}`
- torch: `2.4.0`
- vllm: `0.6.3.post1`
- ray: `2.47.1`
- aiohttp: `3.14.5`
- matplotlib: `3.10.9`
- numpy: `1.26.4`
- pandas: `2.3.3`
- transformers: `4.46.3`
- tokenizers: `0.20.3`

## 模型

- 路径：`/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/model/Qwen/Qwen2-0.5B`
- `.gitattributes` (1,519 bytes)
- `LICENSE` (11,344 bytes)
- `README.md` (4,026 bytes)
- `config.json` (661 bytes)
- `generation_config.json` (138 bytes)
- `merges.txt` (1,671,839 bytes)
- `model.safetensors` (988,097,824 bytes)
- `tokenizer.json` (7,028,015 bytes)
- `tokenizer_config.json` (1,289 bytes)
- `vocab.json` (2,776,833 bytes)

该审计反映本次四卡服务器，不使用仓库中旧的单卡 RTX 3060 审计作为环境依据。

## GPU topology

```text
	[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	SYS	SYS	0-19,40-59	0		N/A
GPU1	PHB	 X 	SYS	SYS	0-19,40-59	0		N/A
GPU2	SYS	SYS	 X 	PHB	20-39,60-79	1		N/A
GPU3	SYS	SYS	PHB	 X 	20-39,60-79	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
