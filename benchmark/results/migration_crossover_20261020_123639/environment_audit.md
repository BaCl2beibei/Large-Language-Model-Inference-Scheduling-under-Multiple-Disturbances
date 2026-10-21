# 四卡基准环境审计

采集时间：2026-10-20T12:36:42.666646+00:00

## 硬件

- GPU 0: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-02df511e-f4af-9773-43de-a96fefa95e2d`
- GPU 1: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-4935a3fa-02ef-f938-9c60-7caa6066906d`
- GPU 2: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-3079b9d7-adf5-8d6f-8e4c-317a293f6434`
- GPU 3: NVIDIA GeForce RTX 3060, 12288 MiB, UUID `GPU-03d4ef83-d5d0-acca-a303-17432267bdac`
- CPU：Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz，56 个逻辑 CPU
- 内存：62 GiB，总可用 57 GiB
- 工作目录盘：64 GiB 可用
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
