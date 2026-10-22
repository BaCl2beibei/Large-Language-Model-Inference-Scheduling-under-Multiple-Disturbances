从仓库根目录执行（CPU，无服务启动）：

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/repairs/bridge_r1_r7/verify_cpu.py
```

复跑会改变本目录日志及证据 hash；必须核验新结果并重新签发 CPU gate，再交独立审查。finalize_repair.py 会更新本次 CPU gate、共享指针、源码 manifest 和命令 hash，不能用于将 failed 独立审查改为 passed。
