CPU 复现（仓库根目录，不启动服务）：

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest rl_scheduler.tests.test_real_reward rl_scheduler.tests.test_real_opportunity rl_scheduler.tests.test_real_metrics rl_scheduler.tests.test_real_runner.LiveFixtureTests.test_complete_CPU_HTTP_episode_through_real_env_IO_and_ledger -v
```

07 的实际 parser/dry-run argv 在 commands/07_opportunity.json。不要在 06 执行 live argv。准备脚本会在已有 p3_ready 或实验后拒绝重新冻结；复跑测试改变日志时应重新核验 gate hash，而非手改 status。新 P3 CLI 与旧 run_real_ppo.py 分开，旧入口仍不支持 opportunity。
