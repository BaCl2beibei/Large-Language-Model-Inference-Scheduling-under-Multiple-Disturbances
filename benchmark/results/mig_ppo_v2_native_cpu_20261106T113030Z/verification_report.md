# Windows / Linux CPU训练包交付验证

- ZIP：`mig_ppo_v2_cpu_windows_linux.zip`，269,996 bytes。
- SHA256：`78ed24e4499835f8632723e065833c77fc7a76e4bf3c3268564569119644c012`。
- Linux：全新Python3.10 CPU虚拟环境，torch2.4.0+cpu、numpy1.26.4；从ZIP解压至含中文和空格的路径后执行。
- doctor、13环境真实profile smoke、PPO更新与保存重载均通过；smoke 32.49秒。
- Actor样本1510，Critic/GAE样本1526；forced Stay不参与Actor。
- 200秒到期48请求显式timeout测试通过。
- 实际训练1次更新后恢复至2次，共8回合；预算120秒在恢复时正确继承。仅诊断，未使用评估选择策略。
- 108项原有相关测试及11项便携测试通过，共119项。后者包含采样/更新中断重做及留出策略冻结。
- 诊断策略导出成功；service导入屏蔽检查通过，CPU构建没有CUDA runtime。
- Windows原生尚未执行；用户需在本机运行安装、doctor及smoke。跨系统checkpoint实际搬迁尚未执行。
- formal_ppo_ready=false，仍缺独立真实验证；当前ZIP不包含完整研究归档。

Windows从包内README_WINDOWS.md执行PowerShell脚本；Linux直接运行相同Python入口。无需WSL。人工步骤见rl_scheduler/docs/p4_local_cpu_migration_prompt.md。
