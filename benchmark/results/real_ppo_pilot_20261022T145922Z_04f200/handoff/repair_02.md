# repair_02：Ray 端口冲突修复

状态：incomplete（配置修正已落盘；真机 smoke 尚未重验）。

在 rl_scheduler/configs/real_pilot.yaml 中，将 service.ray_port 从 16379 改为 20000。失败日志显示 Ray 默认 worker_ports 范围为 10002–19999，原端口落在该范围中；新端口不重叠，且检查时没有监听者。YAML 解析通过。

未运行测试套件、未启动 GPU、未重跑 telemetry。05 的失败证据保持不变。新配置 SHA256：6e7ad9eb64f0a226275122b1d2f23cc644590605b1018d2343f4b85069612882。

配置 hash 改变后，04 review、离线 gates 和旧 commands/05_smoke.json 中绑定的 hash 都不能用于新配置的 live 验收。需要重新生成并审查绑定新 hash 的材料，再单独授权运行 telemetry。

详见本文件对应的 repair_02.json。
