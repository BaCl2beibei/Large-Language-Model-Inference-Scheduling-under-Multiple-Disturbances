# repair_03：Ray 端口配置与验收材料重绑

已确认 pilot 与 Llumnix 两端都使用 Ray 端口 20000。根据之前日志，16379 与默认 worker 端口区间 10002–19999 冲突。端口当前无监听。smoke spec、四条 05 命令、P1/P2 offline gates、04 review 和 active pointer 已重绑新配置；旧 04/05、旧 gate 与旧命令文件保存在 `repairs/ray_port_collision/prior_20261022T1740/`。

9 项独立审查、15 项修复回归和 6 项 runner 集成 CPU 测试全部通过。没有启动 GPU 服务。05 真机命令最大需要 6060 秒，当前 P0–P3 剩余预算不足；后续只允许跑 dry-run 并交接预算阻塞。
