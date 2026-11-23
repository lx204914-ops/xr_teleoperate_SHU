# 初版36次：D82独立运行包

[返回PX4板块](../README.md) · [原36次结果](../data/第一版36次逐项原表.csv) · [全宽原片](../精选16格对比.md) · [来源与适配](../../reproducibility/运行包补齐核验.md)

这一包来自初版D82原目录，与后期`px4-gazebo/runtime`分开。保留原控制器、四基线接口、执行链、三份场景和冻结参数；必要改动限于路径、输入检查、CPU选择和启动失败清理。不是用D89代码补出的“初版”。

| 场景 | 原文件 | 时长 | 障碍数 |
|---|---|---:|---:|
| ten | `ten_faster_same_layout.npz` | 32 s | 10 |
| fifty | `smooth_N050_S2000.npz` | 24 s | 50 |
| hundred | `smooth_N100_S2000_observe40.npz` | 40 s | 100 |

四方法为`ra`、`park`、`ma_cbf_vo`、`c3bf`，各场景各重复3次，构成历史36次。原巡航设置1.5 m/s、控制周期20 ms、三轴输入限幅1.5 m/s²、跟踪预算0.5 m/s²、到达半径0.25 m，均未改变。CLI原默认巡航仍为1.0，因此复现本批时必须显式写`--cruise 1.5`。

## Linux准备：不执行仿真

需要原生Linux x86_64、PX4上游源码及SITL rootfs/模型、Gazebo Sim 8（原机8.14.0）、gz-transport13/gz-msgs10 Python接口、MAVLink与冻结数值依赖。`CBF_PX4_ROOT`指向PX4-Autopilot检出，原记录提交为`6ea3539157ca358c70a515878b77077af7d4611d`。仅有本包并不包含整个操作系统和上游构建产物。

从`3d-ra-dpcbf`目录准备：

```bash
export CBF_PX4_ROOT=/实际路径/PX4-Autopilot
export CBF_PYTHON=/实际路径/venv/bin/python
# 从本机允许的CPU中选择控制核，并为后台任务保留其他物理核。
export CBF_CONTROL_CPU=本机核编号
python px4-gazebo/prepare_firmware.py --runtime first36
cmake -S px4-gazebo/first36/runtime -B px4-gazebo/first36/runtime/build/native
cmake --build px4-gazebo/first36/runtime/build/native
```

固件命令只解包、不执行，已有文件拒绝覆盖。提供的原固件解压后SHA256为`eeb37b2708909381f88bf56de1c00d69dac7eb0dc8640074cee6c357a83cf67a`，与初版36份原回执逐项一致。来源、许可证和未重新编译的边界见[原固件](../artifacts/README.md)。CMake只构建原`RaObstacleMotion.cc`和`periodic_publish75.cc`两个依赖库；构建配方沿用原pkg-config依赖，不更改C++源码。本轮Windows未执行Linux构建。

## 先检查入口

```bash
"$CBF_PYTHON" px4-gazebo/first36/runtime/code/run_continuous.py --name d82_ten_ra_new --test ten --scene px4-gazebo/first36/runtime/scenes/ten_faster_same_layout.npz --method ra --duration 32 --cruise 1.5 --execution acceleration --cpu "$CBF_CONTROL_CPU" --check-only
```

这条命令只报告缺少的本机依赖、场景、固件和CPU设置，**不导入飞行引擎、不修改PX4配置、不启动SITL**。需要以后手动执行新单例时，移除`--check-only`；必须使用新`--name`。另外两种场景按上表同时改`--test`、`--scene`和`--duration`。可加`--gui --native-view`启用原生显示；显示环境由本机登录会话提供，不沿用旧机器HOME或Xauthority。

## 已修复的启动问题

- 缺固件、插件或场景时先失败，不先写日志配置或世界软链接。
- 启动失败和正常退出都清理本次自己的进程，恢复原`logger_topics.txt`和原自有世界软链接；外部路径或并发改动保留并报错。
- 世界文件写在本次新输出目录，旧结果拒绝覆盖。
- 本机CPU用`--cpu`或`CBF_CONTROL_CPU`指定；历史CPU测试只作为来源记录，不自动选择新机核号。
- 原36次全部`initialize=false`。不借用其他批次初始化参数；若显式用`--initialize`，必须另给原始`--initialization-file`，缺失会在启动前报错。

**核验范围：**源码/配置/场景来源核对、入口帮助、失败清理的模拟测试已做；本轮未重新运行36次、未启动SITL、未验证目标Linux闭环或硬实时。原`source_lock.json`和`deployment_freeze.json`是历史冻结清单，保留当时状态，不作为修改后包的即时校验器；当前逐文件绑定见发布清单。历史`comparison_manager.py`等调度脚本仅供溯源，不是本次推荐的便携运行入口。
