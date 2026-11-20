# 复现、版本与数据来源

## 不启动仿真也能完成的核验

```bash
cd 3d-ra-dpcbf
python reproducibility/verify_release.py
python reproducibility/render_results.py
```

第一条只读取文件校验值、CSV和Python语法；第二条只读取原记录生成中文图表，均不导入控制器、不启动PX4、不修改已有仿真数据。绘图需要NumPy、Matplotlib，以及微软雅黑或Noto Sans CJK SC等中文字体。

## 结果和代码怎样对应

| 结果 | 保存的源码版本 | 注意事项 |
|---|---|---|
| 条件绕行6000次 | `numerical/code/conditional_6000` | 原冻结参数、场景生成器、核心与基线；不等同于后期目标推进策略 |
| C03瓶颈七组 | `numerical/code/c03` | 后期推进与方向保持；提供绘图所需紧凑轨迹，未承诺整套历史搜索调度可直接迁移 |
| PyBullet | `pybullet/code` | 历史PD/HOCBF/软RA扩展，不是最新版核心；上游工作树指针失效，精确上游提交仍缺核 |
| RotorPy六次 | `rotorpy/code` | 接受的共享执行实现；原场景按入口预期SHA核验 |
| PX4主100次 | `px4-gazebo/runtime` | 原D89运行链及D83父版本绑定；不使用后来未接受候选替换 |

`来源与校验清单.csv`保存相对归档标识、来源SHA256与发布SHA256；不公开个人盘符、登录配置或聊天记录。`发布文件校验.csv`覆盖交付文件。源码保持原算法；若为了可移植性调整路径，由`路径适配清单.csv`逐项列明。JSON/NPZ/机器CSV中的程序字段和方法标识保留，中文展示表单独提供。历史配置中的个人主目录替换为归档标识，冻结清单仅作为历史校验依据，不冒充当前文件路径。

## 数值实验

原环境：Python3.11.5、NumPy1.26.4、SciPy1.11.1、Numba0.61.2、OSQP1.1.1、Clarabel0.11.1、Matplotlib3.7.2；冻结文件在`numerical/code/conditional_6000/inputs/dependency_versions.json`。本次只做离线核验，不通过重跑改变历史统计。

原生成函数保留`max_attempts=1000`默认值；正式补齐任务在外层调用时明确传入10000，不修改内部放置次数、种子或筛选。`numerical/generate_formal_scene.py`展示这一绑定，并核对保存的场景tape与接受序号；必须显式请求生成，默认不运行。

便携入口位于`numerical/run_example.py`，只绑定原场景读取与输出路径，不改变控制公式。先用`--help`查看；必须明确加`--run`才开始单例仿真，输出目录若存在则拒绝覆盖。完整6000次的调度恢复程序不随本次发布启动。

## RotorPy

上游参考提交`40b09697a6a1596d404a67f459941d46a39eedd3`，四障碍场景SHA256为`45c591518e0f1e3d1fe3d176a16cc6b4cf1c26d895a80ee60248d189afac5b7e`。使用`code/shared_execution_20261030/run_quadrotor.py`，保留其父目录布局与冻结控制器。具体命令见板块说明。

## PX4＋Gazebo的复现限制

运行链包含PX4状态反馈、Gazebo物理状态观察、执行预留、原生显示及20 ms控制周期。原机Gazebo为8.14.0，对应Python接口`gz.transport13`/`gz.msgs10`；还需要MAVLink、C++插件和独立PX4 SITL构建。

归档使用固件SHA256：`eeb37b2708909381f88bf56de1c00d69dac7eb0dc8640074cee6c357a83cf67a`。它已与主100次所在归档中的二进制核对。仓库只提供归档执行适配源码，不上传二进制；本轮没有从这些源码重新构建并复现该二进制SHA，因此不能声称编译同源性已经重新通过。原始`candidate_`前缀仅是归档文件名，不代表后来D91—D93候选。

本地上游源码核验记录给出的PX4提交为`6ea3539157ca358c70a515878b77077af7d4611d`。固件适配还需要按原构建步骤处理附加角速度前馈，见[固件来源说明](../px4-gazebo/firmware_overlay/README.md)，不能直接把六份候选命名文件全部覆盖后声称等于接受版本。

入口`runtime/code/run_continuous.py`需要Linux和完整仿真环境。`platform_session.py`的PX4目录与Python路径可通过`CBF_PX4_ROOT`、`CBF_PYTHON`设置；此处是发布副本的路径适配，控制公式未改变。原生插件、固件配置与上游基准仍需在目标Linux机上核对。本次Windows离线检查不等于Linux闭环复现通过。

100次运行的全部结果、81个原生GIF和19张停止截图均随仓库发布，逐项提供路径与原件SHA256。首页仅展示精选录像，全部媒体在飞控板块的100次录像总目录中打开；短时停止截图没有伪装成完整录像。

## 媒体与中文说明

录制动画保留原字节，界面、方法名和坐标单位可能仍有英文；标题、图注、说明和新增统计图使用中文。没有通过生成式工具替换仿真轨迹或合成成功画面。图的PNG用于浏览，SVG用于矢量查看；两者由同一份原记录生成。
