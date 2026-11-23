# C03：原场景七组单例入口

[返回数值实验](../../README.md) · [复现层级](../../../reproducibility/版本与复现状态.md)

本目录保留已发布的冻结控制器，对应[七组结果表](../../data/C03七组机制消融.csv)保存在数值数据目录。补回原`bindings47.py`、原`sparse_projection.py`、C03场景及协议，用相对路径装配依赖；不修改控制算法或参数，不重新搜索场景。

从`3d-ra-dpcbf`目录只检查：

```bash
python numerical/code/c03/run_case.py --method full --out outputs/c03_full_new --check-only
```

七组选择：`full`、`no_detour`、`no_reselection`、`no_progress_guard`、`fixed_radius`、`no_transverse`、`fixed_strength`。

检查模式会真正导入原模块、核对相对来源清单、读取原场景与配置，并创建原工厂绑定；**不会实例化控制器、求解QP或推进仿真**。七组均完成本轮导入检查。冻结配置和算法文件保持原字节，`inputs/package_sources.json`绑定当前包内实际文件。

若以后手动新算单例，将`--check-only`改为`--run`，输出根目录必须不存在；原时长仍为40 s，场景仍为`tetra12_C03_S8203`、种子8203。`no_reselection`关闭段内重选，保留首次选向和方向运输。绕行暂停时，锁定方向仍按原执行器的逻辑更新。本轮没有执行科学单例，原19.09 s等结果未更新。

这是原C03的手动单例入口，不包含历史大批调度、搜索或自动续跑。输入原件和路径适配见[来源清单](../../../reproducibility/来源与校验清单.csv)。
