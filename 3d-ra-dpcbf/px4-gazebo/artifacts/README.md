# 原SITL固件：压缩保存，不自动执行

`px4_sitl_linux_x86_64.gz`来自留存D77构建产物，解压字节与D82初版36份`run_receipt.json`的固件SHA全部一致，也与后期100次归档固件一致。只保存一份，两个运行包共用。

- 解压SHA256：`eeb37b2708909381f88bf56de1c00d69dac7eb0dc8640074cee6c357a83cf67a`。
- 原始大小55,794,616字节；Linux x86_64 SITL程序，不能作为真实飞机固件。
- [逐回执绑定](原固件绑定.json) · [PX4许可证](LICENSE-PX4.txt) · [执行适配源码与构建边界](../firmware_overlay/README.md)。
- `prepare_firmware.py`只解包和校验，已有目标拒绝覆盖；不联网下载、不启动程序。
- 本文件不包含PX4上游rootfs、Gazebo、模型或系统动态库；目标Linux仍需安装/构建这些依赖。
- 本轮没有重新编译该固件；原二进制一致不等于本轮已从源码复现相同构建SHA。
