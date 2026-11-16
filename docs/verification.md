# 本次改造验证记录

日期：2026-11-16。此记录只描述工程验证，不提供真实商品准确率。

| 检查 | 结果 |
| --- | --- |
| Python 测试 | `python -m pytest tests -q`，7 项通过 |
| 真实预训练模型 | `scripts.verify_openclip` 通过；本地 OpenCLIP 图像和文本输出为 512 维归一化向量 |
| 完整 ML/API 流程 | `scripts.smoke_visual_pipeline` 通过；真实 OpenCLIP + 合成图形，完成权重更新、留出对比、索引构建及 HTTP 查询 |
| Flutter 静态检查 | Flutter 3.50.0 下无 error / warning，81 项 info（主要为 material/withOpacity 弃用提示）；未将 info 当作阻断项 |
| Flutter 编译 | `flutter build bundle --debug --no-pub --target-platform android-arm64` 成功 |
| GitHub 发布历史 | 以当前代码快照建立独立提交历史，保留 README 中的上游来源说明；配置模板不含密钥，旧提交历史已在本地备份 |

本机 Python 3.12.13、PyTorch 2.15.0+cpu、OpenCLIP 4.2.2、FAISS 1.16.0。未验证 CUDA 运行、Android 真机拍照、原云端识别接口或公开服务部署。

完整流程测试使用 8 个合成身份、每个 3 个不同图形视角。它只能检验数据、训练、模型加载、索引和接口之间是否连通；不能用其命中率填写简历。测试产物留在本地 `backend/data/smoke-v1`，不进入 Git。正式数据需要独立商品照片、来源与标签，按照 `visual-retrieval.md` 的协议重新训练及评测。

Flutter 当前完成的是 Dart 代码和资源 bundle 编译，不是签名 APK，也没有完成真机 UI 验收。Windows 桌面插件需要系统支持符号链接；如果 `flutter pub get` 提示该问题，可按 Flutter 提示配置开发者模式，或使用支持符号链接的构建环境。本次没有修改 Windows 的全局开发者模式设置。
