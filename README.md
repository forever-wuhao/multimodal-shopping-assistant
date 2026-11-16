# Multimodal Shopping Assistant · 多模态智能购物助手

Flutter + FastAPI 的商品图片检索与购物决策项目。新增 **OpenCLIP、FAISS、PyTorch 对比学习适配层与 SQLite 商品索引**，可在本地完成图文编码和检索，无需调用商业识图 API。

基于 [Ryan-wu-web/smart_price_ai](https://github.com/Ryan-wu-web/smart_price_ai) 继续开发，保留原有购物工作流。本仓库以当前代码快照初始化独立提交历史；上游代码的原始作者与历史可在来源仓库查看。当前配置模板不含密钥。

## 能做什么

| 模块 | 当前实现 |
| --- | --- |
| 拍照找同款候选 | OpenCLIP 图像/文本编码，FAISS 检索，按商品合并多视角结果 |
| 筛选和对比 | 品类、品牌、预算、属性硬过滤；低价偏好重排；来源与资料对比 |
| 本地训练 | 冻结 OpenCLIP 主干，训练残差适配层；同款多视角正样本、相似异款难负样本、监督对比损失 |
| 独立评测 | 按商品隔离 train/val/test；比较基线与微调后的 Hit@1、Hit@5、宏平均命中率和查询延迟 |
| Flutter | 首页拍照/相册入口，图文查询、条件调整、候选图卡及 2–3 商品对比 |
| 原购物助手 | 需求确认、预算/参数过滤、偏好排序、多轮条件修改和可追溯解释；使用原有虚构样例库 |

**状态说明：** 本地检索、训练及评测代码已实现；真实商品数据与领域训练权重不随仓库提供。测试夹具不代表实际商品准确率，尚无可报告的真实微调提升。微调训练的是特征适配层，不是从零训练大模型。视觉图库与原聊天的 18 件虚构商品库目前是两条独立链路。

## 快速启动

推荐 Python 3.12。以下为 Windows PowerShell，先进入 `backend`。

```powershell
cd E:\code\smart_price_ai\backend
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
# CPU 版本便于直接复现。CUDA 版本应按设备选择匹配的 torch / torchvision。
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-dev.txt
python -m pytest tests -q
```

没有 uv 时，也可使用 `python -m venv .venv` 和 `python -m pip install`。

没有真实照片时，可以先运行完整的工程验证：

```powershell
python -m scripts.smoke_visual_pipeline --output data/smoke-v1
```

它生成带明确标记的合成图片，使用真实 OpenCLIP 权重，跑通适配层训练、对照评测、FAISS/SQLite 索引和 HTTP 图片查询。可将 `VISUAL_INDEX_PATH` 指向生成的 `data/smoke-v1/index` 体验界面；这些测试图和结果不能用于真实商品效果或简历指标。

### 1. 准备商品图库

准备有授权的商品照片、商品 ID 和资料，按 [数据与训练说明](docs/visual-retrieval.md) 制作 JSONL。每件商品至少两个独立视角，不要把复制或轻微裁剪的图片当作独立测试样本。

```powershell
python -m app.visual.cli validate --manifest data/products/manifest.jsonl --root data/products
python -m app.visual.cli split --manifest data/products/manifest.jsonl --root data/products --output data/products/split-v1.jsonl
python -m app.visual.cli index --manifest data/products/split-v1.jsonl --root data/products --output data/index-baseline
```

首次使用 OpenCLIP 会下载预训练权重，此后可使用本地缓存。默认模型以英文文本为主，中文查询质量需要另行评测。

### 2. 启动后端

```powershell
$env:VISUAL_INDEX_PATH = 'E:\code\smart_price_ai\backend\data\index-baseline'
$env:VISUAL_DEVICE = 'cpu'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

访问 `http://127.0.0.1:8000/docs` 查看接口，`GET /api/v1/visual/status` 查看索引状态。无索引时后端仍能启动，视觉检索明确返回 503；原本地购物接口可独立使用。手机访问时可将 host 改为 `0.0.0.0`，在受信任的局域网连接。

当前是本地单用户应用，不包含生产用户鉴权、跨用户隔离或多实例会话存储。GitHub 发布源代码不等于将这个开发服务公开部署。

### 3. 启动 Flutter

安装 Flutter SDK 后：

```powershell
cd E:\code\smart_price_ai\android-app
flutter pub get
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000
```

Android 模拟器默认 `10.0.2.2`；Windows 桌面用 `http://127.0.0.1:8000`；真机改为开发电脑的局域网地址。首页“拍照找同款”使用本地模型。“多目标识别”等旧接口仍需自行配置火山引擎，配置项见 `backend/.env.example`，不影响新增视觉检索。

### 4. 本地训练与对照评测

```powershell
python -m app.visual.cli train --manifest data/products/split-v1.jsonl --root data/products --output data/adapter-v1.npz --epochs 30
python -m app.visual.cli compare --manifest data/products/split-v1.jsonl --root data/products --adapter data/adapter-v1.npz --output data/comparison-v1.json
python -m app.visual.cli index --manifest data/products/split-v1.jsonl --root data/products --adapter data/adapter-v1.npz --output data/index-adapted-v1
```

训练仅使用 train/val，按验证集选择权重；测试集只用于最终对照。将 `VISUAL_INDEX_PATH` 指向新快照并重启即可应用适配层。指标可能下降，应如实保留结果。完整方法、数据要求、延迟口径和边界见 [视觉检索说明](docs/visual-retrieval.md)。

## 代码入口

```text
backend/app/visual/
  data.py       数据验证、商品隔离划分、重复图检查
  encoder.py    OpenCLIP 编码与残差适配层加载
  index.py      FAISS + SQLite 快照、硬过滤、商品合并、重排
  training.py   难负样本挖掘、对比学习、权重选择、基线对照
  cli.py        validate / split / train / index / evaluate / compare
backend/app/routers/visual.py         本地图文检索 API
backend/tests/test_visual_pipeline.py 数据、训练、索引和 API 测试
android-app/lib/screens/visual_search_screen.dart  Flutter 检索与对比
```

## 验证

```powershell
# backend
python -m pytest tests -q
# 可选：下载真实权重，验证本地图像/文本前向推理；不产生准确率报告
python -m scripts.verify_openclip
# android-app
flutter analyze --no-fatal-infos --no-fatal-warnings
```

GitHub Actions 包含 Windows/Linux 后端测试和 Flutter 静态分析。自动测试使用明确标记的合成夹具，不下载权重，不冒充真实模型效果测试。图片、索引、数据文件、权重和密钥均不提交到 Git。

本次本机验证的版本、通过项和未验收项见 [验证记录](docs/verification.md)。

## 参考与原模块

- [OpenCLIP 官方实现](https://github.com/mlfoundations/open_clip)
- [FAISS 官方文档](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [原购物工作流文档](docs/modules/README.md)：历史模块说明，其中旧验收日期、路径和指标不代表本次新增视觉模块结果。

外部模型、数据和原仓库代码的使用需遵循各自许可；本仓库不重新授予第三方资产的许可。
