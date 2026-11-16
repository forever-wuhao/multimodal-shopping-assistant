# 本地视觉检索与训练说明

## 当前实现边界

新模块使用 OpenCLIP 提取图片/文本向量、FAISS 精确内积检索和 SQLite 商品快照。默认 `ViT-B-32 / laion2b_s34b_b79k`，图文向量统一 L2 归一化。

微调方法是 **冻结 OpenCLIP 主干，训练共享的残差瓶颈适配层**：`normalize(x + W_up ReLU(W_down x))`。这是特征适配，不是从零训练大模型，也不是 LoRA。图像监督对比损失加 0.2 倍图文联合监督对比损失；每个商品在批次中抽取两个不同视角。按训练集商品的基础视觉特征中心挖掘最近的异款商品作为难负样本。难负样本可能存在错误标注，需要人工检查。

只有有授权的真实商品数据完成训练与留出测试，才能报告真实指标。仓库测试生成的几何图形只验证流程，不能证明实际拍照检索效果。目前不提供未经验证的同款准确率、模型权重或性能提升承诺。

## 数据准备

目录示例（图片、模型和索引不提交到 Git）：

```text
backend/data/products/
  manifest.jsonl
  images/shoe_a_front.jpg
  images/shoe_a_side.jpg
  ...
```

每张图片一行 JSON。`image` 相对于 `--root`，禁止绝对路径、越界路径和失效文件。完整示例见 `backend/examples/product-manifest.example.jsonl`，示例里的文件名需要替换成实际照片。

| 字段 | 含义 |
| --- | --- |
| image_id | 图片唯一 ID，只允许字母、数字、横线和下划线 |
| product_id | 同款身份 ID；同一商品的不同角度共用此值 |
| image | 相对图片路径 |
| title / text | 展示标题 / 用于图文训练的描述；默认模型建议英文描述 |
| category / brand | 品类、品牌，筛选采用精确匹配 |
| price_cny | 资料中的人民币价格，可为 null；未知价格不能通过预算筛选 |
| attributes | 字符串属性，例如 color、size；不能从图片臆测未提供的参数 |
| source | 商品资料来源和版本，不能伪造；价格不是实时电商报价 |
| split | unassigned、train、val、test |
| role | gallery 或 query；同一商品的不同照片分别作检索参考和查询 |

同一商品的标题、价格、品牌、属性和来源必须一致。照片的 `text` 可以描述不同视角。商品身份不要仅按拍摄文件夹随意划分；外观无法区分的尺码不适合作为视觉检索中相互排斥的负样本。尽量覆盖实拍背景、遮挡、不同角度和外观相似款，并记录图像授权和采集日期。

`split` 按商品 ID 隔离，约 70%/15%/15%；每个评估分区至少两个商品，因此极小数据的比例会不同。每个商品至少两张不同照片，建议收集至少 100 个商品、每个 3–5 个独立视角，再按业务复杂度扩大。程序拒绝重复 ID、跨分区商品和完全相同像素的图片；**裁剪、压缩、连拍等近重复还需人工排查**。小于等于五个图库商品时 Top5 没有区分力，报告会提示。

## 可复现命令

以下命令均在 `backend` 目录及已安装 ML 依赖的 Python 环境运行。输出路径已存在时会拒绝覆盖，保留每次实验。

```powershell
python -m app.visual.cli validate --manifest data/products/manifest.jsonl --root data/products
python -m app.visual.cli split --manifest data/products/manifest.jsonl --root data/products --output data/products/split-v1.jsonl --seed 42

# 先运行基础模型，可直接构建索引，不依赖训练结果
python -m app.visual.cli index --manifest data/products/split-v1.jsonl --root data/products --output data/index-baseline

# CPU 可运行；安装匹配的 CUDA PyTorch 后可添加 --device cuda
python -m app.visual.cli train --manifest data/products/split-v1.jsonl --root data/products --output data/adapter-v1.npz --epochs 30 --products-per-batch 8
python -m app.visual.cli compare --manifest data/products/split-v1.jsonl --root data/products --adapter data/adapter-v1.npz --output data/comparison-v1.json
python -m app.visual.cli index --manifest data/products/split-v1.jsonl --root data/products --adapter data/adapter-v1.npz --output data/index-adapted-v1
```

训练只编码 train/val，冻结主干后缓存该次训练所需特征至内存，适配层使用 AdamW。权重选择只看验证集商品宏平均 Hit@1；测试集不参与选择。默认超参数、随机种子、验证历史、训练/验证商品 ID、难负样本配对及清单哈希保存在 `.npz` 中。训练不保证超过基线，最终对比结果可为负提升；不应按测试集表现反复挑参数。

索引目录包含 `vectors.faiss`、`catalog.sqlite3`、`metadata.json`、去除 EXIF 后的 JPEG 缩略图，以及可选的 `adapter.npz`。目录通过同卷临时目录构建后发布，快照不可原地覆盖。启动时检查索引与 SQLite 哈希、数量以及模型/适配层标识；替换模型必须重建图库向量。只加载自己构建或信任的快照，不能把任意上传文件当作 FAISS 索引。

## 检索与排序

1. 图像或文本提取向量；同时提供时使用 0.85 图像 + 0.15 文本后归一化。
2. FAISS `IndexFlatIP` 精确检索。当前实现扫描整个小型图库再过滤，避免先截断 TopK 丢失满足硬条件的商品。
3. 按品类、品牌、预算与属性过滤；缺少被要求的属性则不通过。按商品 ID 合并多个视角，取最高相似度。
4. 可选低价偏好作为独立加分，范围 0–0.03；返回原始相似度与偏好加分，方便解释和核对。
5. 返回资料来源、快照哈希、条件核对信息和候选图片。相似度不是概率，结果不保证同款。当前未校准开放集拒识阈值；图库外商品可能返回相似候选。

这是小型图库的精确检索基线，不能据此声称支持百万商品或高并发。推理单进程串行执行，繁忙返回 429；图库增大后需另行评估分片、过滤索引和批处理。

## 评测口径

- gallery/query 为同一留出商品的不同照片，属于闭集检索；训练商品和测试商品互斥。
- **Hit@1 / Hit@5**：前 1 / 5 个不同商品候选中是否出现正确商品，按查询图片平均。由于每个查询只有一个正确商品身份，也可理解为身份级 Recall@K。
- **Macro Hit@1**：先按商品平均，再跨商品平均，避免照片多的商品占更大权重。
- 延迟记录预热后的单张预处理、模型编码、FAISS 查询和商品合并，排除磁盘解码、索引构建及网络传输。输出平均值、P95、设备名称、PyTorch 版本；CPU 和 GPU 结果不可直接混比。
- `compare` 在同一进程、同一数据和设备上依次评测基线与适配层，输出失败样本的真实 ID 与 Top5，供检查背景、视角和相似款误匹配。

数据清单哈希标识标注版本，不等同于所有原始图片的内容哈希。发布正式实验应另行保存不可变的数据归档及校验和。只有人工确认真实标注、近重复排查与采集授权后才能将结果用于简历。

## 应用接口

- `GET /api/v1/visual/status`：是否已准备好及商品数。
- `POST /api/v1/visual/search`：multipart；`image` / `text` 至少一个。可带 `category`、`brand`、`max_price`、`attributes` JSON 字符串、`prefer_low_price` 和 `limit`。
- `GET /api/v1/visual/images/{image_id}`：当前图库缩略图。

不保存用户上传的查询图，不读取客户端传入的本地路径。上传最多 8MB、解码最多 2000 万像素。返回 422 输入无效、413 过大、429 忙碌、503 未配置/不可用。

Flutter 首页进入拍照找同款，支持图文查询、调整条件再次搜索、选择 2–3 个候选对比资料。原购物对话仍使用独立的 **18 件虚构样例商品库**；不会把视觉图库商品暗中转换为虚构库里的同款。自然语言多轮购物和视觉库的统一事实来源是下一步工作，当前需区分两条链路。

## 实现参考

- [OpenCLIP 官方仓库](https://github.com/mlfoundations/open_clip)：模型加载、预处理和图文编码。
- [FAISS 官方入门文档](https://github.com/facebookresearch/faiss/wiki/Getting-started)：float32 向量与精确索引。
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)：多正样本监督对比目标。
