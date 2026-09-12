# OmniGen

SDF-caption 数据生成与质量审核工具。

## ModelHub API

凭据只通过环境变量注入，不要把带 `ak` 的完整 URL 写入配置：

```bash
cp .env.example .env
export MODELHUB_AK="..."
export MODELHUB_MODEL_NAME="gpt-5.6-sol"
export MODELHUB_ENDPOINT="..."
```

生成配置使用统一的 `modelhub` 类型：

```yaml
model_type: modelhub
model_config:
  ak: ${MODELHUB_AK}
  endpoint: ${MODELHUB_ENDPOINT}
  model_name: ${MODELHUB_MODEL_NAME}
  timeout: 120
```

客户端会自动构造 `ak` 查询参数、生成 `X-TT-LOGID`，并支持文本、多轮消息、图片 URL、本地图片及 PIL 图片。需要保存调用元数据时使用 `generate_with_metadata()`。

```python
from src.models import ModelHub

with ModelHub() as model:
    result = model.generate_with_metadata("What is the result of 1+1?", max_out_len=500)
```

## SDF-caption 生成

输入目录按分子 ID 配对：

```text
data/sdf_caption/
├── gdb_1.sdf
├── gdb_1.facts.json
└── reference.json
```

`facts.json` 包含 `composition`、`topology` 和 `geometry`。SMILES 可直接写在 facts 中，也可由 `reference.json` 按 `molecule_id` 提供。

```bash
python -m main.generate \
  --type sdf_caption \
  --data-source data/sdf_caption \
  --output output/sdf_caption.jsonl

python -m main.validate \
  --type sdf_caption \
  --file output/sdf_caption.jsonl
```

模型输入同时包含参考风格、SMILES、verified facts 和当前构象 SDF，模型只输出 `{"caption":"..."}`。生成结果可直接挂载到审核看板。

## 本地审核

安装 Python 与前端依赖：

```bash
conda activate omnigen
pip install -e ".[dashboard,test]"
cd dashboard && npm install
```

启动 API：

```bash
omnigen-review
```

另开终端启动前端：

```bash
cd dashboard
npm run dev
```

访问 `http://localhost:5173`。未指定数据文件时会加载 `dashboard/sample-data/items.json`。

指定自己的 JSON 或 JSONL：

```bash
OMNIGEN_DATA_PATH=/absolute/path/items.jsonl \
OMNIGEN_REVIEW_DB=/absolute/path/reviews.sqlite3 \
omnigen-review
```

## Docker

```bash
OMNIGEN_DATA_DIR=/absolute/path/to/data \
OMNIGEN_DATA_FILE=items.jsonl \
docker compose up --build
```

访问 `http://localhost:8000`。审核结果保存在 `omnigen-reviews` volume，可从页面右上角导出。

## 数据格式

每条记录至少需要稳定的 `id`。审核台直接识别以下字段：

```json
{
  "id": "qm9-000001",
  "display_key": "QM9-000001",
  "source": "qm9",
  "source_label": "QM9",
  "source_id": "gdb_1",
  "formula": "CH4",
  "smiles": "C",
  "caption": "...",
  "prompt": "...",
  "sdf": "...",
  "facts": {},
  "provenance": {}
}
```

JSON 支持顶层数组或 `{"items": [...]}`，JSONL 每行一条记录。

## 验证

```bash
conda run -n omnigen python -m pytest -q
cd dashboard && npm run build
```

原有生成入口仍可通过 `python -m main.generate` 使用。
