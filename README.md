# Self-evloving-market-simulation

这是独立的框架源码目录。框架代码、测试、配置和必要说明集中在这里；Case、画像缓存、权重、运行结果、旧实验和历史代码均不包含在本目录中。

**待确认：环境向量 `e` 的设计存在未解决问题，暂不修改。** 本次只是把现有框架独立出来，保留当前算法及三 head；详细边界见 [环境待确认事项](docs/ENVIRONMENT_PENDING.md)。

## 目录

```text
Self-evloving-market-simulation/
├── src/
│   ├── sems/             # 调度、Decision、协议、评测、RSI
│   ├── sems_attention/   # 四阶段画像、编码、三 head Attention
│   └── sems_tools/       # Case 载入、市场环境、查询和写入工具
├── tests/                # 测试；样例在临时目录生成
├── tools/                # 独立环境 query 检查脚本
├── docs/                 # 当前代码链路及待确认问题
├── pyproject.toml        # 安装、依赖、命令和测试配置
├── .env.example          # 配置占位示例，不含密钥
└── ruff.toml             # 代码格式配置
```

全链路见 [FRAMEWORK.md](docs/FRAMEWORK.md)。这里没有第二份历史画像实现或并列的实验入口。旧 `state_v2` Attention 仅保留显式加载与运行支持；当前环境表示默认值仍是 `trend_v3`，其设计仍处于待确认状态。

## 安装和测试

Python 3.11 及以上，在自己的 Python 环境中安装：

```powershell
python -m pip install -e ".[dev]"
python -B -m pytest -q
```

默认测试自行在系统临时目录生成小型合成 Case，不需要原 `SEMS_v2` 目录、已有结果或模型下载。若要验证自己的真实 Case，可设置 `SEMS_CASE_PATH` 指向外部 ZIP 后运行环境测试：

```powershell
$env:SEMS_CASE_PATH = "D:/your-data/case.zip"
python -B -m pytest -q tests/test_tools_v2.py tests/test_simulation_time.py
```

## 运行

Case 与输出路径必须明确传入；真实模式还须指定外部画像缓存和权重。运行产物建议存放在本目录之外。

```powershell
# 离线链路检查；purchase 可以不购买。
sems-simulate --case "D:/your-data/case.zip" --mode offline --decision-mode purchase --max-sessions 2 --output "D:/your-runs/purchase/result.json"

# review_event 必须选商品并评分评论，保留既有模拟购买写入契约。
sems-simulate --case "D:/your-data/case.zip" --mode offline --decision-mode review_event --max-sessions 2 --output "D:/your-runs/review/result.json"

# 自动选择与运行记录对应的评测器；默认不调用 Judge。
sems-evaluate --case "D:/your-data/case.zip" --simulation "D:/your-runs/review/result.json" --output-dir "D:/your-runs/review/evaluation"

# 独立检查环境 query，不启动 Attention、Decision、训练或外部 API。
python tools/check_environment_queries.py --case "D:/your-data/case.zip" --output-dir "D:/your-runs/new-query-check"
```

也支持 `python -m sems.run` 和 `python -m sems.evaluate`。独立 query 检查的输出目录必须尚不存在，避免覆盖记录。

真实模式使用 `--mode real --policy <权重路径> --profile-cache <四阶段画像缓存目录> --max-sessions <上限> --max-api-requests <上限>`。缺失画像直接报错。旧权重须显式加 `--attention-version state_v2`，不自动改写其含义。

API 配置通过进程环境变量、明确指定的 `SEMS_ENV_FILE` 或本项目 `.env` 读取；只分发 `.env.example`。MiniLM 可通过 `SEMS_EMBEDDING_MODEL_PATH` 指向外部模型。未显式给 API 日志路径时，写入 `SEMS_RUN_DIR`，或用户主目录下的 `SEMS-runtime/runs`。
