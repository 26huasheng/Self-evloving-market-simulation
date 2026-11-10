# 框架代码链路

## 主流程

```text
sems-simulate / python -m sems.run
  → sems/cli/simulate.py：显式读取外部输入、输出与模式参数
  → sems/simulation_runner.py：统一模拟调度
      CaseStore：只读加载 Case
      Clock / build_schedule：按天构建会话，真值独立保留
      CachedProfiles：读取并校验四阶段画像缓存
      MarketEnvironment：当日货架和隔离的用户查询会话
      AttentionRunner：三 head 选择查询，直到 STOP 或预算耗尽
      DecisionModel：购买或评论事件协议
      protocol：校验并转换成环境写入动作
      commit_day：提交当日行为，推进共享市场
  → 把轨迹写到调用者指定的外部结果路径
```

## 模块职责

所有路径相对于 `src/`。

| 模块 | 职责 |
|---|---|
| `sems/api_client.py` | 统一 API 客户端、请求上限、重试、用量日志 |
| `sems/embeddings.py` | MiniLM 文本编码、完整文本分块 |
| `sems/offline.py` | 明确标记的本地测试替身 |
| `sems/simulation_time.py` | 时间推进与评论事件条件下的会话调度 |
| `sems/simulation_runner.py` | 用户、查询、决策和环境提交的编排 |
| `sems/decision.py` | 两种正式 Decision 协议 |
| `sems/protocol.py` | 输出合法性和环境动作转换 |
| `sems/evaluation/` | 两种独立评测契约与 Judge |
| `sems/rsi/` | 诊断、数值训练、结构提案、验证、接受/拒绝、恢复与回滚 |
| `sems_attention/profiles/four_stage.py` | 唯一正式画像生成：初稿→证据审核→修订→一致性检查 |
| `sems_attention/profiles/cache.py` | 读取符合历史哈希、模型和提示词版本的缓存 |
| `sems_attention/profile_encoder.py` | 将画像文本、历史文本和统计编码成个人向量 `p` |
| `sems_attention/trend_encoder.py` | 保留当前公共变化量到 `e` 的编码；设计待确认 |
| `sems_attention/lambda_layer.py` | 动态映射 `Λ(e)` |
| `sems_attention/policy.py` | 共用的工具、商品、参数三 head；兼容旧表示加载 |
| `sems_attention/trend_policy.py` | 当前趋势表示与查询状态融合 |
| `sems_attention/observation.py` | 个人已见信息和查询进度 |
| `sems_attention/actions.py / runner.py` | 合法动作集合、查询执行和停止 |
| `sems_tools/case_store.py` | ZIP/目录输入适配 |
| `sems_tools/environment.py` | 市场状态、用户会话、预算、时间和行为提交 |
| `sems_tools/registry.py` | 查询及写入工具的单一契约 |
| `sems_tools/public_trends.py` | 当前公共变化指标；设计待确认 |
| `sems_tools/keepa.py` | 本地历史字段读取 |

## Attention

```text
公共变化量与掩码 → e → Λ(e)
固定画像 → p
z = LayerNorm(W + Λ(e)p)
z、已见商品信息 H、查询进度 q → 融合状态
  → Tool head：哪个查询工具，或 STOP
  → Product head：查询哪件商品
  → Parameter head：过滤、排序、分页等参数
```

三 head 决定查询，不决定最终购买。私人查询不改变公共 `e`。`state_v2` 保留给显式的旧权重运行；其环境编码与 `trend_v3` 含义不同，不能互换权重。

环境部分的待确认问题见 [ENVIRONMENT_PENDING.md](ENVIRONMENT_PENDING.md)。本次没有改变其算法。

## Decision 与评测

| 模式 | 接口 | 契约 | 对应评测 |
|---|---|---|---|
| `purchase` | `decide_purchase` | 可以不买；不买不产生商品写入 | `evaluation/sessions.py` |
| `review_event` | `decide_review_event` | 必选一个商品、评分、评论 | `evaluation/review_events.py` |

评论事件协议沿用模拟购买、评分、评论三个写入动作，不代表输入提供了真实购买日期。`decide_simulation` 只是购买接口的兼容别名。真实标签只用于评测，不交给查询策略或 Decision。

`sems-evaluate` 从运行记录识别模式；模式不一致时报错。会话 Judge 和评论事件 Judge 分开处理评分尺度和缓存。

## RSI

```text
运行记录 + 对应评测
  → Diagnoser
  → Trainer 或结构候选
  → Validator 同条件比较
  → History 记录、接受/拒绝、恢复/回滚
```

RSI 是可选外层，不在模拟命令结束后自动启动。调用者提供训练资料、评价回调、判据和独立存储目录。未配置 trainer 或判据时，不假定已经训练或接受候选。

`query_context_v1` 结构提案只适用于 `state_v2`，对 `trend_v3` 明确拒绝。数值训练、两种 Decision 和完整推理链仍保留。

## 文件边界

源码包不导入实验或原工作区。这里没有拷贝原始 Case、个人画像缓存、模型文件、运行结果、虚拟环境、历史实现或替代的单阶段画像方案。测试小样例在系统临时目录动态生成；query 检查回执写到显式指定的外部路径。
