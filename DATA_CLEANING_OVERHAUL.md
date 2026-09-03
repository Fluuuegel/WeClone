# 数据清洗管线大修记录（QQ chatlog 数据源适配）

> 日期：2026-09-02 ~ 2026-09-03
> 分支：`feat/qq-chatlog-clean-pipeline`
> 状态：已在 `chat-sft-cleaned`（13143 条）上完成训练与合并导出

## 1. 背景

`dataset/csv/` 实际存放的是 **QQ chatlog 导出**（`QQ_/QQ群_/私聊_` 前缀文件夹），并非微信导出。chatlog 将所有消息导出为 `type_name=text`，语音、图片、引用、表情等以文本代码内联表达（如 `[语音通话]`、`[引用 昵称：内容]`、`/呲牙`、`@昵称`）。因此原有的按 `type_name` 的 cut/skip 类型过滤对该数据源**全部失效**，大量占位符、真实昵称、系统消息直接进入训练数据。

本次修改围绕四个目标：

1. 修复 QA 配对状态机导致的**消息静默丢失**
2. 清理**系统消息/占位符/引用/@提及**等非人类文本污染
3. 将 PII 处理从"整条删除"改为**脱敏**，并消除大量误报
4. 对**群聊**多说话人导致的语义错乱记录做 LLM 清洗

## 2. 修改内容（按文件）

### `weclone/data/qa_generator.py`

- **match_qa 累积指令**：对方连续发多条消息时，不再用后一条覆盖前一条（原实现静默丢弃约 1.5 万条消息），而是累积成一个 user 轮（换行连接）。
- **超窗回复保留**：自己的回复超过时间窗口时，原实现既不保存也不清空对话直接丢弃；现在先保存旧对话，再把该回复作为 `<begin_chat>` 轮保留。
- **删除 `<begin_chat>` 人工注入**：原实现把"自己主动发的消息"构造成 `<begin_chat>你应该说：X</begin_chat>` 的注入指令，占 48.4% 记录，会让模型学到"看到 tag 就提取文本"的路径依赖（推理时该 tag 永远不会出现）。现改为删除注入消息对、保留后续自然轮次；只剩注入对的记录整条丢弃。
- **`clean_chat_text` 文本清洗**（`load_file` 内新增）：
  - 删除控制符（如 `\x14`）
  - 整条删除：纯占位符（`[语音通话]` 含时长/已取消形态）、`撤回了一条消息`、`你已添加了…`、`我通过了你的朋友验证请求`、`[自动回复]…`、`[转账]￥xx` / `[转账收款]￥xx`、泄漏的 XML 块（`<msg><appmsg…`，来自文件消息序列化）、`微信红包`
  - 内联清理：语音/视频通话标记、支付通知句（"你有一笔待接收的转账"）
  - `[引用 昵称：内容]` → `[引用] 内容`（去掉昵称，保留被引用内容作上下文）
  - `@昵称` → `@我` / `@联系人N`（全局昵称映射，见 `_build_nickname_map`）
- **`_build_nickname_map`**：处理前预扫描全部 CSV，收集 @提及 ≥2 次的昵称，全数据集统一映射（自己 → `@我`，他人 → `@联系人N`），`@media` 等代码文本不受影响。
- **群聊标记**：`QQ群_` 前缀文件夹的消息标记 `is_group`，贯穿 ChatMessage → QaPair → 输出 JSON 的 `group` 字段。

### `weclone/core/PII/pii_detector.py`

- **脱敏替代删除**：新增 `anonymize_batch()`，检测到 PII 后用中文掩码替换（`【电话号码】`/`【数字ID】`/`【敏感信息】`等），不再整条删除消息（删除会破坏 QA 配对）。
- **实体收紧**：`LOCATION`/`ORGANIZATION`/`AGE` 移出过滤列表（"我在上海"等日常内容不再误杀）。
- **正则收紧**：数字 ID 改为 7 位以上连续数字 / 6 位以上字母数字混合；不再匹配 `2022-05-18` 日期和 `3-4` 量词范围（原实现误报重灾区）。
- **`WECLONE_PII_N_PROCESS`**：批量分析进程数可配置（默认 24）。部分容器宿主机上多进程 worker 无法初始化 CUDA，设 `WECLONE_PII_N_PROCESS=1` 降级。

### `weclone/data/qq_emojis.py`（新增）

QQ 表情代码**删除白名单**（约 240 个名称：`/呲牙`、`[捂脸]`、`[OK]`、`[Sleepy]` 等中英文新旧格式）。命中即从文本中删除，避免模型学到输出无法渲染的 `/斜杠代码`；白名单外的斜杠/方括号（如"和/或"、"@media"）保持原样。

### `weclone/data/clean/strategies.py`

- LLM 评分**只针对群聊记录**（`group=true`），私聊记录 score=6 直通不评分，避免误杀私聊数据。
- `clean()` 加固：`dataset_info.json` 里没有的数据集名不再抛 `TypeError`，改为告警并回退原数据集。

### `weclone/data/models.py`

- `ChatMessage.is_group` / `QaPair.group` 字段。

### `weclone/core/inference/offline_infer.py`

- guided decoding 后端从 `guidance` 改为 `xgrammar`（guidance 需要额外依赖，xgrammar 随 vLLM 常见安装自带）。

### `settings.jsonc` 相关建议

- `qa_match_time_window: 10`（5 分钟切碎对话且丢弃约 1.5 万条自己的消息）
- `cutoff_len: 1536`（24G 显存下 2048 会 OOM；数据 99.96% < 1536 tokens）
- `clean_dataset.enable_clean: true` + `train_sft_args.dataset: chat-sft`（train-sft 自动切换清洗后数据；注意 `dataset` 直接填 `chat-sft-cleaned` 会让 clean() 拼出 `chat-sft-cleaned-cleaned`）
- `default_system` 已清空（数据中不再携带"请你扮演一名人类"系统指令）
- 训练时建议加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 3. 数据生成流程

```bash
# Phase 1：生成数据集（CPU spaCy；本机多进程 CUDA 受限时的稳妥跑法）
CUDA_VISIBLE_DEVICES="" python -c "
from weclone.data.qa_generator import DataProcessor
dp = DataProcessor(); dp.enable_clean = False; dp.main()
"
# Phase 2：GPU 上对群聊记录做 LLM 评分并写回 score 字段
python /tmp/score_groups.py   # 即 LLMCleaningStrategy.judge 的独立封装

# Phase 3：按 accept_score 过滤，生成 sft-my-cleaned.json
python -c "
from weclone.data.clean.strategies import LLMCleaningStrategy
from weclone.utils.config import load_config
LLMCleaningStrategy(make_dataset_config=load_config('make_dataset')).clean()
"
```

产物：`dataset/res_csv/sft/sft-my.json`（带评分）、`sft-my-cleaned.json`（训练用，已 gitignore）。

## 4. 训练与合并结果（2026-09-03）

- 数据：`chat-sft-cleaned` 13143 条（私聊 11769 + 群聊 1374），2 epochs / 822 步 / 30.6 分钟
- 步均 loss：8.06 → 3.55（`cutoff_len=1536` + `expandable_segments` 修复了 39% 处的 OOM）
- 合并导出：`/root/autodl-tmp/merged_model`（bf16，15G，含 `Modelfile`），tar 压缩包 `merged_model.tar.gz`（12G）

## 5. 已知环境注意事项

- 本机（autodl）**多进程 worker 无法初始化 CUDA**（cupy/spawn 子进程报 `cudaErrorInitializationError`），影响 presidio 批量分析与 vLLM V1 EngineCore。CPU spaCy 多进程正常；transformers 路径（web-demo / server）不受影响。
- 数据源是 QQ chatlog 时，`cut_type_list`/`skip_type_list` 的 type_name 过滤不生效，所有非文本消息都要靠 `clean_chat_text` 的文本规则处理。

## 6. 追加修复（2026-09-03 下午）

### 6.1 跨联系人串台 bug

**根因**：`match_qa` 的 `WAITING_RESPONSE` 分支在窗口断裂（新会话）时只保存并清空 `conversation_messages`，不清空 `current_instructions`（待回复的累积指令队列），导致上一个会话/上一个文件遗留的对方消息被 `"\n".join` 进下一个会话的 user 轮。

**修复**（双重保障）：
1. 窗口断裂时 `current_instructions = [msg]`（重置而非累积），否则继续累积；
2. `main()` 改为**按文件逐个调用 `match_qa`**（每个 CSV 一个联系人，id 计数器跨文件共享）——联系人之间的消息在结构上不可能互相串入。

### 6.2 清洗规则调整

- 括号统一：所有含括号的正则使用统一字符类 `_L_BR = [\[［（(]` / `_R_BR = [\]］）)]`，半角/全角的方括号、圆括号四种形态一次覆盖（占位符 `[图片]`/`（图片）`/`(图片)`/`［图片］` 均匹配）。
- 引用：`[引用 昵称：内容]` **整体删除**（昵称只以冒号定界，容忍昵称内出现括号）；支持嵌套括号与导出截断导致的未闭合引用（删除到消息末尾）。
- 系统邀请长句：`_system_invite_re` 专杀 `（链接） 邀请你加入群聊` / `XX参与了接龙` 等带固定后缀的系统提示（**必须先于行内占位符执行**，否则占位符先被转换导致标记失效）。
- 链接：`https?://[^\s\\]*`（可选吞掉一个换行）删除所有 http/https 链接，含后跟空格被截断的分享链接。
- 合并转发的聊天记录（`[转发的聊天记录]…`，第三方对话序列化文本，含他人真名/链接）整条删除。

### 6.3 代码结构

正则清洗从 `qa_generator.py` 迁移至 `weclone/data/clean/strategies.py` 的 `ChatTextCleaner` 类（昵称映射由 `_build_nickname_map` 收集后经 `set_mention_map()` 注入）。

### 6.4 配置联动

`train_sft_args.dataset` 与 `make_dataset` 共用同一个字段：必须保持 `chat-sft`（`enable_clean=true` 时训练自动切换 `chat-sft-cleaned`）。直接填 `chat-sft-cleaned` 会让 `length_cdf` 与 `clean()` 出错（已加防御：`length_cdf` 自动去掉 `-cleaned` 后缀，`clean()` 对缺失数据集名告警回退）。

### 6.5 后续规则补充（2026-09-03 晚）

- 分享卡片：消息开头的 `[链接]`/`[小程序]`/`[卡片链接]` + 标题正文 → 整条丢弃；句中的 `[链接]` 仍转换为 `（链接）`。
- 付款诈骗话术：以"就可以帮我付款啦"结尾 → 整条丢弃（基于原始文本，先于 URL 删除判断）。
- 闪照：纯占位符整条丢弃；客户端提示语"请使用新版手机QQ查看闪照。"逐句删除（保留前后用户文字）。
- 无价值占位符 `（闪照）/（名片）/（通话）/（QQ红包）` 出现在任何位置都直接删除，不再转换保留。
- 单元测试：`tests/test_chat_text_cleaner.py`（29 个用例，pytest）。
