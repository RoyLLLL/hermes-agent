# Hermes Agent 内存系统深度分析文档

> **仓库**: [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)  
> **分析日期**: 2026-05-10  
> **分析范围**: 内置文件存储内存 + 可插拔外部 Provider 架构

---

## 一、总体架构概览

Hermes Agent 的内存系统是一个 **双层架构**，同时运行两套独立的内存机制：

```
┌──────────────────────────────────────────────────────────────┐
│                    AIAgent (run_agent.py)                      │
├───────────────────────────────┬──────────────────────────────┤
│   第一层：内置文件存储内存       │   第二层：外部 Provider 插件     │
│   (Built-in Memory Store)      │   (External Memory Provider)   │
├───────────────────────────────┼──────────────────────────────┤
│  tools/memory_tool.py          │  agent/memory_manager.py       │
│  MemoryStore 类                │  MemoryManager 类              │
│  ├── MEMORY.md (agent笔记)     │  ├── 仅允许绑定一个外部provider │
│  └── USER.md (用户画像)       │  └── 自动加载/路由工具调用     │
│                                │                              │
│  文件存储于 ~/.hermes/memories/│  支持的插件:                   │
│                                │  ├── Honcho (AI-native memory)│
│                                │  ├── Mem0 (事实提取+语义搜索) │
│                                │  ├── Hindsight (事后分析)     │
│                                │  ├── Holographic (全息记忆)   │
│                                │  ├── Supermemory              │
│                                │  ├── RetainDB                 │
│                                │  ├── OpenViking               │
│                                │  └── Byterover                │
└───────────────────────────────┴──────────────────────────────┘
```

**核心设计原则**：
1. **两者可独立启用** — 可只用内置存储，也可内置+外部并存
2. **单外部 Provider 约束** — 同一时间只能运行一个外部 Provider（防冲突）
3. **失败隔离** — 一个 Provider 的异常不会影响另一个
4. **内置存储是基础** — 始终可用，外部 Provider 是增强

---

## 二、详细代码模块分析

### 2.1 MemoryProvider 抽象基类 (`agent/memory_provider.py`)

这是整个外部内存 Provider 体系的 **抽象基类**（ABC），定义了完整的生命周期接口：

```
┌─────────────────────────────────────────┐
│            MemoryProvider (ABC)          │
├─────────────────────────────────────────┤
│  属性:                                   │
│  ├── name() -> str         ← 提供商标识 │
│                                          │
│  [核心生命周期 — 必须实现]               │
│  ├── is_available() -> bool  ← 就绪检查 │
│  ├── initialize(session_id)  ← 初始化   │
│  ├── get_tool_schemas()      ← 暴露工具 │
│  └── handle_tool_call()      ← 工具路由 │
│                                          │
│  [可选实现]                              │
│  ├── system_prompt_block()   ← 注入提示│
│  ├── prefetch(query)         ← 预取上下文│
│  ├── queue_prefetch(query)   ← 后台预取  │
│  ├── sync_turn(user, asst)   ← 同步轮次 │
│  ├── shutdown()              ← 关闭      │
│                                          │
│  [可选钩子 — 扩展点]                     │
│  ├── on_turn_start()         ← 轮次开始 │
│  ├── on_session_end()        ← 会话结束 │
│  ├── on_session_switch()     ← 会话切换 │
│  ├── on_pre_compress()       ← 压缩前   │
│  ├── on_memory_write()       ← 内置写入 │
│  ├── on_delegation()         ← 子任务完成│
│  ├── get_config_schema()     ← 配置描述 │
│  └── save_config()           ← 保存配置 │
└─────────────────────────────────────────┘
```

#### 关键设计细节

**1. `initialize()` 方法**
- 接收 `session_id` 和 `**kwargs`（包含 `hermes_home`, `platform`, `agent_context`, `agent_identity`, `user_id` 等）
- `agent_context` 区分 `"primary"`/`"subagent"`/`"cron"`/`"flush"` —— cron 和非 primary 上下文不应写入用户记忆

**2. 可选钩子系统**
- `on_session_switch()`: 处理 `/resume`、`/branch`、`/reset`、`/new` 等会话切换场景
  - `reset=True` 表示真正的清空新会话
  - `parent_session_id` 追踪分支/续会话的 lineage
- `on_pre_compress()`: 在上下文压缩前提取洞察，返回的文本会注入压缩提示
- `on_memory_write()`: 当内置 `memory` 工具写入时同步通知外部 Provider（**双向桥接**）
- `on_delegation()`: 当子代理完成时，父代理的 Provider 收到任务+结果对

**3. `get_config_schema()` + `save_config()`**
- 配合 `hermes memory setup` CLI 命令使用
- 每个 field 可声明: `key`, `description`, `secret`, `required`, `default`, `choices`, `url`, `env_var`
- secrets 自动写入 `.env`，非 secrets 通过 `save_config()` 写入

---

### 2.2 MemoryManager 编排器 (`agent/memory_manager.py`)

核心编排类，管理内置 + 外部 Provider 的完整生命周期：

```
┌─────────────────────────────────────────────┐
│               MemoryManager                  │
├─────────────────────────────────────────────┤
│  内部状态:                                   │
│  ├── _providers: List[MemoryProvider]        │
│  ├── _tool_to_provider: Dict[str, Provider]  │
│  └── _has_external: bool                     │
│                                              │
│  ┌─ Provider 注册 ─────────────────────────┐ │
│  │ add_provider(provider)                   │ │
│  │  ├── builtin 永远接受                    │ │
│  │  └── 非 builtin 仅能一个 (单例约束)     │ │
│  └──────────────────────────────────────────┘ │
│                                              │
│  ┌─ 系统提示注入 ──────────────┐             │
│  │ build_system_prompt()       │             │
│  │ → 收集所有 Provider 的      │             │
│  │   system_prompt_block()     │             │
│  └─────────────────────────────┘             │
│                                              │
│  ┌─ 轮次级操作 ──────────────────┐          │
│  │ prefetch_all(query)           │           │
│  │   → 返回合并的预取上下文      │           │
│  │ sync_all(user, asst)          │           │
│  │   → 持久化完成轮次            │           │
│  │ queue_prefetch_all(query)     │           │
│  │   → 发起下一轮后台预取        │           │
│  └────────────────────────────────┘           │
│                                              │
│  ┌─ 工具路由 ────────────────────┐          │
│  │ get_all_tool_schemas()        │           │
│  │   → 合并去重的工具 Schema     │           │
│  │ handle_tool_call(name, args)  │           │
│  │   → 按 _tool_to_provider 路由 │           │
│  └────────────────────────────────┘           │
│                                              │
│  ┌─ 生命周期广播 ───────────────┐           │
│  │ initialize_all()              │           │
│  │ on_turn_start()               │           │
│  │ on_session_end()              │           │
│  │ on_session_switch()           │           │
│  │ on_pre_compress()             │           │
│  │ on_delegation()               │           │
│  │ on_memory_write()             │           │
│  │ shutdown_all()                │           │
│  └────────────────────────────────┘           │
└─────────────────────────────────────────────┘
```

#### 特殊工具：StreamingContextScrubber

处理流式输出中 `<memory-context>` 标签拆分到多个 chunk 的问题：

```python
# 状态机设计
# 维护 _in_span 标志和 _buf 缓存
# feed(chunk) → 返回可见内容
# flush() → 返回末尾缓存/丢弃未闭合 span

# 边界情况: "<memory" 在 chunk N, "-context>" 在 chunk N+1
# _max_partial_suffix() 检测最长标签前缀匹配
```

**核心策略**：宁可丢弃内容也不泄漏部分 memory context，因为泄漏错误上下文比截断回答更危险。

---

### 2.3 MemoryStore 内置存储 (`tools/memory_tool.py`)

提供 **文件级持久化** 的键值存储，两个独立文件：

```
~/.hermes/memories/
├── MEMORY.md    → agent的个人笔记/环境事实/经验教训
└── USER.md      → 用户画像（偏好/风格/习惯）
```

#### 核心架构

```
┌─────────────────────────────────────────────┐
│              MemoryStore                     │
├─────────────────────────────────────────────┤
│  双重状态:                                   │
│  ├── memory_entries / user_entries (实时)    │
│  └── _system_prompt_snapshot (冻结快照)      │
│                                              │
│  [文件操作]                                  │
│  ├── load_from_disk()   → 读 MEMORY.md/     │
│  │                        USER.md, 去重      │
│  │                        捕获冻结快照       │
│  ├── save_to_disk()     → 原子写入（临时    │
│  │                        文件 + 重命名）    │
│  └── _file_lock()       → 文件锁（防并发）  │
│                                              │
│  [CRUD 操作]                                │
│  ├── add(target, content)                    │
│  │   ├── 扫描注入/泄露模式                   │
│  │   ├── 文件锁下重新读取（防跨会话冲突）    │
│  │   ├── 去重检查                            │
│  │   └── 字符预算检查                        │
│  ├── replace(target, old_text, new_content)  │
│  │   ├── short unique substring 匹配         │
│  │   ├── 多匹配时请求更精确的匹配词          │
│  │   └── 预算检查 + 注入扫描                 │
│  └── remove(target, old_text)                │
│       └── 同上子串匹配                       │
│                                              │
│  [系统提示注入]                              │
│  └── format_for_system_prompt(target)        │
│      → 返回冻结快照（非实时状态）            │
│      → 保持系统提示不变 → 前缀缓存稳定      │
│                                              │
│  [安全扫描]                                  │
│  └── _scan_memory_content()                  │
│      ├── 不可见 Unicode 检测                 │
│      └── 威胁模式检测：                      │
│          ├── prompt_injection                │
│          ├── role_hijack                     │
│          ├── exfil_curl/wget                 │
│          ├── read_secrets                    │
│          ├── ssh_backdoor                    │
│          └── ssh_access / hermes_env         │
└─────────────────────────────────────────────┘
```

#### 关键设计决策

**1. 冻结快照模式 (Frozen Snapshot Pattern)**

```python
# 在 load_from_disk() 时捕获一次
self._system_prompt_snapshot = {
    "memory": self._render_block("memory", self.memory_entries),
    "user": self._render_block("user", self.user_entries),
}

# 整个会话期间返回这个快照
def format_for_system_prompt(self, target):
    return self._system_prompt_snapshot.get(target, "") or None

# 实时状态通过 memory tool 响应返回
# tool 调用 → memory_tool() → JSON 结果含最新 entries
```

**效果**：
- 系统提示在会话期间**完全不变** → LLM Provider 的前缀缓存命中率最高
- 内存写入立即持久化到磁盘，但当前会话不感知
- 下次会话启动时才反映最新状态

**2. 原子写入**

```python
# 临时文件 → fsync → 原子重命名
# 避免：open("w") 在加锁前截断文件 → 并发读取者看到空文件
fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".mem_")
# 写入 → fsync → 原子替换
atomic_replace(tmp_path, path)
```

**3. 字符预算而非 Token 预算**
- `MEMORY.md`: 默认 2200 字符
- `USER.md`: 默认 1375 字符
- 选择字符而非 Token：**模型无关**，不同模型 Token 化方式不同
- 写入超过预算时返回错误，要求先替换或删除

**4. 安全扫描层**
- 阻止 prompt injection 注入系统提示
- 阻止通过 curl/wget 泄露 API Key
- 阻止读取 `.env` 等敏感文件
- 阻止设置 SSH 后门
- 阻止不可见 Unicode（zero-width 字符）隐式注入

**5. 条目分隔符**: `§` (节号) — 支持多行条目

---

### 2.4 内置 Memory 工具 Schema

作为 OpenAI Function-Calling 格式注册：

```python
MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "...详细的使用指南内嵌在 description 中..."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
            "target": {"type": "string", "enum": ["memory", "user"]},
            "content": {"type": "string"},
            "old_text": {"type": "string"},
        },
        "required": ["action", "target"],
    },
}
```

**设计亮点**：工具 Schema 的 description 本身就是一份**完整的 LLM 行为规范**——告诉模型什么时候该写、写什么、不写什么、怎么写，而不是仅在 Python 代码中硬编码规则。Schema 即文档。

---

### 2.5 Agent 主循环中的内存集成 (`run_agent.py`)

在 `AIAgent` 中，内存系统被分散在多个生命周期点：

```
┌───────────────────────────────────────────────────┐
│                 AIAgent 初始化                       │
├───────────────────────────────────────────────────┤
│  1. 读取 memory config                              │
│     - memory_enabled / user_profile_enabled         │
│     - memory_char_limit / user_char_limit           │
│     - nudge_interval                                │
│     - provider (外部 Provider 名称)                  │
│                                                     │
│  2. 初始化内置 MemoryStore                          │
│     self._memory_store = MemoryStore(...)            │
│     self._memory_store.load_from_disk()              │
│                                                     │
│  3. 初始化外部 MemoryManager                        │
│     if provider_name:                               │
│         self._memory_manager = MemoryManager()       │
│         provider = load_memory_provider(name)         │
│         self._memory_manager.add_provider(provider)  │
│         # 注入 session_id, platform, user_id,        │
│         # agent_identity, gateway_session_key 等      │
│         self._memory_manager.initialize_all(...)      │
└───────────────────────────────────────────────────┘
```

```
┌───────────────────────────────────────────────────┐
│              每轮系统提示组装                         │
├───────────────────────────────────────────────────┤
│  1. 内置 MEMORY.md 快照 (若 memory_enabled)         │
│  2. 内置 USER.md 快照 (若 user_profile_enabled)     │
│  3. 外部 Provider system_prompt_block()              │
│  4. memory 使用指南 (MEMORY_GUIDANCE)               │
└───────────────────────────────────────────────────┘
```

```
┌───────────────────────────────────────────────────┐
│              每轮工具调用处理                         │
├───────────────────────────────────────────────────┤
│  if function_name == "memory":                     │
│      1. memory_tool()  ← 内置 MemoryStore          │
│      2. Bridge: self._memory_manager.on_memory_    │
│         write()  ← 通知外部 Provider                │
│                                                     │
│  elif memory_manager.has_tool(function_name):       │
│      self._memory_manager.handle_tool_call()        │
│      ← 路由到外部 Provider 的工具                   │
└───────────────────────────────────────────────────┘
```

```
┌───────────────────────────────────────────────────┐
│             完整生命周期事件序列                       │
├───────────────────────────────────────────────────┤
│  初始化时:                                         │
│    initialize_all(session_id, hermes_home, ...)     │
│                                                     │
│  每轮开始:                                         │
│    on_turn_start(turn_number, message, ...)         │
│    prefetch_all(query) → 注入预取上下文              │
│                                                     │
│  每轮结束:                                         │
│    sync_all(user_msg, asst_response)                │
│    queue_prefetch_all(user_msg) → 后台        │
│                                                     │
│  上下文压缩前:                                     │
│    on_pre_compress(messages) → 提取洞察              │
│                                                     │
│  /resume / /branch / /reset 时:                     │
│    on_session_switch(new_id, reset, ...)             │
│                                                     │
│  子代理完成时:                                     │
│    on_delegation(task, result, child_session_id)     │
│                                                     │
│  会话结束 (exit / timeout):                         │
│    on_session_end(messages)                          │
│    shutdown_all()                                    │
└───────────────────────────────────────────────────┘
```

---

### 2.6 外部 Provider 插件加载系统 (`plugins/memory/__init__.py`)

```
┌─────────────────────────────────────────────┐
│         Memory Provider 插件发现系统          │
├─────────────────────────────────────────────┤
│  [扫描目录]                                 │
│  1. 内置: plugins/memory/<name>/             │
│  2. 用户: $HERMES_HOME/plugins/<name>/       │
│                                              │
│  [匹配规则]                                  │
│  - 文件名重复 → 内置优先                      │
│  - 仅扫描含 "register_memory_provider" 或    │
│    "MemoryProvider" 的 __init__.py            │
│                                              │
│  [加载方式]                                  │
│  1. register(ctx) 模式 (插件标准)            │
│     → _ProviderCollector 模拟 ctx 捕获实例   │
│  2. MemoryProvider 子类自动发现              │
│     → 实例化第一个匹配的子类                  │
│                                              │
│  [CLI 集成]                                  │
│  - discover_plugin_cli_commands()             │
│  - 仅加载当前激活 Provider 的 cli.py          │
│  - 注册为 hermes <provider_name> 子命令       │
└─────────────────────────────────────────────┘
```

---

### 2.7 已实现的外部 Provider

| Provider | 位置 | 工具数 | 主要能力 |
|----------|------|--------|----------|
| **Honcho** | `plugins/memory/honcho/` | 4 | AI-native 用户建模: 画像卡片、语义搜索、记忆上下文、用户结论 |
| **Mem0** | `plugins/memory/mem0/` | 1 | LLM 事实提取、语义搜索重排序、自动去重 |
| **Hindsight** | `plugins/memory/hindsight/` | 1+ | 事后分析 → 提取洞察写入内置或外部 |
| **Holographic** | `plugins/memory/holographic/` | 2+ | 结构化事实存储，全息关联 |
| **Supermemory** | `plugins/memory/supermemory/` | - | 第三方记忆后端 |
| **RetainDB** | `plugins/memory/retaindb/` | - | 本地持久化存储 |
| **OpenViking** | `plugins/memory/openviking/` | - | 完整 MemoryProvider 双向接口 |
| **Byterover** | `plugins/memory/byterover/` | - | 字节级存储 |

---

### 2.8 CLI 命令 (`hermes_cli/memory_setup.py` + `hermes_cli/main.py`)

```bash
hermes memory setup        # 交互式配置 Provider（curses UI）
hermes memory status       # 显示当前内存配置
hermes memory off          # 禁用外部 Provider
hermes memory reset        # 擦除内置 MEMORY.md/USER.md
  hermes memory reset --target all | memory | user
```

此外还有系统提示中的 `MEMORY_GUIDANCE` 和 `SESSION_SEARCH_GUIDANCE`，它们指导 LLM 如何正确使用记忆系统。

---

## 三、核心设计模式总结

### 模式 1: 插件化 Provider + 单例约束
```
MemoryProvider(ABC) ←─ HonchoMemoryProvider
                    ←─ Mem0MemoryProvider
                    ←─ HindsightMemoryProvider
                    ←─ ... (最多一个激活)
```
**目的**: 防止多个外部 Provider 的工具 Schema 冲突，简化系统提示复杂度

### 模式 2: 冻结快照 (Frozen Snapshot)
```
┌──── 会话开始 ────┐     ┌─── 会话期间 ───┐     ┌─── 下次会话 ──┐
│ load_from_disk() │────▶│ system_prompt  │────▶│ load 新快照   │
│ 捕获冻结快照      │     │ 不变(缓存命中)  │     │ 反映上次写入   │
│ 写入操作 → 磁盘   │     │ tool返回实时    │     │               │
└──────────────────┘     └────────────────┘     └───────────────┘
```
**目的**: 最大化 LLM Provider 的前缀缓存命中，同时保证写入即持久化

### 模式 3: 安全优先的注入检测
```
写入内容 → 不可见 Unicode 检测 → 威胁模式匹配 → 放行/拦截
```
**目的**: 内存内容注入系统提示 → 是 prompt injection 的高价值目标

### 模式 4: 双向桥接 (Bidirectional Bridge)
```
内置 memory 工具写入 ──→ 通知 MemoryManager.on_memory_write()
                       ──→ 外部 Provider 同步（mirror）
```
**目的**: 内置写入自动同步到外部 Provider，无需用户/模型手动调用两套 API

### 模式 5: 上下文隔离 (Context Fencing)
```
<memory-context>
[System note: ...Treat as authoritative reference data...]
  ... memory content ...
</memory-context>
```
流式输出时由 `StreamingContextScrubber` 跨 chunk 状态机处理

**目的**: 严格区分"模型记忆"和"新用户输入"，防止模型混淆

---

## 四、配置文件映射

```yaml
memory:
  memory_enabled: true       # 启用 MEMORY.md
  user_profile_enabled: true # 启用 USER.md
  memory_char_limit: 2200    # MEMORY.md 字符上限
  user_char_limit: 1375      # USER.md 字符上限
  nudge_interval: 10         # 每 N 轮提示更新记忆
  provider: ""               # 外部 Provider 名称
```

---

## 五、文件依赖关系图

```
agent/memory_provider.py          ← 抽象基类（无依赖）
       ↑
agent/memory_manager.py           ← 编排器（依赖 MemoryProvider）
       ↑
tools/memory_tool.py              ← 内置存储（依赖 get_hermes_home, registry）
  MemoryStore 类
       ↑
run_agent.py                      ← 主循环集成
  ├── 初始化 MemoryStore
  ├── 初始化 MemoryManager
  ├── 系统提示组装（注入快照）
  └── 工具调用分发（memory + provider tools）
       ↑
plugins/memory/__init__.py        ← 插件发现
  ├── discover_memory_providers()
  └── load_memory_provider(name)
       ↑
plugins/memory/<name>/__init__.py ← 具体 Provider 实现
  ├── HonchoMemoryProvider
  ├── Mem0MemoryProvider
  ├── HindsightMemoryProvider
  └── HolographicMemoryProvider
       ↑
hermes_cli/memory_setup.py        ← CLI 配置向导
hermes_cli/main.py                ← hermes memory 命令注册
```

---

## 六、设计亮点与权衡

### 亮点
1. **双层架构**：简单场景用内置文件存储，复杂场景接入外部 Provider
2. **单 Provider 约束**：避免工具 Schema 爆炸和语义冲突
3. **冻结快照 + 原子写入**：兼顾缓存性能和写入安全性
4. **内嵌 Schema 文档**：工具描述本身就是 LLM 行为指南
5. **字符预算而非 Token 预算**：模型无关的公平限制
6. **完整生命周期钩子**：从初始化到会话切换到关闭全面覆盖

### 需注意的权衡
1. **内置存储无语义搜索**：全靠子串匹配，大记忆集合效率下降
2. **字符预算较小**：默认 2200+1375 字符，复杂场景可能不够
3. **冻结快照延迟**：当前会话写入了记忆但看不到，需等待下次会话
4. **无内置记忆合并/去重**：靠模型自觉写入，重复条目仅去重同一字符串

---

*文档生成于 Hermes Agent 仓库代码分析，覆盖 `agent/memory_provider.py`、`agent/memory_manager.py`、`tools/memory_tool.py`、`plugins/memory/__init__.py`、`run_agent.py`（相关部分）、`hermes_cli/memory_setup.py` 和所有 8 个外部 Provider 插件。*
