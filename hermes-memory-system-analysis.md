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

---

## 四、Memory 完整保存流程分析

本章深入分析 Agent 如何**决策**保存记忆，判断条件是什么，以及三条互补的触发路径如何协同工作。

### 4.1 保存记忆的三大触发机制

```
保存记忆 = Agent 的"自省决策" × 3 条独立路径
```

| 触发路径 | 触发条件 | 谁决策 | 触发时机 | 对应代码 |
|----------|----------|--------|----------|----------|
| **① 前置引导** | 系统提示中的 MEMORY_GUIDANCE 持续告知 | Agent 自主（LLM 自行判断） | 任一回合中，Agent 认为合适时 | `prompt_builder.py:150` |
| **② 周期性 nudge** | 每 N 轮用户消息（默认 10 轮）自动触发 | 后台 review agent 判断 | **本轮响应交付后**，后台异步执行 | `run_agent.py:11607-11617` |
| **③ 模型自主调用** | Agent 认为有值得保存的信息 | Agent 自主 | 任意工具调用回合 | `memory_tool.py` 的 tool schema |

> 注意：这三条路径**并行存在、互补**。如果 Agent 在对话中自主调用了 `memory` 工具，nudge 计数器归零，不会重复触发后台 review。

---

### 4.2 触发路径①：系统提示前置引导（MEMORY_GUIDANCE）

#### 注入时机

在每次 AI 调用前，`run_agent.py` 组装系统提示：

```python
# run_agent.py:5700-5730
prompt_parts.append(MEMORY_GUIDANCE)       # → 行为规则
prompt_parts.append(SESSION_SEARCH_GUIDANCE)  # → 查询规则

# 注入冻结快照（已有记忆内容）
if self._memory_enabled:
    mem_block = self._memory_store.format_for_system_prompt("memory")
    if mem_block:
        prompt_parts.append(mem_block)
if self._user_profile_enabled:
    user_block = self._memory_store.format_for_system_prompt("user")
    if user_block:
        prompt_parts.append(user_block)
```

#### MEMORY_GUIDANCE 定义的完整决策规则

**5 个"应该保存"的信号：**

| 信号 | 原文 | 举例 |
|------|------|------|
| S1. 用户纠正 | "User corrects you or says 'remember this' / 'don't do that again'" | "不要用这么啰嗦的格式" |
| S2. 用户分享偏好 | "User shares a preference, habit, or personal detail" | "我喜欢用 pytest" |
| S3. 环境发现 | "You discover something about the environment" | "OS 是 Ubuntu 24.04" |
| S4. 约定/API 怪癖 | "You learn a convention, API quirk, or workflow" | "这个 API 需要 X 头" |
| S5. 稳定事实 | "You identify a stable fact that will be useful again" | "项目使用 Poetry 管理" |

**4 个"不应该保存"的信号：**

| 反信号 | 原因 |
|--------|------|
| 任务进度 | "不要保存 PR 编号、issue 编号、commit SHA、'修复了 bug X'" |
| 临时 TODO | "使用 session_search 来回忆" |
| 7 天内会过期的 | "如果一个事实一周后会过时，它不属于记忆" |
| 指令式语句 | "写声明式事实，不是指令自己" → '用户偏好简洁' ✓, '要简洁' ✗ |

**优先级规则：**
```
用户偏好/纠正 > 环境事实 > 程序性知识
最有价值的记忆 = 防止用户将来重复纠正 agent 的事实
```

---

### 4.3 触发路径②：周期性 Nudge 后台 Review

这是**最精巧的机制**——Agent 不必在每轮对话中自我判断，系统自动在后台检查。

#### 触发条件链

```python
# run_agent.py:11607-11617 — 每轮用户消息开始时
_should_review_memory = False
if (self._memory_nudge_interval > 0                         # (1) nudge 间隔 > 0（默认 10）
    and "memory" in self.valid_tool_names                    # (2) memory 工具已启用
    and self._memory_store):                                 # (3) MemoryStore 已初始化
    self._turns_since_memory += 1                            # (4) 递增计数器
    if self._turns_since_memory >= self._memory_nudge_interval:  # (5) 达到阈值
        _should_review_memory = True
        self._turns_since_memory = 0                         # (6) 重置计数器
```

**5 个条件必须全部满足**才会触发。

#### 执行时机——响应交付后异步执行

```python
# run_agent.py:15128-15153
# 等待本轮所有工具调用和最终响应完成后
if final_response and not interrupted and (_should_review_memory or _should_review_skills):
    try:
        self._spawn_background_review(
            messages_snapshot=list(messages),    # 传入当前对话快照
            review_memory=_should_review_memory,
            review_skills=_should_review_skills,
        )
    except Exception:
        pass  # 后台 review 是 best-effort
```

**中断的轮次不触发**——不完整的工具链不应污染持久记忆。

#### 后台 Review Agent 的完整生命周期

```python
def _spawn_background_review(self, messages_snapshot, review_memory, review_skills):
    # (1) 选定 review prompt
    if review_memory and review_skills:
        prompt = _COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = _MEMORY_REVIEW_PROMPT
    else:
        prompt = _SKILL_REVIEW_PROMPT

    # (2) 在后台线程中 fork 一个完整的 AIAgent
    def _run_review():
        review_agent = AIAgent(
            model=self.model,                       # 继承父 agent 的模型
            max_iterations=16,                      # 最多 16 轮工具调用
            quiet_mode=True,                        # 静默模式，不输出到屏幕
            enabled_toolsets=["memory", "skills"],   # 只暴露 memory 和 skill 工具
        )
        # 关键：共享同一个 MemoryStore 实例
        review_agent._memory_store = self._memory_store
        # 关闭 review agent 自身的 nudge（防递归）
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0

        # (3) review agent 分析对话历史并行写入
        review_agent.run_conversation(
            user_message=prompt,                    # review prompt 作输入
            conversation_history=messages_snapshot, # 全程对话快照
        )

        # (4) 汇总操作，输出用户可见摘要
        actions = self._summarize_background_review_actions(
            getattr(review_agent, "_session_messages", []),
            messages_snapshot,
        )
        if actions:
            summary = " · ".join(dict.fromkeys(actions))
            self._safe_print(f"  💾 Self-improvement review: {summary}")

    threading.Thread(target=_run_review, daemon=True).start()
```

#### Review Prompt 的决策逻辑

```python
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)
```

#### 计数器重置时机

```python
# run_agent.py:10386-10390
# Agent 每调用一次 memory 或 skill_manage 工具，计数器立刻归零
if function_name == "memory":
    self._turns_since_memory = 0
elif function_name == "skill_manage":
    self._iters_since_skill = 0
```

这意味着：**如果 Agent 在对话中主动保存了记忆，后台 review 就不会触发**——避免重复劳动。

---

### 4.4 触发路径③：模型自主调用 memory 工具

最直接的路径——Agent 在对话中自行判断并调用 memory 工具。

**工具 Schema 的 description 本身就是一份完整的保存指南：**

```python
# tools/memory_tool.py:517-538
MEMORY_SCHEMA = {
    "description": (
        "Save durable information to persistent memory...\n\n"
        "WHEN TO SAVE:\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail\n"
        "- You discover something about the environment\n"
        "- You learn a convention, API quirk, or workflow\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs...\n"
        "If you've discovered a reusable approach, save it as a **skill** instead.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is — name, role, preferences, communication style\n"
        "- 'memory': your notes — environment facts, project conventions, tool quirks\n\n"
        "ACTIONS: add (new entry), replace (update existing), remove (delete)\n\n"
        "Write memories as declarative facts, not instructions to yourself."
    ),
}
```

---

### 4.5 三种触发路径的完整关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        完整记忆保存流程图                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  会话开始                                                                │
│     │                                                                    │
│     ▼                                                                    │
│  ┌──────────────────┐                                                    │
│  │ 系统提示注入       │  ← MEMORY_GUIDANCE（前置指导）                    │
│  │  + 已有记忆快照    │  ← 冻结快照（format_for_system_prompt）           │
│  └────────┬─────────┘                                                    │
│           │                                                              │
│   ┌───────▼───────┐                                                      │
│   │ 用户发送消息    │                                                     │
│   └───────┬───────┘                                                     │
│           │                                                              │
│   ┌───────▼─────────┐                                                    │
│   │ 判断 nudge 触发   │ ◄── _turns_since_memory ≥ nudge_interval(默认10) │
│   └───────┬─────────┘                                                    │
│           │                                                              │
│   ┌───────▼──────────┐                                                   │
│   │ Agent 处理用户请求 │                                                  │
│   └───────┬──────────┘                                                   │
│           │                                                              │
│           │   ┌─────────────────────────────┐                            │
│           │   │ Agent 自主调用了 memory ？    │                           │
│           │   └─────────────────────────────┘                            │
│           │                              │                               │
│           │  ┌───────────────────┐        │  ┌──────────────┐            │
│           │  │ 写入 MemoryStore   │        │  │ 继续正常对话  │            │
│           │  │ 计数器归零         │        │  └──────┬───────┘            │
│           │  │ 通知外部 Provider   │        │         │                   │
│           │  └───────────────────┘        │         │                   │
│           │                              │         │                   │
│           └──────┬───────────────────────┘         │                     │
│                  │                                 │                     │
│                  ▼                                 ▼                     │
│          ┌───────────────────┐        ┌────────────────────┐             │
│          │ 本轮响应已交付用户   │        │ 本轮响应已交付用户   │             │
│          └─────────┬─────────┘        └────────┬───────────┘             │
│                    │                           │                         │
│                    ▼                           ▼                         │
│                    ┌──────────────────────┐                              │
│                    │ 本轮是否被中断？       │                              │
│                    └──────┬───────────────┘                              │
│                           │                                              │
│                ┌──────────▼──────────┐                                   │
│                │                     │                                   │
│          ┌─────▼─────┐     ┌────────▼────────┐                           │
│          │ 跳过不保存  │     │ nudge 标志为真？  │                          │
│          └───────────┘     └────────┬────────┘                           │
│                                     │                                    │
│                          ┌──────────▼──────────┐                        │
│                          │ 后台 review agent     │ ← fork AIAgent       │
│                          │ 分析对话历史          │    16 轮上限          │
│                          │ 调用 memory 工具      │ ← 若发现值得保存      │
│                          │ 写入共享 MemoryStore   │                       │
│                          └─────────────────────┘                        │
│                                                                          │
│  ┌──────────────────────────────────────┐                               │
│  │ 每次写入后的统一流程                     │                              │
│  │ 1. MemoryStore.add() / replace()      │                              │
│  │    → 安全检查（注入/泄露检测）          │                              │
│  │    → 文件锁下重读磁盘（防跨会话冲突）     │                              │
│  │    → 字符预算检查                      │                              │
│  │    → 原子写入（临时文件 + fsync + 重命名）│                             │
│  │ 2. Bridge: MemoryManager.on_memory_write()                          │
│  │    → 通知外部 Provider 同步写入          │                             │
│  └──────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 4.6 保存时的内部执行流程

当 Agent 调用 `memory(action="add", target="memory", content="...")` 时：

```python
# tools/memory_tool.py:224-267
def add(self, target: str, content: str) -> Dict[str, Any]:
    content = content.strip()
    if not content:
        return {"success": False, "error": "Content cannot be empty."}

    # 1️⃣ 安全检查：注入/泄露检测
    scan_error = _scan_memory_content(content)
    if scan_error:
        return {"success": False, "error": scan_error}

    # 2️⃣ 文件锁（防多进程并发写入）
    with self._file_lock(self._path_for(target)):
        # 3️⃣ 重读磁盘最新状态
        self._reload_target(target)
        entries = self._entries_for(target)

        # 4️⃣ 去重检查
        if content in entries:
            return {"success": True, "message": "Entry already exists."}

        # 5️⃣ 字符预算检查
        limit = self._char_limit(target)  # memory=2200, user=1375
        new_total = len(ENTRY_DELIMITER.join(entries + [content]))
        if new_total > limit:
            return {"success": False, "error": f"Memory at {current}/{limit} chars..."}

        # 6️⃣ 追加并持久化
        entries.append(content)
        self._set_entries(target, entries)
        self.save_to_disk(target)  # 原子写入
```

**保存后**，在工具调用分发器中：

```python
# run_agent.py:10290-10314
if function_name == "memory":
    result = memory_tool(action, target, content, store=self._memory_store)

    # 🔄 桥接：通知外部 Provider
    if self._memory_manager and action in ("add", "replace"):
        self._memory_manager.on_memory_write(
            action, target, content,
            metadata=self._build_memory_write_metadata(
                task_id=..., tool_call_id=...,
            ),
        )
```

---

### 4.7 Skill 保存 vs Memory 保存的边界

| 维度 | 记忆 (memory) | 技能 (skill) |
|------|--------------|-------------|
| **保存目标** | `~/.hermes/memories/MEMORY.md` / `USER.md` | `~/.hermes/skills/<category>/<name>/SKILL.md` |
| **保存内容** | 用户是谁、环境事实、工具怪癖 | 怎么做某类任务：步骤、陷阱、模板 |
| **格式要求** | 简短声明式事实（"用户喜欢 pytest"） | 完整 SKILL.md 含 frontmatter + markdown |
| **触发条件** | 用户的偏好/纠正、值得跨会话保存的事实 | 复杂任务完成（5+ 工具调用）、用户纠正 workflow、发现新流程 |
| **生命周期** | 由模型自行决定更新/删除 | Agent 自动创建，Curator 自动维护（闲置→归档） |
| **注入方式** | 冻结快照注入系统提示（整个会话不变） | `/skill` 命令或 `-s` 标志加载 |

**关键决策规则**（来自 `run_agent.py:3943-3948`）：

> "User-preference embedding: when the user expressed a style/format/workflow preference, the update belongs in the SKILL.md body, not just in memory. Memory captures 'who the user is'; skills capture 'how to do this class of task for this user'."

---

### 4.8 Nudge 参数配置

```yaml
memory:
  memory_enabled: true       # 启用 MEMORY.md
  user_profile_enabled: true # 启用 USER.md
  memory_char_limit: 2200    # MEMORY.md 字符上限
  user_char_limit: 1375      # USER.md 字符上限
  nudge_interval: 10         # ⚡ 关键：每多少轮触发一次 memory review

skills:
  creation_nudge_interval: 10  # 每多少轮工具调用触发一次 skill review
```

设置 `nudge_interval: 0` 或 `creation_nudge_interval: 0` 可**完全禁用**对应的后台 review。

---

### 4.9 验证：计数器命中流程源码级确认

```python
# run_agent.py:11611 — 每轮开始时递增 nudge 计数器
if (self._memory_nudge_interval > 0              # 默认 10
    and "memory" in self.valid_tool_names        # memory 工具已注册
    and self._memory_store):                      # MemoryStore 已初始化
    self._turns_since_memory += 1                 # +1
    if self._turns_since_memory >= self._memory_nudge_interval:  # 10 轮到了
        _should_review_memory = True
        self._turns_since_memory = 0              # 重置
```

```python
# run_agent.py:15128-15153 — 本轮完全结束后
if final_response and not interrupted and (_should_review_memory or _should_review_skills):
    self._spawn_background_review(messages_snapshot, review_memory, review_skills)
```

### 4.10 总结

Hermes Agent 的 memory 保存不是单一路径的 "if-then" 判断，而是 **3 层互补的决策网络**：

```
1. 前置引导层 （静态） → 系统提示持续教育 LLM 何时保存
                               ↓
2. 主动触发层 （动态） → Agent 在对话中自行判断并调用 memory 工具
                               ↓
3. 后台兜底层 （周期性）→ 每 N 轮自动 fork review agent 检查遗漏
```

其中后台 review 是**最具特色的设计**——即使 Agent 忙于处理用户请求而"忘记"保存重要信息，系统也会在后台替它完成记忆管理，并输出 `💾 Self-improvement review: ...` 摘要通知用户。
