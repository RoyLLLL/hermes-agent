# Hermes Agent 内存系统深度分析文档

> **仓库**: [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)  
> **分析日期**: 2026-05-10  
> **覆盖范围**: `agent/memory_provider.py`, `agent/memory_manager.py`, `tools/memory_tool.py`, `plugins/memory/__init__.py`, `run_agent.py`, `agent/prompt_builder.py`, `hermes_cli/memory_setup.py`, `hermes_cli/main.py`, `plugins/memory/honcho/__init__.py`

---

## 一、总体架构概览

Hermes Agent 的内存系统是一个**双层架构**，同时运行两套独立的内存机制：

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
2. **单外部 Provider 约束** — 同一时间只能运行一个外部 Provider（防工具 Schema 冲突）
3. **失败隔离** — 一个 Provider 的异常不会影响另一个
4. **内置存储是基础** — 始终可用，外部 Provider 是增强

---

## 二、核心模块源码分析

### 2.1 MemoryProvider 抽象基类 (`agent/memory_provider.py`)

这是整个外部内存 Provider 体系的**抽象基类**（ABC），定义了完整的生命周期接口。所有外部 Provider 插件都必须实现它。

```python
"""agent/memory_provider.py — 抽象基类 (ABC)"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight')."""

    # ==== 核心生命周期（必须实现） ====

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if configured, has credentials, and is ready.
        Should not make network calls — just check config and installed deps."""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """Called once at agent startup.
        kwargs always include: hermes_home, platform
        kwargs may include: agent_context, agent_identity, user_id, chat_id, etc."""

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas in OpenAI function-calling format.
        Return empty list if no tools (context-only provider)."""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call. Must return a JSON string."""

    # ==== 可选实现 ====

    def system_prompt_block(self) -> str:
        """Static text for the system prompt. Return empty string to skip."""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn.
        Called before each API call. Return formatted text or empty string."""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background recall for the NEXT turn."""

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Persist a completed turn. Should be non-blocking."""

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # ==== 可选钩子（扩展点） ====

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at start of each turn."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends (exit, /reset, timeout)."""

    def on_session_switch(self, new_session_id: str, *,
                          parent_session_id: str = "", reset: bool = False,
                          **kwargs) -> None:
        """Called on /resume, /branch, /reset, /new, and context compression."""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called before context compression. Return text to preserve in summary."""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Called on the PARENT agent when a subagent completes."""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when the built-in memory tool writes. Used to mirror writes."""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Config fields for 'hermes memory setup'. Each field: key, description,
        secret, required, default, choices, url, env_var."""

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to native location."""
```

**关键设计细节**：

- `initialize()` 接收的 `kwargs` 中 `agent_context` 区分 `"primary"` / `"subagent"` / `"cron"` / `"flush"`——cron 和非 primary 上下文**不应写入用户记忆**
- `on_session_switch()` 的 `reset=True` 表示真正的 /reset 清空，`False` 表示 /resume 或 /branch
- `on_memory_write()` 实现与内置存储的**双向桥接**
- `on_delegation()` 让父 Agent 的 Provider 能观察到子任务的结果

---

### 2.2 MemoryManager 编排器 (`agent/memory_manager.py`)

核心编排类，管理内置 + 外部 Provider 的完整生命周期。关键源码：

```python
"""agent/memory_manager.py — 编排器"""

class MemoryManager:
    """Orchestrates the built-in provider plus at most one external provider."""

    def __init__(self) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False

    # ── Provider 注册 ──────────────────────────────────────

    def add_provider(self, provider: MemoryProvider) -> None:
        """Only ONE external (non-builtin) provider is allowed."""
        is_builtin = provider.name == "builtin"
        if not is_builtin:
            if self._has_external:
                # 拒绝第二个外部 Provider
                return
            self._has_external = True
        self._providers.append(provider)
        # 索引 tool name → provider
        for schema in provider.get_tool_schemas():
            tool_name = schema.get("name", "")
            self._tool_to_provider[tool_name] = provider

    # ── 系统提示 ───────────────────────────────────────────

    def build_system_prompt(self) -> str:
        """Collect system_prompt_block() from all providers."""
        blocks = []
        for provider in self._providers:
            try:
                block = provider.system_prompt_block()
                if block and block.strip():
                    blocks.append(block)
            except Exception as e:
                logger.warning("Provider '%s' system_prompt_block failed: %s",
                               provider.name, e)
        return "\n\n".join(blocks)

    # ── 预取 / 回顾 ────────────────────────────────────────

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context from all providers. Failures isolated."""
        parts = []
        for provider in self._providers:
            try:
                result = provider.prefetch(query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug("Provider '%s' prefetch failed: %s", provider.name, e)
        return "\n\n".join(parts)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch for the next turn."""

    # ── 同步 ────────────────────────────────────────────────

    def sync_all(self, user_content: str, assistant_content: str,
                 *, session_id: str = "") -> None:
        """Sync a completed turn to all providers."""

    # ── 工具路由 ────────────────────────────────────────────

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Collect deduplicated tool schemas from all providers."""
        schemas = []
        seen = set()
        for provider in self._providers:
            for schema in provider.get_tool_schemas():
                name = schema.get("name", "")
                if name and name not in seen:
                    schemas.append(schema)
                    seen.add(name)
        return schemas

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_to_provider

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Route tool call to the correct provider."""
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return tool_error(f"No memory provider handles tool '{tool_name}'")
        return provider.handle_tool_call(tool_name, args, **kwargs)

    # ── 生命周期广播 ──────────────────────────────────────

    def initialize_all(self, session_id: str, **kwargs) -> None:
        if "hermes_home" not in kwargs:
            from hermes_constants import get_hermes_home
            kwargs["hermes_home"] = str(get_hermes_home())
        for provider in self._providers:
            provider.initialize(session_id=session_id, **kwargs)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None: ...
    def on_session_end(self, messages: List[Dict[str, Any]]) -> None: ...
    def on_session_switch(self, new_session_id, *, parent_session_id="",
                          reset=False, **kwargs) -> None: ...
    def on_pre_compress(self, messages) -> str: ...
    def on_memory_write(self, action, target, content, metadata=None) -> None: ...
    def on_delegation(self, task, result, *, child_session_id="", **kwargs) -> None: ...
    def shutdown_all(self) -> None: ...
```

#### StreamingContextScrubber（特殊工具）

处理流式输出中 `<memory-context>` 标签拆分到多个 chunk 的问题：

```python
class StreamingContextScrubber:
    """Stateful scrubber for streaming text that may contain split memory-context spans.
    
    The one-shot sanitize_context regex cannot survive chunk boundaries:
    a <memory-context> opened in one delta and closed in a later delta
    leaks its payload because the non-greedy regex needs both tags in one string."""
    
    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    def __init__(self):
        self._in_span: bool = False
        self._buf: str = ""

    def feed(self, text: str) -> str:
        """Return visible portion after scrubbing. Holds back partial-tag fragments."""
        # 状态机：out-of-span → 查找 OPEN_TAG → 丢弃内容 → 查找 CLOSE_TAG → 回到 out-of-span
        # 每次 feed 跨 chunk 边界时，_max_partial_suffix() 检测标签前缀
        ...

    def flush(self) -> str:
        """Emit held-back buffer at end-of-stream.
        If still in unterminated span, discard remaining content
        (leaking partial memory context is worse than truncated answer)."""
```

**核心策略**：宁可丢弃内容也不泄漏部分 memory context，因为泄漏错误上下文比截断回答更危险。

---

### 2.3 MemoryStore 内置存储 (`tools/memory_tool.py`)

提供**文件级持久化**的键值存储，两个独立文件：

```
~/.hermes/memories/
├── MEMORY.md    → agent的个人笔记/环境事实/经验教训
└── USER.md      → 用户画像（偏好/风格/习惯）
```

#### 完整类结构源码

```python
"""tools/memory_tool.py — 内置文件存储"""

from __future__ import annotations
import json, logging, os, re, tempfile
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional
from utils import atomic_replace

ENTRY_DELIMITER = "\n§\n"

# ── 安全扫描 ──────────────────────────────────────────────

_MEMORY_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # 泄露检测
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # 后门检测
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/.hermes/\.env', "hermes_env"),
]

_INVISIBLE_CHARS = {'\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
                    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e'}

def _scan_memory_content(content: str) -> Optional[str]:
    """Scan for injection/exfil patterns. Returns error string if blocked."""
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: invisible unicode U+{ord(char):04X}"

    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: threat pattern '{pid}'"

    return None


class MemoryStore:
    """Bounded curated memory with file persistence. One instance per AIAgent.
    
    Maintains TWO parallel states:
      - _system_prompt_snapshot: frozen at load time, for system prompt injection
      - memory_entries / user_entries: live state, mutated by tool calls
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot."""
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Capture frozen snapshot
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    # ── CRUD ─────────────────────────────────────────────────

    def add(self, target: str, content: str) -> Dict[str, Any]:
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # 1️⃣ 安全检查
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # 2️⃣ 重读磁盘（防跨会话冲突）
            self._reload_target(target)
            entries = self._entries_for(target)

            # 3️⃣ 去重
            if content in entries:
                return {"success": True, "message": "Entry already exists."}

            # 4️⃣ 字符预算检查
            limit = self._char_limit(target)  # memory=2200, user=1375
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))
            if new_total > limit:
                current = self._char_count(target)
                return {"success": False,
                        "error": f"Memory at {current:,}/{limit:,} chars. Adding exceeds limit."}

            # 5️⃣ 持久化
            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return {"success": True, "message": "Entry added.",
                "entries": self._entries_for(target)}

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace with new_content."""
        # 安全检查 → 文件锁 → 重读 → 子串匹配 → 唯一性检查 → 预算检查 → 替换

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, remove it."""
        # 文件锁 → 重读 → 子串匹配 → 唯一性检查 → 删除

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Return the FROZEN snapshot. NEVER the live state.
        
        This keeps the system prompt stable across all turns,
        preserving the LLM provider's prefix cache."""
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # ── 原子写入 ─────────────────────────────────────────

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Atomic temp-file + rename write.
        
        Previous implementation used open("w") + flock, but "w" truncates
        BEFORE the lock is acquired — readers see an empty file.
        Atomic rename avoids this entirely."""
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".mem_"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)

    # ── 系统提示渲染 ──────────────────────────────────────

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render with header and usage indicator."""
        if not entries:
            return ""
        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        return f"{'═' * 46}\n{header}\n{'═' * 46}\n{content}"
```

#### 关键设计决策

**1. 冻结快照模式 (Frozen Snapshot Pattern)**

```
┌──── 会话开始 ────┐     ┌─── 会话期间 ───┐     ┌─── 下次会话 ──┐
│ load_from_disk() │────▶│ system_prompt  │────▶│ load 新快照   │
│ 捕获冻结快照      │     │ 不变(缓存命中)  │     │ 反映上次写入   │
│ 写入操作 → 磁盘   │     │ tool返回实时    │     │               │
└──────────────────┘     └────────────────┘     └───────────────┘
```

效果：
- 系统提示在会话期间**完全不变** → LLM Provider 的前缀缓存命中率最高
- 内存写入立即持久化到磁盘，但当前会话不感知
- 下次会话启动时才反映最新状态

**2. 字符预算而非 Token 预算**
- `MEMORY.md`: 默认 2200 字符  
- `USER.md`: 默认 1375 字符
- 选择字符而非 Token：**模型无关**

---

### 2.4 内置 Memory 工具 Schema 与注册

工具 Schema 本身就是一份**完整的 LLM 行为规范**——告诉模型什么时候该写、写什么、不写什么：

```python
"""tools/memory_tool.py — 工具 Schema 与注册"""

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into every turn, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail "
        "(name, role, timezone, coding style)\n"
        "- You discover something about the environment "
        "(OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, save it as a skill with the skill_manage tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"],
                       "description": "The action to perform."},
            "target": {"type": "string", "enum": ["memory", "user"],
                       "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."},
            "content": {"type": "string",
                        "description": "The entry content. Required for 'add' and 'replace'."},
            "old_text": {"type": "string",
                         "description": "Short unique substring identifying the entry to replace or remove."},
        },
        "required": ["action", "target"],
    },
}

# ── 注册到工具系统 ────────────────────────────────────

from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=lambda: True,  # 永远可用
    emoji="🧠",
)
```

---

### 2.5 外部 Provider 插件加载系统 (`plugins/memory/__init__.py`)

插件发现系统扫描两个目录并加载 Provider：

```python
"""plugins/memory/__init__.py — 插件发现与加载"""

from pathlib import Path
from typing import List, Optional, Tuple

_MEMORY_PLUGINS_DIR = Path(__file__).parent

def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """Yield (name, path) for discovered providers. Bundled takes precedence."""
    seen: set = set()
    dirs: List[Tuple[str, Path]] = []

    # 1. 内置: plugins/memory/<name>/
    for child in sorted(_MEMORY_PLUGINS_DIR.iterdir()):
        if child.is_dir() and not child.name.startswith(("_", ".")):
            if (child / "__init__.py").exists():
                seen.add(child.name)
                dirs.append((child.name, child))

    # 2. 用户: $HERMES_HOME/plugins/<name>/
    user_dir = _get_user_plugins_dir()
    if user_dir:
        for child in sorted(user_dir.iterdir()):
            if child.name in seen:
                continue  # 内置优先
            if _is_memory_provider_dir(child):
                dirs.append((child.name, child))

    return dirs


def load_memory_provider(name: str) -> Optional["MemoryProvider"]:
    """Load and return a MemoryProvider instance by name."""
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        return None

    # 尝试 1: register(ctx) 模式（插件标准）
    if hasattr(mod, "register"):
        collector = _ProviderCollector()
        try:
            mod.register(collector)
            if collector.provider:
                return collector.provider
        except Exception:
            pass

    # 尝试 2: MemoryProvider 子类自动发现
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, MemoryProvider)
                and attr is not MemoryProvider):
            return attr()

    return None


class _ProviderCollector:
    """Fake plugin context that captures register_memory_provider calls."""
    def __init__(self):
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider
```

**CLI 集成**：`discover_plugin_cli_commands()` 仅加载当前激活 Provider 的 `cli.py`，注册为 `hermes <provider_name>` 子命令。

---

### 2.6 Honcho Provider 示例

以 Honcho 为例，展示外部 Provider 的完整实现模式：

```python
"""plugins/memory/honcho/__init__.py — Honcho Provider 示例 (简化)"""

from agent.memory_provider import MemoryProvider

class HonchoMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "honcho"

    def is_available(self) -> bool:
        return bool(os.getenv("HONCHO_API_KEY") or self._load_config())

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._user_id = kwargs.get("user_id", "default")
        # 启动 Honcho 客户端连接
        self._client = HonchoClient(...)

    def system_prompt_block(self) -> str:
        return "You have access to Honcho AI-native memory for cross-session user modeling."

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONTEXT_SCHEMA, CONCLUDE_SCHEMA]

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Semantic recall relevant context for the upcoming turn."""
        return self._client.search(query, limit=3, user_id=self._user_id)

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Store the turn in Honcho for future recall."""
        self._client.store_turn(user_content, assistant_content)

    def on_memory_write(self, action, target, content, metadata=None):
        """Mirror built-in memory writes to Honcho backend."""
        ...

    def shutdown(self) -> None:
        self._client.close()


# 注册点
def register(ctx):
    ctx.register_memory_provider(HonchoMemoryProvider())
```

Honcho 暴露 4 个工具：
| 工具名 | 作用 |
|--------|------|
| `honcho_profile` | 读取/更新用户画像卡片 |
| `honcho_search` | 语义搜索过去对话记录 |
| `honcho_reasoning` | LLM 合成的用户洞察 |
| `honcho_conclude` | 持久化用户结论 |

---

## 三、Agent 主循环中的内存集成 (`run_agent.py`)

### 3.1 初始化流程

```python
"""run_agent.py — AIAgent.__init__ 中内存初始化"""

# 内置 MemoryStore
self._memory_store = None
self._memory_enabled = False
self._user_profile_enabled = False
self._memory_nudge_interval = 10
self._turns_since_memory = 0
self._iters_since_skill = 0

if not skip_memory:
    mem_config = _agent_cfg.get("memory", {})
    self._memory_enabled = mem_config.get("memory_enabled", False)
    self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
    self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))

    if self._memory_enabled or self._user_profile_enabled:
        from tools.memory_tool import MemoryStore
        self._memory_store = MemoryStore(
            memory_char_limit=mem_config.get("memory_char_limit", 2200),
            user_char_limit=mem_config.get("user_char_limit", 1375),
        )
        self._memory_store.load_from_disk()


# 外部 MemoryManager
self._memory_manager = None
if not skip_memory:
    _mem_provider_name = mem_config.get("provider", "")

    if _mem_provider_name:
        from agent.memory_manager import MemoryManager as _MemoryManager
        from plugins.memory import load_memory_provider as _load_mem

        self._memory_manager = _MemoryManager()
        _mp = _load_mem(_mem_provider_name)
        if _mp and _mp.is_available():
            self._memory_manager.add_provider(_mp)

        if self._memory_manager.providers:
            _init_kwargs = {
                "session_id": self.session_id,
                "platform": platform or "cli",
                "hermes_home": str(get_hermes_home()),
                "agent_context": "primary",
                "user_id": self._user_id,
                "user_name": self._user_name,
                "agent_identity": get_active_profile_name(),
                # ... 更多上下文
            }
            self._memory_manager.initialize_all(**_init_kwargs)
            logger.info("Memory provider '%s' activated", _mem_provider_name)
```

### 3.2 系统提示组装

```python
"""run_agent.py — _build_system_prompt 中内存注入"""

# 内置 MEMORY.md 冻结快照
if self._memory_store:
    if self._memory_enabled:
        mem_block = self._memory_store.format_for_system_prompt("memory")
        if mem_block:
            prompt_parts.append(mem_block)

    if self._user_profile_enabled:
        user_block = self._memory_store.format_for_system_prompt("user")
        if user_block:
            prompt_parts.append(user_block)

# 外部 Provider 系统提示块
if self._memory_manager:
    _ext_mem_block = self._memory_manager.build_system_prompt()
    if _ext_mem_block:
        prompt_parts.append(_ext_mem_block)
```

### 3.3 工具调用分发

```python
"""run_agent.py — handle_function_call 中 memory 路由"""

elif function_name == "memory":
    target = function_args.get("target", "memory")
    from tools.memory_tool import memory_tool as _memory_tool
    result = _memory_tool(
        action=function_args.get("action"),
        target=target,
        content=function_args.get("content"),
        old_text=function_args.get("old_text"),
        store=self._memory_store,
    )

    # 🔄 Bridge: 通知外部 Provider 同步写入
    if self._memory_manager and function_args.get("action") in ("add", "replace"):
        try:
            self._memory_manager.on_memory_write(
                function_args.get("action", ""),
                target,
                function_args.get("content", ""),
                metadata=self._build_memory_write_metadata(
                    task_id=effective_task_id,
                    tool_call_id=tool_call_id,
                ),
            )
        except Exception:
            pass
    return result

# 外部 Provider 工具路由
elif self._memory_manager and self._memory_manager.has_tool(function_name):
    return self._memory_manager.handle_tool_call(function_name, function_args)
```

### 3.4 完整生命周期事件序列

```
┌───────────────────────────────────────────────────┐
│             完整生命周期事件序列                       │
├───────────────────────────────────────────────────┤
│  AIAgent 初始化时:                                 │
│    MemoryStore.load_from_disk()                     │
│    MemoryManager.initialize_all(session_id, ...)    │
│                                                     │
│  每轮用户消息开始时:                                 │
│    · 递增 _turns_since_memory nudge 计数器           │
│    · MemoryManager.on_turn_start(turn_number, msg)   │
│    · MemoryManager.prefetch_all(query)               │
│      → 预取上下文注入本轮系统提示                      │
│                                                     │
│  每轮工具调用分发:                                   │
│    · "memory" → MemoryStore + on_memory_write 桥接    │
│    · provider工具 → MemoryManager.handle_tool_call() │
│    · 调用 memory 时 → 计数器归零                     │
│                                                     │
│  每轮响应交付后:                                     │
│    · MemoryManager.sync_all(user_msg, asst_response) │
│    · MemoryManager.queue_prefetch_all(user_msg)      │
│    · 若 nudge 触发: spawn_background_review()        │
│                                                     │
│  上下文压缩前:                                       │
│    · MemoryManager.on_pre_compress(messages)         │
│                                                     │
│  /resume / /branch / /reset / 压缩时:               │
│    · MemoryManager.on_session_switch(new_id, reset)  │
│                                                     │
│  子代理完成时:                                       │
│    · MemoryManager.on_delegation(task, result, id)   │
│                                                     │
│  会话结束时:                                         │
│    · MemoryManager.on_session_end(messages)          │
│    · MemoryManager.shutdown_all()                    │
└───────────────────────────────────────────────────┘
```

---

## 四、记忆保存流程深度分析

### 4.1 保存记忆的三大触发机制

```
保存记忆 = Agent 的"自省决策" × 3 条独立路径
```

| 触发路径 | 触发条件 | 谁决策 | 触发时机 | 对应代码 |
|----------|----------|--------|----------|----------|
| **① 前置引导** | 系统提示中 MEMORY_GUIDANCE 持续告知 | Agent 自主（LLM 自行判断） | 任一回合中 | `prompt_builder.py:150` |
| **② 周期性 nudge** | 每 N 轮（默认 10）自动触发 | 后台 review agent 判断 | **响应交付后**异步执行 | `run_agent.py:11607-11617` |
| **③ 模型自主调用** | Agent 认为有值得保存的信息 | Agent 自主 | 任意工具调用回合 | `memory_tool.py` 的 tool schema |

> 三条路径**并行存在、互补**。若 Agent 在对话中自主调用了 `memory` 工具，nudge 计数器归零，不会重复触发后台 review。

---

### 4.2 触发路径①：系统提示前置引导

```python
"""agent/prompt_builder.py:150 — MEMORY_GUIDANCE"""

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Save durable facts using the memory "
    "tool: user preferences, environment details, tool quirks, and stable conventions. "
    "Memory is injected into every turn, so keep it compact and focused on facts that "
    "will still matter later.\n"
    "Prioritize what reduces future user steering — the most valuable memory is one "
    "that prevents the user from having to correct or remind you again. "
    "User preferences and recurring corrections matter more than procedural task details.\n"
    "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
    "state to memory; use session_search to recall those from past transcripts. "
    "Specifically: do not record PR numbers, issue numbers, commit SHAs, 'fixed bug X', "
    "'submitted PR Y', 'Phase N done', file counts, or any artifact that will be stale "
    "in 7 days. If a fact will be stale in a week, it does not belong in memory. "
    "If you've discovered a new way to do something, solved a problem that could be "
    "necessary later, save it as a skill with the skill_manage tool.\n"
    "Write memories as declarative facts, not instructions to yourself. "
    "'User prefers concise responses' ✓ — 'Always respond concisely' ✗. "
    "'Project uses pytest with xdist' ✓ — 'Run tests with pytest -n 4' ✗. "
    "Imperative phrasing gets re-read as a directive in later sessions and can "
    "cause repeated work or override the user's current request. Procedures and "
    "workflows belong in skills, not memory."
)
```

**5 个"应该保存"的信号：**

| 信号 | 原文 | 举例 |
|------|------|------|
| S1. 用户纠正 | "User corrects you or says 'remember this'" | "不要用这么啰嗦的格式" |
| S2. 用户分享偏好 | "User shares a preference, habit, or personal detail" | "我喜欢用 pytest" |
| S3. 环境发现 | "You discover something about the environment" | "OS 是 Ubuntu 24.04" |
| S4. 约定/API 怪癖 | "You learn a convention, API quirk, or workflow" | "这个 API 需要 X 头" |
| S5. 稳定事实 | "You identify a stable fact that will be useful again" | "项目使用 Poetry 管理" |

**4 个"不应该保存"的信号：**

| 反信号 | 原因 |
|--------|------|
| 任务进度 | PR 编号、commit SHA、"修复了 bug X" |
| 临时 TODO | 使用 session_search 来回忆 |
| 7 天内会过期的 | 一个事实一周后过时 → 不属于记忆 |
| 指令式语句 | "写声明式事实，不是指令自己" |

**优先级：** `用户偏好/纠正 > 环境事实 > 程序性知识`

---

### 4.3 触发路径②：周期性 Nudge 后台 Review

这是**最精巧的机制**——Agent 不必每轮自我判断，系统自动在后台检查。

#### 触发条件链（5 条件必须全部满足）

```python
"""run_agent.py:11607-11617 — 每轮开始时递增 nudge 计数器"""
_should_review_memory = False
if (self._memory_nudge_interval > 0                         # (1) nudge 间隔 > 0（默认 10）
    and "memory" in self.valid_tool_names                    # (2) memory 工具已启用
    and self._memory_store):                                 # (3) MemoryStore 已初始化
    self._turns_since_memory += 1                            # (4) 递增
    if self._turns_since_memory >= self._memory_nudge_interval:  # (5) 达到阈值
        _should_review_memory = True
        self._turns_since_memory = 0                         # 重置
```

#### 执行时机——响应交付后异步执行

```python
"""run_agent.py:15128-15153 — 本轮完全结束后"""
if final_response and not interrupted and (_should_review_memory or _should_review_skills):
    try:
        self._spawn_background_review(
            messages_snapshot=list(messages),    # 当前对话快照
            review_memory=_should_review_memory,
            review_skills=_should_review_skills,
        )
    except Exception:
        pass  # background review is best-effort
```

#### 后台 Review Agent 完整生命周期

```python
"""run_agent.py:4117-4236 — _spawn_background_review"""

def _spawn_background_review(self, messages_snapshot, review_memory, review_skills):
    # (1) 选定 review prompt
    if review_memory and review_skills:
        prompt = self._COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = self._MEMORY_REVIEW_PROMPT
    else:
        prompt = self._SKILL_REVIEW_PROMPT

    # (2) 后台线程 fork 完整 AIAgent
    def _run_review():
        review_agent = AIAgent(
            model=self.model,                       # 继承父 agent 的模型
            max_iterations=16,                      # 最多 16 轮工具调用
            quiet_mode=True,                        # 静默模式
            enabled_toolsets=["memory", "skills"],   # 只暴露 memory/skill 工具
        )
        # 关键：共享同一个 MemoryStore 实例
        review_agent._memory_store = self._memory_store
        # 关闭自身的 nudge（防递归）
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0

        # (3) review agent 分析对话历史并行写入
        review_agent.run_conversation(
            user_message=prompt,
            conversation_history=messages_snapshot,
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

#### Review Prompt 决策逻辑

```python
"""run_agent.py:3871 — _MEMORY_REVIEW_PROMPT"""
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
"""run_agent.py:10386-10390 — 工具调用时重置"""
if function_name == "memory":
    self._turns_since_memory = 0
elif function_name == "skill_manage":
    self._iters_since_skill = 0
```

> **关键推论**：如果 Agent 在对话中主动保存了记忆，后台 review 就不会触发——避免重复劳动。

---

### 4.4 触发路径③：模型自主调用 memory 工具

最直接的路径——Agent 在对话中自行判断并调用。工具 Schema 的 description 本身就是保存指南（见 2.4 节）。

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
│   │ 判断 nudge 触发   │ ◄── _turns_since_memory ≥ nudge_interval(10)    │
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

### 4.6 写入内部完整执行流程

```python
"""tools/memory_tool.py:224-267 — add() 方法的完整 6 步流程"""

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
            return {"success": False, "error": f"...exceeds limit: {current}/{limit}"}

        # 6️⃣ 追加并持久化（原子写入）
        entries.append(content)
        self._set_entries(target, entries)
        self.save_to_disk(target)

    return {"success": True, "message": "Entry added.", "entries": entries}
```

写入后，工具调用分发器同步执行桥接：

```python
"""run_agent.py:10290-10314 — memory 工具调用后的桥接"""
if function_name == "memory":
    result = memory_tool(action, target, content, store=self._memory_store)
    # 🔄 通知外部 Provider
    if self._memory_manager and action in ("add", "replace"):
        self._memory_manager.on_memory_write(
            action, target, content,
            metadata=self._build_memory_write_metadata(task_id=..., tool_call_id=...),
        )
```

---

### 4.7 Skill 保存 vs Memory 保存的边界

| 维度 | 记忆 (memory) | 技能 (skill) |
|------|--------------|-------------|
| **保存目标** | `~/.hermes/memories/MEMORY.md` / `USER.md` | `~/.hermes/skills/<category>/<name>/SKILL.md` |
| **保存内容** | 用户是谁、环境事实、工具怪癖 | 怎么做某类任务：步骤、陷阱、模板 |
| **格式要求** | 简短声明式事实 | 完整 SKILL.md 含 frontmatter + markdown |
| **触发条件** | 用户的偏好/纠正、值得跨会话的事实 | 复杂任务（5+ 工具调用）、用户纠正 workflow |
| **生命周期** | 由模型自行更新/删除 | Agent 自动创建，Curator 自动维护 |
| **注入方式** | 冻结快照注入系统提示 | `/skill` 命令或 `-s` 标志加载 |

**关键决策规则**（来自 `run_agent.py:3943-3948`）：
> "Memory captures 'who the user is'; skills capture 'how to do this class of task for this user'."

---

### 4.8 记忆系统的完整配置

```yaml
memory:
  memory_enabled: true       # 启用 MEMORY.md
  user_profile_enabled: true # 启用 USER.md
  memory_char_limit: 2200    # MEMORY.md 字符上限
  user_char_limit: 1375      # USER.md 字符上限
  nudge_interval: 10         # ⚡ 每多少轮触发一次 memory review
  provider: ""               # 外部 Provider 名称（honcho/mem0/holographic/...）

skills:
  creation_nudge_interval: 10  # 每多少轮工具调用触发一次 skill review
```

设置 `nudge_interval: 0` 或 `creation_nudge_interval: 0` 可**完全禁用**后台 review。

---

## 五、CLI 命令 (`hermes_cli/main.py` + `memory_setup.py`)

```bash
hermes memory setup        # 交互式配置 Provider（curses UI 选择器 + 配置向导）
hermes memory status       # 显示当前内存 Provider 配置和状态
hermes memory off          # 禁用外部 Provider（仅保留内置）
hermes memory reset        # 擦除内置 MEMORY.md/USER.md
  hermes memory reset --target all | memory | user
```

`hermes memory setup` 的交互流程：

```
1. 扫描 plugins/memory/ 和 $HERMES_HOME/plugins/ 发现 Provider
2. 调用 find_provider_dir() + 读取 plugin.yaml 获取元信息
3. curses 界面让用户选择 Provider
4. 安装 pip 依赖（plugin.yaml 中的 pip_dependencies）
5. 遍历 get_config_schema() 的字段逐个提示用户输入
6. secrets 写入 .env，非 secrets 通过 save_config() 写入
7. 设置 memory.provider = <name> 到 config.yaml
```

---

## 六、核心设计模式总结

### 模式 1: 插件化 Provider + 单例约束
```
MemoryProvider(ABC) ←─ HonchoMemoryProvider
                    ←─ Mem0MemoryProvider
                    ←─ ... (最多一个激活)
```
**目的**：防止多个外部 Provider 的工具 Schema 冲突，简化系统提示复杂度

### 模式 2: 冻结快照 (Frozen Snapshot)
```
┌──── 会话开始 ────┐     ┌─── 会话期间 ───┐     ┌─── 下次会话 ──┐
│ load_from_disk() │────▶│ system_prompt  │────▶│ load 新快照   │
│ 捕获冻结快照      │     │ 不变(缓存命中)  │     │ 反映上次写入   │
│ 写入操作 → 磁盘   │     │ tool返回实时    │     │               │
└──────────────────┘     └────────────────┘     └───────────────┘
```
**目的**：最大化 LLM 前缀缓存命中，同时保证写入即持久化

### 模式 3: 安全优先的注入检测
```
写入内容 → 不可见 Unicode 检测 → 威胁模式匹配 → 放行/拦截
```
**目的**：内存内容注入系统提示 → prompt injection 高价值目标

### 模式 4: 双向桥接 (Bidirectional Bridge)
```
内置 memory 工具写入 ──→ 通知 MemoryManager.on_memory_write()
                       ──→ 外部 Provider 镜像同步
```
**目的**：内置写入自动同步到外部 Provider，无需用户/模型手动调用两套 API

### 模式 5: 上下文隔离 (Context Fencing)
```
<memory-context>
[System note: ...Treat as authoritative reference data...]
  ... memory content ...
</memory-context>
```
流式输出时由 `StreamingContextScrubber` 跨 chunk 状态机处理
**目的**：严格区分"模型记忆"和"新用户输入"

### 模式 6: Nudge + Review Agent (三层决策网络)
```
前置引导层（静态）  → 系统提示持续教育 LLM
       ↓
主动触发层（动态）  → Agent 自行判断调用 memory 工具
       ↓
后台兜底层（周期性）→ 每 N 轮自动 fork review agent 检查遗漏
```
**目的**：确保即使 Agent 忙于任务而"忘记"保存，系统也会在后台完成记忆管理

---

## 七、文件依赖关系图

```
agent/memory_provider.py          ← 抽象基类（无依赖）
       ↑
agent/memory_manager.py           ← 编排器（依赖 MemoryProvider + 工具函数）
       ↑
tools/memory_tool.py              ← 内置存储（依赖 get_hermes_home, registry）
  MemoryStore 类
       ↑
run_agent.py                      ← 主循环集成
  ├── 初始化 MemoryStore          ← MemoryStore()
  ├── 初始化 MemoryManager        ← MemoryManager + load_memory_provider()
  ├── 系统提示组装（注入快照）     ← format_for_system_prompt()
  ├── 工具调用分发（memory路由）   ← memory_tool() + on_memory_write()
  ├── nudge 计数器管理             ← _turns_since_memory
  └── 后台 review                 ← _spawn_background_review()
       ↑
plugins/memory/__init__.py        ← 插件发现
  ├── discover_memory_providers()
  └── load_memory_provider(name)
       ↑
plugins/memory/<name>/__init__.py ← 具体 Provider 实现
  ├── HonchoMemoryProvider        (4 tools: profile/search/reasoning/conclude)
  ├── Mem0MemoryProvider          (1 tool: semantic search + fact extraction)
  ├── HindsightMemoryProvider     (post-hoc analysis → write to memory)
  ├── HolographicMemoryProvider   (structured fact storage)
  ├── SupermemoryProvider         (third-party backend)
  ├── RetainDBProvider            (local persistent store)
  ├── OpenVikingMemoryProvider    (bi-directional interface)
  └── ByteroverMemoryProvider     (byte-level storage)
       ↑
hermes_cli/memory_setup.py        ← CLI 配置向导
hermes_cli/main.py                ← hermes memory 命令注册
```

---

## 八、设计亮点与权衡

### 亮点
1. **双层架构** — 简单场景用内置文件存储，复杂场景接入外部 Provider
2. **单 Provider 约束** — 避免工具 Schema 爆炸和语义冲突
3. **冻结快照 + 原子写入** — 兼顾缓存性能和写入安全性
4. **内嵌 Schema 文档** — 工具描述本身就是 LLM 行为指南，不依赖硬编码规则
5. **字符预算而非 Token 预算** — 模型无关的公平限制
6. **完整生命周期钩子** — 从初始化到会话切换、压缩、子任务完成全覆盖
7. **Nudge 后台 Review** — fork 独立 AIAgent 异步检查，不阻塞主流程
8. **安全扫描层** — 13 种威胁模式 + 不可见 Unicode 检测

### 需注意的权衡
1. **内置存储无语义搜索** — 全靠子串匹配，大集合效率下降
2. **字符预算较小** — 默认 2200+1375 字符，复杂场景可能不够
3. **冻结快照延迟** — 当前会话写入但看不到，需等下次会话
4. **无内置记忆合并/去重** — 靠模型自觉写入，重复条目仅去重同一字符串
5. **外部 Provider 仅限一个** — 无法同时使用多个外部记忆后端

---

*文档生成于 NousResearch/hermes-agent 仓库代码分析，覆盖 9 个核心源文件。*
