# Canonical Architecture

**The Authoritative Reference for Lilux System Architecture**

This document provides the single canonical diagram and layer definitions for the Lilux architecture. All other documents should be consistent with this reference.

---

## The Complete Lilux Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│              "Find email scripts and refactor the best one"                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONVERSATIONAL FRONTEND (1-4B)                          │
│                                                                              │
│  Models: Phi-3 Mini, Gemma 2 2B, Llama 3.2 3B, Qwen 2.5 3B                  │
│                                                                              │
│  Responsibilities:                                                           │
│  • Natural conversation and personality                                      │
│  • Intent detection and marker generation                                    │
│  • Result synthesis into natural language                                    │
│                                                                              │
│  Output: "I'll search for those! [TOOL: search for email scripts]"          │
│                                                                              │
│  Latency: 80-150ms │ Cost: $0 (local)                                       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   │ "[TOOL: search for email scripts]"
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INTENT ROUTING HARNESS                                  │
│                                                                              │
│  Responsibilities:                                                           │
│  • Intercept [TOOL: ...] markers from frontend stream                       │
│  • Route to FunctionGemma for translation                                   │
│  • Execute via MCP protocol                                                 │
│  • Inject [RESULT: ...] back into conversation                              │
│                                                                              │
│  Latency: <5ms overhead                                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
┌───────────────────────────────┐  ┌─────────────────────────────────────────┐
│   SEMANTIC DISCOVERY (opt)    │  │        FUNCTIONGEMMA ROUTER (270M)      │
│                               │  │                                         │
│  For 100+ directives:         │  │  Input: "search for email scripts"      │
│  • Vector search (10-20ms)    │  │                                         │
│  • Returns top 7 candidates   │  │  Output:                                │
│  • Scales to 1M+ directives   │  │  {                                      │
│                               │  │    "name": "search",                    │
│  Models: all-MiniLM-L6-v2     │  │    "arguments": {                       │
│         (22M embeddings)      │  │      "item_type": "script",             │
│                               │  │      "query": "email"                   │
│  Latency: 10-20ms             │  │    },                                   │
└───────────────┬───────────────┘  │    "confidence": 0.95                   │
                │                  │  }                                      │
                │ top 7 candidates │                                         │
                └────────────────► │  Latency: 40-80ms │ Cost: $0 (local)    │
                                   └──────────────────┬──────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MCP LAYER (UNCHANGED)                                │
│                                                                              │
│  Protocol: stdio / SSE (standard MCP)                                       │
│  API: call_tool(name, arguments)                                            │
│                                                                              │
│  Compatible Servers:                                                         │
│  • Kiwi MCP (search, load, execute)                                         │
│  • Filesystem MCP                                                           │
│  • Git MCP                                                                  │
│  • Any MCP server from the ecosystem                                        │
│                                                                              │
│  Latency: 5-50ms                                                            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KIWI MCP EXECUTION LAYER                                │
│                                                                              │
│  Three Core Primitives:                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │   SEARCH    │  │    LOAD     │  │   EXECUTE   │                          │
│  │             │  │             │  │             │                          │
│  │ Find items  │  │ Retrieve &  │  │ Run actions │                          │
│  │ by query    │  │ copy items  │  │ (run/create │                          │
│  │             │  │             │  │ /update/del)│                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
│                                                                              │
│  Directive Execution:                                                        │
│  • Parses directive XML                                                     │
│  • Routes steps to appropriate model tier                                   │
│  • Manages script execution in isolated venvs                               │
│                                                                              │
│  Latency: Variable (depends on directive complexity)                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   │ (Only for high-reasoning directives)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REASONING ENGINE (70B+ / Cloud)                           │
│                                                                              │
│  Models: Llama 3.3 70B, Claude Sonnet, GPT-4o, Gemini 2.0                   │
│                                                                              │
│  Invoked When:                                                               │
│  • Directive declares model_class="high-reasoning"                          │
│  • Router confidence < 70%                                                  │
│  • Complex multi-step planning required                                     │
│                                                                              │
│  Responsibilities:                                                           │
│  • Deep analysis and reasoning                                              │
│  • Complex code generation                                                  │
│  • Multi-step planning                                                      │
│                                                                              │
│  Latency: 500-2000ms │ Cost: $0.08-0.30 per invocation                      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION LAYER                                      │
│                                                                              │
│  • Python scripts in isolated venvs (100% deterministic)                    │
│  • API calls to external services                                           │
│  • Shell commands (sandboxed)                                               │
│  • File system operations                                                   │
│                                                                              │
│  Latency: Variable (depends on task)                                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │     REAL WORLD      │
                        │                     │
                        │ Files, APIs, DBs,   │
                        │ Infrastructure      │
                        └─────────────────────┘
```

---

## Layer Responsibilities Summary

| Layer              | Responsibility                            | Model/Tech         | Latency    |
| ------------------ | ----------------------------------------- | ------------------ | ---------- |
| **Frontend**       | Conversation, personality, intent markers | 1-4B LLM           | 80-150ms   |
| **Harness**        | Intercept markers, coordinate routing     | Python/code        | <5ms       |
| **Discovery**      | Semantic search for directives (optional) | Embeddings (22M)   | 10-20ms    |
| **Router**         | Intent → tool call translation            | FunctionGemma 270M | 40-80ms    |
| **MCP Layer**      | Standard tool protocol                    | MCP stdio/SSE      | 5-50ms     |
| **Kiwi Execution** | Directive parsing, script management      | Kiwi MCP server    | Variable   |
| **Reasoner**       | Complex reasoning (when needed)           | 70B+ / Cloud       | 500-2000ms |
| **Execution**      | Actual work (scripts, APIs)               | Python, shell      | Variable   |

---

## Architecture Configurations

### Dual-Model (Minimal)

```
User → Router (270M) → MCP → Reasoner (70B+) → Execution
```

- Reasoner handles both conversation AND complex reasoning
- Simpler deployment, fewer models
- Best for: Development, simple applications

### Multi-Net (Full)

```
User → Frontend (3B) → Router (270M) → MCP → [Reasoner if needed] → Execution
```

- Dedicated frontend for personality
- Router for fast tool calling
- Reasoner only when truly needed
- Best for: Production, personalized agents

---

## Data Flow Example

**User:** "Find email scripts and refactor the best one"

```
1. Frontend (3B) receives input
   └─► Output: "I'll search for those! [TOOL: search for email scripts]"

2. Harness intercepts [TOOL: ...]
   └─► Routes to FunctionGemma

3. Router (270M) translates intent
   └─► Output: {"name": "search", "arguments": {...}, "confidence": 0.95}

4. MCP Layer executes
   └─► Kiwi MCP search() returns: [email_enricher.py, email_validator.py]

5. Harness injects result
   └─► [RESULT: search | Found 2 scripts | email_enricher.py, email_validator.py]

6. Frontend continues
   └─► "Found 2 email scripts! Now refactoring... [TOOL: run refactor_script on email_enricher]"

7. Router translates
   └─► {"name": "execute", "arguments": {"action": "run", "item_id": "refactor_script", ...}}

8. Kiwi MCP loads directive
   └─► refactor_script declares model_class="high-reasoning"

9. Reasoner (70B) invoked
   └─► Performs deep code analysis, generates refactored version

10. Result flows back
    └─► Frontend synthesizes: "Done! Refactored email_enricher with 3 improvements..."
```

---

## Key Principles

### 1. MCP Unchanged

All MCP servers work unmodified. Only the routing layer changes.

### 2. Right-Sized Intelligence

Use the smallest model sufficient for each task:

- Conversation → 3B
- Routing → 270M
- Reasoning → 70B+ (only when needed)

### 3. Three Primitives

Everything is built on: **search**, **load**, **execute**

### 4. Deterministic Execution

Scripts are 100% deterministic. Only orchestration is probabilistic.

### 5. Local-First

Router runs locally for speed, privacy, and offline capability.

---

## Performance Targets

| Scenario                       | Target Latency | Achieved     |
| ------------------------------ | -------------- | ------------ |
| Simple tool call (router only) | <100ms         | 40-80ms ✓    |
| Tool call with search          | <150ms         | 60-100ms ✓   |
| Complex reasoning needed       | <3000ms        | 500-2000ms ✓ |
| End-to-end simple query        | <500ms         | 200-400ms ✓  |

---

_This is the canonical architecture. All documents should be consistent with this reference._

_Last updated: 2026-01-17_
