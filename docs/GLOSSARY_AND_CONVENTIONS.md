# Glossary and Conventions

**The Single Source of Truth for Lilux Terminology**

This document defines canonical terminology, formats, and conventions used throughout the Lilux documentation. All other documents should reference this file rather than re-defining terms.

---

## Core Concepts

### Lilux

**LLM + Linux + Lux (light)** — The AI-native operating system built on Kiwi MCP. Lilux reimagines computing where the primary operator is artificial intelligence.

### DOE Framework

**Directive-Orchestration-Execution** — The kernel architecture of Lilux:

- **Directive**: WHAT to do (natural language instructions)
- **Orchestration**: HOW to decide (AI reasoning layer)
- **Execution**: DO the work (deterministic scripts)

### Kiwi MCP

The Model Context Protocol server that serves as Lilux's kernel, providing the interface between AI agents and system resources.

---

## The Three Primitives

Lilux uses exactly **three core primitives**. All operations are built on these:

| Primitive   | Purpose                                               | Example                                            |
| ----------- | ----------------------------------------------------- | -------------------------------------------------- |
| **search**  | Find items (directives, scripts, knowledge)           | `search(item_type="script", query="email")`        |
| **load**    | Retrieve and optionally copy items                    | `load(item_id="email_enricher", source="project")` |
| **execute** | Run operations (run, create, update, delete, publish) | `execute(action="run", item_id="sync_directives")` |

> **Note:** `help` is implemented as a directive or convenience tool, not a core primitive.

---

## Architecture Components

### Model Sizes (Canonical Definitions)

| Component    | Size Class       | Examples                                    | Role                                        |
| ------------ | ---------------- | ------------------------------------------- | ------------------------------------------- |
| **Frontend** | 1-4B parameters  | Phi-3 Mini (3.8B), Gemma 2 2B, Llama 3.2 3B | Conversation + personality + intent markers |
| **Router**   | ~270M parameters | FunctionGemma 270M                          | Intent → tool call translation              |
| **Reasoner** | 70B+ or cloud    | Llama 3.3 70B, Claude, GPT-4o               | Complex reasoning, only when needed         |

### Architecture Configurations

| Configuration  | Components                   | Best For                                           |
| -------------- | ---------------------------- | -------------------------------------------------- |
| **Dual-Model** | Router + Reasoner            | Simpler deployments; reasoner handles conversation |
| **Multi-Net**  | Frontend + Router + Reasoner | Full personalization, fastest response times       |

---

## Intent Marker Syntax

### Standard Format

```
[TOOL: <natural language intent>]
```

### Examples

```
[TOOL: search for email scripts]
[TOOL: run the sync directive]
[TOOL: create a new parser script]
[TOOL: load email_enricher from project]
```

### Result Injection Format

```
[RESULT: <tool_name> | <summary> | <optional details>]
```

### Example

```
[RESULT: search | Found 3 scripts | email_enricher.py, email_validator.py, email_sender.py]
```

---

## Tool Call JSON Schema

### Canonical Format

All tool calls use this exact structure:

```json
{
  "name": "<tool_name>",
  "arguments": {
    "<param1>": "<value1>",
    "<param2>": "<value2>"
  },
  "confidence": 0.95
}
```

### Examples

**Search operation:**

```json
{
  "name": "search",
  "arguments": {
    "item_type": "script",
    "query": "email",
    "source": "local",
    "project_path": "/home/user/project"
  },
  "confidence": 0.95
}
```

**Execute operation:**

```json
{
  "name": "execute",
  "arguments": {
    "action": "run",
    "item_type": "directive",
    "item_id": "sync_directives",
    "project_path": "/home/user/project"
  },
  "confidence": 0.92
}
```

> **Note:** The `confidence` field is optional but recommended for routing decisions.

---

## MCP 2.0 Definition

> **MCP 2.0 is a routing strategy built on top of MCP 1.0 protocol.**

- MCP servers remain **completely unchanged**
- Only the client-side harness and routing layer changes
- This is NOT a replacement for MCP protocol—it's an enhancement to how tool calls are routed

### What Changes (5%)

- WHO decides which tool to call (FunctionGemma router instead of cloud model)
- HOW intents are expressed (natural language markers instead of JSON schemas in prompt)

### What Stays the Same (95%)

- MCP server protocol (stdio/SSE)
- Tool schema format (JSON Schema)
- `call_tool()` API
- All existing MCP servers work unchanged

---

## Performance Benchmarks

### Canonical Latency Figures

All benchmarks assume: **Apple M1 Mac, 8-bit quantized model, 256 token context**

| Component              | Latency (p50) | Latency (p95) |
| ---------------------- | ------------- | ------------- |
| **Router inference**   | 40-80ms       | 100ms         |
| **Semantic search**    | 10-20ms       | 30ms          |
| **MCP tool execution** | 5-50ms        | 100ms         |

### End-to-End Routing

| Path     | Latency | Description                            |
| -------- | ------- | -------------------------------------- |
| **Cold** | 65-80ms | Full search → load → route → execute   |
| **Warm** | 45-60ms | Cached context, search bypassed        |
| **Hot**  | 40-50ms | Predictive loading, one-shot execution |

> **Hardware Variations:** Mobile devices may add 20-50ms. WebGPU adds 50-100ms.

---

## Cost Definitions

### Marginal Cost Model (Recommended)

Cost per request, API charges only:

| Component                | Cost                           |
| ------------------------ | ------------------------------ |
| **Router**               | $0.00 (local inference)        |
| **Reasoner (cloud)**     | $0.08-0.15 per complex request |
| **Reasoner (local 70B)** | $0.00 (electricity negligible) |

### Blended Cost Example

Assuming 90% router-only, 10% reasoner fallback:

- **Per 1,000 requests:** ~$8-15
- **Per 1,000,000 requests:** ~$8,000-15,000

> Traditional cloud-only approach: ~$150,000 per 1M requests

---

## Accuracy and Determinism

### Router Accuracy

FunctionGemma achieves **96-99% accuracy** on well-defined tool routing tasks after fine-tuning.

### Determinism

- Router outputs are **highly deterministic** (constrained output format)
- Scripts are **100% deterministic** (Python code)
- LLM reasoning is **probabilistic** (90% per step typical)

### Fallback Strategy

When router confidence < 70%, defer to reasoning model.

---

## File Naming Conventions

### Document Files

Use descriptive names with spaces, capitalize major words:

- `Training FunctionGemma for Kiwi MCP.md`
- `Deployment Guide - Edge Device Implementation.md`

### URL Encoding for Links

When linking between documents, URL-encode special characters:

- Spaces → `%20`
- Ampersands → `%26`

Example:

```markdown
[Streaming Architecture](./Streaming%20Architecture%20%26%20Concurrent%20Execution.md)
```

---

## Version History

| Version | Date       | Changes                       |
| ------- | ---------- | ----------------------------- |
| 1.0.0   | 2026-01-17 | Initial canonical definitions |

---

_This is the single source of truth. When in doubt, reference this document._
