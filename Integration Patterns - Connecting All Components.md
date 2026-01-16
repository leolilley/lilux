# Integration Patterns - Connecting All Components

## Overview

This document covers practical patterns for integrating:

1. **FunctionGemma Router** (local, fast tool calling)
2. **High-Reasoning Model** (Claude/GPT-4o/Gemini, conversation)
3. **Kiwi MCP** (tool execution)

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  CLI / Desktop App / Mobile App / Web Interface              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT COORDINATOR                          │
│  Orchestrates router + reasoning model + tool execution      │
└────────┬────────────────────────┬───────────────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐      ┌──────────────────────┐
│  FunctionGemma   │      │  Reasoning Model     │
│  Router (Local)  │      │  (Cloud API)         │
│                  │      │                      │
│  • 40-80ms       │      │  • Claude Sonnet 4   │
│  • Deterministic │      │  • GPT-4o            │
│  • Offline       │      │  • Gemini 2.0        │
└────────┬─────────┘      └──────┬───────────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    KIWI MCP TOOLS     │
         │  • search             │
         │  • load               │
         │  • execute            │
         └───────────────────────┘
```

## Pattern 1: Sequential Consultation (Simple)

### When to Use

- Development/testing
- Low-frequency requests
- Predictable workflows

### Flow

```
User Query
    ↓
Router predicts tool (40-80ms)
    ↓
Execute tool (100ms)
    ↓
Reasoning model synthesizes response (800ms)
    ↓
Stream to user

Total: ~980ms
```

### Implementation

```python
# patterns/sequential.py

from typing import AsyncIterator, Dict
from kiwi_router import RouterFactory
from anthropic import Anthropic
import asyncio

class SequentialAgent:
    def __init__(
        self,
        router_model_path: str,
        kiwi_mcp_client: KiwiMCPClient
    ):
        self.router = RouterFactory.create(router_model_path)
        self.reasoning = Anthropic()
        self.kiwi = kiwi_mcp_client

    async def process(self, query: str, project_path: str) -> AsyncIterator[str]:
        """Process query sequentially: route → execute → explain"""

        # Step 1: Router predicts tool call
        yield "🔍 Analyzing query...\n"

        tool_call = self.router.predict(query, project_path)

        yield f"🎯 Will use: {tool_call['name']}({tool_call['arguments']})\n"

        # Step 2: Execute tool via Kiwi MCP
        yield "⚙️ Executing...\n"

        tool_result = await self.kiwi.execute(
            name=tool_call['name'],
            arguments=tool_call['arguments']
        )

        yield f"✅ Executed successfully\n\n"

        # Step 3: Reasoning model explains result
        yield "💭 Generating response...\n\n"

        async for chunk in self._synthesize_with_reasoning(
            query=query,
            tool_call=tool_call,
            tool_result=tool_result
        ):
            yield chunk

    async def _synthesize_with_reasoning(
        self,
        query: str,
        tool_call: Dict,
        tool_result: Dict
    ) -> AsyncIterator[str]:
        """Use reasoning model to explain results"""

        context = f"""The user asked: "{query}"

I used the Kiwi MCP tool: {tool_call['name']}
With parameters: {tool_call['arguments']}

Result:
{tool_result}

Please explain this result to the user in a natural, conversational way."""

        async with self.reasoning.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": context}]
        ) as stream:
            async for text in stream.text_stream:
                yield text

# Usage
agent = SequentialAgent(
    router_model_path="kiwi-router.gguf",
    kiwi_mcp_client=kiwi_client
)

async for chunk in agent.process(
    "find email enrichment scripts",
    "/home/user/project"
):
    print(chunk, end="", flush=True)
```

### Pros & Cons

✅ **Pros:**

- Simple to implement
- Easy to debug
- Predictable behavior
- Clear error handling

❌ **Cons:**

- Sequential = slower
- User waits for each step
- Doesn't leverage router speed advantage

## Pattern 2: Trigger-Based Execution (Fast) ⭐ **RECOMMENDED**

### When to Use

- Production applications
- User-facing interfaces
- Latency-sensitive workflows

### Flow

```
Continuous Background:
    FunctionGemma maintains warm context based on conversation

User Query → Frontend Model
    ↓
Frontend outputs: [TOOL: intent]
    ↓
TRIGGER: Execute FunctionGemma's current prediction (40-80ms)
    ↓
Tool executes via MCP (100ms)
    ↓
Result injected back into frontend stream
```

FunctionGemma's prediction is ready before the frontend even outputs the `[TOOL:]` marker!

### Implementation

```python
# patterns/trigger_based_execution.py

import asyncio
from typing import AsyncIterator, Dict, Optional

class TriggerBasedAgent:
    def __init__(
        self,
        router_model_path: str,
        kiwi_mcp_client: KiwiMCPClient
    ):
        self.router = RouterFactory.create(router_model_path)
        self.frontend = FrontendModel()  # 3B conversational model
        self.kiwi = kiwi_mcp_client

        # Predictive context management
        self.current_prediction = None
        self.context_state = "cold"  # cold, warm, hot
        
        # Start continuous prediction
        asyncio.create_task(self._maintain_predictions())

    async def _maintain_predictions(self):
        """Continuously update predictions based on conversation"""
        
        while True:
            if self.conversation_history:
                # Analyze conversation signals
                signals = self.extract_signals(self.conversation_history[-5:])
                
                # Update context and prediction
                if signals["confidence"] > 0.7:
                    self.current_prediction = await self.router.predict_with_context(
                        signals=signals,
                        context_state=self.context_state
                    )
                    self.context_state = "warm"
            
            await asyncio.sleep(0.1)  # Update every 100ms

    async def process(
        self,
        query: str,
        project_path: str
    ) -> AsyncIterator[str]:
        """Process with trigger-based execution"""

        # Stream frontend model output
        async for chunk in self.frontend.stream(query):
            
            # Check for [TOOL: ...] marker
            if "[TOOL:" in chunk:
                intent = self.extract_intent(chunk)
                
                # TRIGGER: Execute current prediction
                if self.current_prediction and self.current_prediction.confidence > 0.85:
                    # Prediction is ready! Execute immediately
                    tool_result = await self._execute_prediction()
                else:
                    # Fallback: Search + route
                    tool_result = await self._fallback_route(intent)
                
                # Inject result into stream
                yield f"[RESULT: {tool_result.summary}]"
            else:
                yield chunk

    async def _execute_prediction(self):
        """Execute FunctionGemma's current prediction"""

        try:
            # Execute with current prediction
            tool_result = await self.kiwi.execute(
                name=self.current_prediction.tool_call['name'],
                arguments=self.current_prediction.tool_call['arguments']
            )

            return {
                "status": "success",
                "tool_call": tool_call,
                "result": tool_result,
                "latency_ms": tool_call.get('latency_ms', 0)
            })

            self.router_completed.set()

        except Exception as e:
            # Router failed, reasoning will handle
            await self.tool_result_queue.put({
                "status": "error",
                "error": str(e)
            })
            self.router_completed.set()

    async def _reasoning_path(
        self,
        query: str,
        router_timeout: float
    ) -> AsyncIterator[str]:
        """Slow path: Reasoning model with optional tool injection"""

        # Wait briefly for router (give it a head start)
        try:
            await asyncio.wait_for(
                self.router_completed.wait(),
                timeout=router_timeout
            )
            router_won = True
        except asyncio.TimeoutError:
            router_won = False

        if router_won:
            # Router finished! Get result and synthesize
            tool_data = await self.tool_result_queue.get()

            if tool_data["status"] == "success":
                yield f"✅ Found: {self._summarize_result(tool_data['result'])}\n\n"

                # Stream explanation
                async for chunk in self._synthesize(query, tool_data):
                    yield chunk
            else:
                # Router failed, reasoning decides
                async for chunk in self._full_reasoning_process(query):
                    yield chunk
        else:
            # Router too slow/uncertain, reasoning decides
            yield "💭 Let me think about that...\n\n"

            # Start streaming reasoning
            reasoning_stream = self._full_reasoning_process(query)

            # But keep checking for router result
            async for chunk in self._stream_with_late_injection(
                reasoning_stream,
                self.tool_result_queue
            ):
                yield chunk

    async def _stream_with_late_injection(
        self,
        reasoning_stream: AsyncIterator[str],
        result_queue: asyncio.Queue
    ) -> AsyncIterator[str]:
        """Stream reasoning, but inject tool result if it arrives"""

        async for chunk in reasoning_stream:
            # Check if router finished mid-stream
            try:
                tool_data = result_queue.get_nowait()

                # Router finished! Inject result
                yield "\n\n"
                yield f"✅ Tool executed: {self._summarize_result(tool_data['result'])}\n\n"

                # Continue streaming with result context
                # (This is simplified - real impl would modify context)

            except asyncio.QueueEmpty:
                pass

            yield chunk

    def _summarize_result(self, result: Dict) -> str:
        """Create brief summary of tool result"""
        # Simplified - customize based on your result format
        if isinstance(result, list):
            return f"{len(result)} items"
        return str(result)[:100]

    async def _synthesize(
        self,
        query: str,
        tool_data: Dict
    ) -> AsyncIterator[str]:
        """Synthesize explanation with tool result"""

        context = f"""User asked: "{query}"

Tool executed: {tool_data['tool_call']['name']}
Result: {tool_data['result']}

Explain this result naturally."""

        async with self.reasoning.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": context}]
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _full_reasoning_process(self, query: str) -> AsyncIterator[str]:
        """Full reasoning when router isn't used"""

        # Give reasoning model the Kiwi MCP tools as native tools
        async with self.reasoning.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": query}],
            tools=self._get_kiwi_tools()
        ) as stream:
            async for text in stream.text_stream:
                yield text

# Usage
agent = TriggerBasedAgent(
    router_model_path="kiwi-router.gguf",
    kiwi_mcp_client=kiwi_client
)

# FunctionGemma maintains predictions continuously
# Frontend's [TOOL:] markers trigger execution
async for chunk in agent.process(
    "find scripts about email enrichment",
    "/home/user/project"
):
    print(chunk, end="", flush=True)
```

### Timeline Example: Router Wins

```
t=0ms:    Query arrives
          ├─ Router starts
          └─ Reasoning starts

t=40-80ms: Router: Tool call ready
          → Execute tool

t=150ms:  Router: Tool executed ✅
          → Push to queue
          → Set completion event

t=150ms:  Reasoning: Check event
          → Event is SET!
          → Get result from queue
          → Start synthesizing with result

t=200ms:  User sees: "✅ Found 3 scripts"

t=800ms:  Reasoning synthesis complete
          User sees full explanation

Result: Tool already executed, user doesn't wait!
```

### Pros & Cons

✅ **Pros:**

- **Fast**: Tool executes while reasoning thinks
- **Graceful**: Fallback if router fails
- **Efficient**: Leverages both models optimally

❌ **Cons:**

- More complex to implement
- Requires careful coordination
- Harder to debug

## Pattern 3: Speculative Execution (Aggressive)

### When to Use

- Very high-frequency requests
- Known patterns
- Acceptable false-positive risk

### Concept

Start executing tool **before router finishes** if confidence crosses threshold early.

### Implementation

```python
# patterns/speculative.py

class SpeculativeAgent:
    def __init__(
        self,
        router_model_path: str,
        kiwi_mcp_client: KiwiMCPClient,
        speculation_threshold: float = 0.70,
        commit_threshold: float = 0.88
    ):
        self.router = RouterFactory.create(router_model_path)
        self.kiwi = kiwi_mcp_client
        self.speculation_threshold = speculation_threshold
        self.commit_threshold = commit_threshold

    async def process(
        self,
        query: str,
        project_path: str
    ) -> AsyncIterator[str]:
        """Process with speculative execution"""

        partial_json = ""
        confidence = 0.0
        preparation_task = None

        # Stream router predictions
        async for token_data in self.router.stream_predict(query, project_path):
            partial_json += token_data.text
            confidence = token_data.confidence

            yield f"[{confidence:.0%}] {token_data.text}"

            # Try parsing
            try:
                tool_call = self._parse_partial(partial_json)

                # Speculation threshold - start preparing
                if confidence > self.speculation_threshold and not preparation_task:
                    yield f"\n🔮 Speculating: {tool_call['name']}...\n"

                    preparation_task = asyncio.create_task(
                        self._prepare_tool(tool_call)
                    )

                # Commit threshold - execute!
                if confidence > self.commit_threshold and tool_call.is_complete:
                    yield f"\n✅ Confident! Executing...\n"

                    # Wait for preparation
                    if preparation_task:
                        await preparation_task

                    # Execute
                    result = await self.kiwi.execute(
                        name=tool_call['name'],
                        arguments=tool_call['arguments']
                    )

                    yield f"\n📊 Result: {result}\n"
                    return

            except ParseError:
                continue

    async def _prepare_tool(self, tool_call: Dict):
        """Speculatively prepare tool execution"""
        # Examples:
        # - Validate parameters
        # - Check paths exist
        # - Warm up connections
        # - Pre-load resources

        if tool_call['name'] == 'search':
            # Pre-validate search parameters
            await self.kiwi.validate_search_params(tool_call['arguments'])

        elif tool_call['name'] == 'load':
            # Check if item exists
            await self.kiwi.check_item_exists(tool_call['arguments'])

# Usage
agent = SpeculativeAgent(
    router_model_path="kiwi-router.gguf",
    kiwi_mcp_client=kiwi_client,
    speculation_threshold=0.70,  # Start preparing at 70%
    commit_threshold=0.88        # Execute at 88%
)

async for chunk in agent.process(
    "search for email scripts",
    "/home/user/project"
):
    print(chunk, end="", flush=True)
```

### Pros & Cons

✅ **Pros:**

- **Fastest**: Preparation starts early
- **Transparent**: User sees confidence levels
- **Educational**: Shows AI reasoning

❌ **Cons:**

- May waste work on false positives
- More complex error handling
- Verbose output (good for debug, not production)

## Pattern 4: Confidence-Based Routing

### When to Use

- Mixed workloads (simple + complex)
- Cost optimization
- Quality vs. speed tradeoff

### Flow

```
Query arrives
    ↓
Router predicts with confidence
    ↓
    ├─ High confidence (>0.90) → Execute immediately
    ├─ Medium confidence (0.70-0.90) → Ask reasoning to verify
    └─ Low confidence (<0.70) → Reasoning decides from scratch
```

### Implementation

```python
# patterns/confidence_routing.py

class ConfidenceBasedAgent:
    def __init__(
        self,
        router_model_path: str,
        kiwi_mcp_client: KiwiMCPClient
    ):
        self.router = RouterFactory.create(router_model_path)
        self.reasoning = Anthropic()
        self.kiwi = kiwi_mcp_client

        # Confidence thresholds
        self.HIGH_CONFIDENCE = 0.90
        self.MEDIUM_CONFIDENCE = 0.70

    async def process(
        self,
        query: str,
        project_path: str
    ) -> AsyncIterator[str]:
        """Route based on router confidence"""

        # Get router prediction with confidence
        prediction = self.router.predict_with_confidence(query, project_path)

        yield f"🎯 Confidence: {prediction.confidence:.0%}\n"

        if prediction.confidence >= self.HIGH_CONFIDENCE:
            # HIGH CONFIDENCE: Execute immediately
            yield "✅ High confidence - executing directly\n\n"
            async for chunk in self._execute_directly(prediction):
                yield chunk

        elif prediction.confidence >= self.MEDIUM_CONFIDENCE:
            # MEDIUM CONFIDENCE: Ask reasoning to verify
            yield "🤔 Medium confidence - verifying...\n\n"
            async for chunk in self._verify_with_reasoning(query, prediction):
                yield chunk

        else:
            # LOW CONFIDENCE: Reasoning decides
            yield "🧠 Low confidence - using full reasoning\n\n"
            async for chunk in self._full_reasoning(query):
                yield chunk

    async def _execute_directly(self, prediction) -> AsyncIterator[str]:
        """Execute router's suggestion directly"""

        result = await self.kiwi.execute(
            name=prediction.tool_call['name'],
            arguments=prediction.tool_call['arguments']
        )

        yield f"Result: {result}\n"

    async def _verify_with_reasoning(
        self,
        query: str,
        prediction
    ) -> AsyncIterator[str]:
        """Ask reasoning model to verify router's suggestion"""

        verification_prompt = f"""The user asked: "{query}"

Our router suggests this tool call:
Tool: {prediction.tool_call['name']}
Arguments: {prediction.tool_call['arguments']}
Confidence: {prediction.confidence:.0%}

Is this correct? Reply with just "yes" or "no" and explanation."""

        response = await self.reasoning.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": verification_prompt}]
        )

        verification = response.content[0].text.lower()

        if verification.startswith("yes"):
            yield "✅ Verified - executing\n\n"
            result = await self.kiwi.execute(
                name=prediction.tool_call['name'],
                arguments=prediction.tool_call['arguments']
            )
            yield f"Result: {result}\n"
        else:
            yield f"❌ Rejected - {verification}\n\n"
            async for chunk in self._full_reasoning(query):
                yield chunk

    async def _full_reasoning(self, query: str) -> AsyncIterator[str]:
        """Full reasoning model process"""

        async with self.reasoning.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": query}],
            tools=self._get_kiwi_tools()
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

### Cost Analysis

```
Scenario: 1000 requests/day

High confidence (70%): 700 × $0.00 = $0
Medium confidence (20%): 200 × $0.02 = $4  (verification call)
Low confidence (10%): 100 × $0.15 = $15   (full reasoning)

Total: $19/day vs $150/day (all reasoning)

Savings: 87%
```

## Pattern 5: Unified Router with Orchestrator Fallback

### When to Use

- Complex workflows requiring both fast routing and deep reasoning
- Need for instant predictions with orchestrator backup
- All tool domains (Kiwi MCP, file ops, git, etc.)

### Architecture

```
                    Query
                      ↓
          ┌───────────────────────────┐
          │  Continuous Router        │
          │  FunctionGemma 270M       │
          │                           │
          │  Knows 3 primitives:      │
          │  - search()               │
          │  - load()                 │
          │  - execute(directive)     │
          │                           │
          │  Context cache:           │
          │  - Predicted directives   │
          │    (email_validator,      │
          │     git_workflow, etc.)   │
          └───────────┬───────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    High Confidence         Low Confidence
    Execute directive       → Orchestrator
         ↓
    Backend LLM runs directive
    (directive knows MCP tools)
```

### Implementation

```python
# patterns/unified_router.py

class UnifiedRouterSystem:
    def __init__(self):
        # Single universal router for ALL tools
        self.router = FunctionGemmaRouter("universal-router.gguf")
        self.reasoning = Anthropic()
        self.mcp = KiwiMCPClient()

    async def process(self, query: str) -> AsyncIterator[str]:
        """Process query with single router + orchestrator fallback"""

        # Get prediction from universal router
        prediction = await self.router.predict(
            query,
            cached_context=self.router.context_cache.get("directives")
        )

        if prediction.confidence > 0.85:
            # High confidence - execute immediately
            yield f"🎯 Router: {prediction.tool_name} ({prediction.confidence:.0%})\n\n"
            
            result = await self._execute_tool(prediction)
            yield f"✅ Result: {result}\n"
            
        elif prediction.confidence > 0.60:
            # Medium confidence - orchestrator confirms
            yield f"🤔 Router suggests: {prediction.tool_name} ({prediction.confidence:.0%})\n"
            yield f"📞 Consulting orchestrator...\n\n"
            
            confirmation = await self.reasoning.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": f"""Confirm tool call:
                    
Query: {query}
Router suggests: {prediction.tool_name}({prediction.arguments})
Confidence: {prediction.confidence:.0%}

Reply with 'APPROVED' or 'REJECTED' and explanation."""
                }]
            )
            
            if "APPROVED" in confirmation.content[0].text:
                result = await self._execute_tool(prediction)
                yield f"✅ Approved and executed: {result}\n"
            else:
                yield f"❌ Rejected, falling back to full reasoning\n"
                async for chunk in self._full_reasoning(query):
                    yield chunk
                    
        else:
            # Low confidence - full orchestrator reasoning
            yield f"🧠 Low confidence, using orchestrator\n\n"
            async for chunk in self._full_reasoning(query):
                yield chunk

    async def _execute_tool(self, prediction):
        """Execute the predicted tool call"""
        
        if prediction.tool_name in ["search", "load", "execute"]:
            # Kiwi MCP primitive
            return await self.mcp.call_tool(
                prediction.tool_name,
                prediction.arguments
            )
        else:
            # Other MCP tool (git, file, etc.)
            return await self.mcp.call_tool(
                prediction.tool_name,
                prediction.arguments
            )
```

## Best Practices

### 1. Always Have Fallbacks

```python
try:
    # Try fast router path
    result = await router.predict(query)
except Exception as e:
    # Fallback to reasoning
    logging.warning(f"Router failed: {e}, falling back to reasoning")
    result = await reasoning_model.decide(query)
```

### 2. Monitor Performance

```python
import time

class MonitoredAgent:
    def __init__(self):
        self.metrics = {
            "router_wins": 0,
            "reasoning_wins": 0,
            "router_latency": [],
            "total_latency": []
        }

    async def process(self, query: str):
        start = time.time()

        # ... processing ...

        latency = (time.time() - start) * 1000
        self.metrics["total_latency"].append(latency)

        # Log metrics periodically
        if len(self.metrics["total_latency"]) % 100 == 0:
            self._log_metrics()
```

### 3. Cache Common Patterns

```python
from functools import lru_cache

class CachedAgent:
    @lru_cache(maxsize=1000)
    def predict_cached(self, query: str, project_path: str):
        """Cache predictions for identical queries"""
        return self.router.predict(query, project_path)
```

### 4. Implement Circuit Breakers

```python
class ResilientAgent:
    def __init__(self):
        self.router_failures = 0
        self.max_failures = 5
        self.use_router = True

    async def process(self, query: str):
        if self.use_router and self.router_failures < self.max_failures:
            try:
                return await self.router_path(query)
            except Exception:
                self.router_failures += 1
                if self.router_failures >= self.max_failures:
                    logging.error("Router circuit breaker opened")
                    self.use_router = False

        # Fallback to reasoning
        return await self.reasoning_path(query)
```

## Recommended Pattern by Use Case

| Use Case        | Pattern                 | Why                             |
| --------------- | ----------------------- | ------------------------------- |
| **CLI tool**    | Sequential              | Simple, debuggable              |
| **Desktop app** | Trigger-Based Execution | Fast, predictive context        |
| **Mobile app**  | Confidence Routing      | Optimize battery/latency        |
| **Web app**     | Trigger-Based Execution | Best UX, feels instant          |
| **Enterprise**  | Multi-Agent             | Handle complexity               |
| **Embedded**    | Speculative             | Maximum local efficiency        |

## Next Steps

- **Monitoring & Optimization** - Coming Soon
- **Example Applications** - Coming Soon
- **Troubleshooting Guide** - Coming Soon

---

**Recommendation**: Start with **Sequential** for development, move to **Trigger-Based Execution** for production.
