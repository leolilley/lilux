# Streaming Architecture & Concurrent Execution

## The Challenge

LLMs generate tokens sequentially over time. When using FunctionGemma for tool routing, we face a concurrency problem:

```
The Router generates tokens:
t=0ms:   {"
t=15ms:  {"name
t=30ms:  {"name":
t=45ms:  {"name":"search"
...

Questions:
1. When do we know the tool call is complete?
2. When should we execute?
3. What if the prediction changes mid-stream?
4. How do we coordinate with the reasoning model?
```

This document covers strategies for handling streaming token generation and concurrent model execution.

## Core Problem: Token Generation is Sequential

### Traditional Approach (Slow)

```python
# Wait for complete generation
async def traditional_approach(query: str):
    # 1. Generate all tokens (80ms)
    full_response = await router.generate(query)

    # 2. Parse JSON (5ms)
    tool_call = json.loads(full_response)

    # 3. Execute (50ms)
    result = await execute_tool(tool_call)

    # Total: 135ms
    return result
```

### Streaming Approach (Fast)

```python
# Execute as soon as confident
async def streaming_approach(query: str):
    partial = ""

    # 1. Stream tokens and parse incrementally
    async for token in router.stream(query):
        partial += token

        # 2. Try to parse and calculate confidence
        try:
            tool_call = parse_partial_json(partial)
            confidence = calculate_confidence(token.logprobs)

            # 3. Execute early if confident (at ~45ms)
            if confidence > 0.90 and is_valid(tool_call):
                result = await execute_tool(tool_call)
                return result  # Stop generation!
        except:
            continue  # Keep accumulating

    # Total: 60ms (shaved off 75ms!)
```

## Solution 1: Speculative Execution with Confidence Thresholds

### Concept

Start preparing the tool call **before token generation completes**, then commit when confidence crosses a threshold.

### Implementation

```python
class SpeculativeRouter:
    def __init__(self, confidence_threshold: float = 0.88):
        self.threshold = confidence_threshold

    async def predict_with_speculation(self, query: str) -> ToolCall:
        """Stream tokens and speculatively prepare tool execution"""

        partial_json = ""
        accumulated_logprobs = []
        preparation_task = None

        async for token_data in self.model.stream_with_logprobs(query):
            # Accumulate token
            partial_json += token_data.text
            accumulated_logprobs.append(token_data.logprob)

            # Calculate running confidence
            confidence = self._calculate_confidence(accumulated_logprobs)

            # Try parsing partial JSON
            try:
                tool_call = self._parse_partial(partial_json)

                # Speculation threshold (70%) - start preparing
                if confidence > 0.70 and tool_call.name and not preparation_task:
                    print(f"[{confidence:.0%}] Starting preparation for {tool_call.name}")
                    preparation_task = asyncio.create_task(
                        self._prepare_tool_execution(tool_call)
                    )

                # Commitment threshold (88%) - execute!
                if confidence > self.threshold and tool_call.is_complete():
                    print(f"[{confidence:.0%}] Confident! Executing {tool_call.name}")

                    # Wait for preparation if it was started
                    if preparation_task:
                        await preparation_task

                    return ToolCallResult(
                        tool_call=tool_call,
                        confidence=confidence,
                        tokens_generated=len(accumulated_logprobs),
                        early_exit=True
                    )

            except ParseError:
                continue  # Keep accumulating

        # Fallback: generation completed without early exit
        final_tool_call = json.loads(partial_json)
        return ToolCallResult(
            tool_call=final_tool_call,
            confidence=1.0,
            tokens_generated=len(accumulated_logprobs),
            early_exit=False
        )

    def _calculate_confidence(self, logprobs: List[float]) -> float:
        """Calculate confidence from token log probabilities"""
        # Average probability across tokens
        avg_logprob = sum(logprobs) / len(logprobs)
        confidence = math.exp(avg_logprob)
        return confidence

    def _parse_partial(self, partial_json: str) -> ToolCall:
        """Parse potentially incomplete JSON"""
        # Try closing braces optimistically
        attempts = [
            partial_json,
            partial_json + "}",
            partial_json + "\"}",
            partial_json + "\"}}",
        ]

        for attempt in attempts:
            try:
                parsed = json.loads(attempt)
                return ToolCall.from_dict(parsed)
            except:
                continue

        raise ParseError("Cannot parse partial JSON")

    async def _prepare_tool_execution(self, tool_call: ToolCall):
        """Speculatively prepare tool execution"""
        # Examples of preparation:
        # - Validate parameters
        # - Check file paths exist
        # - Warm up connections
        # - Load required resources
        print(f"Preparing {tool_call.name}...")
        await asyncio.sleep(0.020)  # Simulate prep work
```

### Timeline Example

```
Query: "find email scripts"

t=0ms:    Start streaming
t=15ms:   Token: "search"
          → Confidence: 0.45

t=30ms:   Token: "search|{"
          → Confidence: 0.65

t=40ms:   Token: "search|{\"item_type\""
          → Confidence: 0.72 ✅ SPECULATION THRESHOLD
          → Start preparing search tool in background

t=55ms:   Token: "search|{\"item_type\":\"script\""
          → Confidence: 0.85
          → Preparation ongoing...

t=65ms:   Token: "search|{\"item_type\":\"script\",\"query\":\"email\""
          → Confidence: 0.91 ✅ COMMITMENT THRESHOLD
          → Preparation complete!
          → EXECUTE IMMEDIATELY

t=70ms:   Tool execution started
          (Router stopped generating tokens early)

Savings: 20-30ms by not waiting for full generation
```

## Solution 2: Prefix Caching for Instant Recognition

### Concept

Train the model to output **deterministic prefixes** for common patterns, allowing instant tool detection.

### Training Pattern

```json
// Train model to always output tool name first, before arguments
{
  "user": "find email scripts",
  "output": "search|{\"item_type\":\"script\",\"query\":\"email\"}"
}

{
  "user": "run sync_directives",
  "output": "execute|{\"action\":\"run\",\"item_id\":\"sync_directives\"}"
}
```

### Implementation

```python
class PrefixRouter:
    # Pre-defined tool prefixes
    TOOL_PREFIXES = {
        "search|": "search",
        "execute|": "execute",
        "load|": "load"
    }

    async def predict_with_prefix_detection(self, query: str) -> ToolCall:
        """Detect tool from prefix, then stream arguments"""

        buffer = ""
        detected_tool = None

        async for token in self.model.stream(query):
            buffer += token

            # Check if we've hit a known prefix
            if not detected_tool:
                for prefix, tool_name in self.TOOL_PREFIXES.items():
                    if buffer.startswith(prefix):
                        detected_tool = tool_name
                        print(f"Tool detected: {tool_name} (at {len(buffer)} chars)")

                        # Start preparation immediately
                        prep_task = asyncio.create_task(
                            self._prepare_tool(tool_name)
                        )

                        # Continue streaming only arguments
                        remaining_tokens = self.stream_remaining(token)
                        args_json = await self._collect_arguments(remaining_tokens)

                        # Preparation should be done by now
                        await prep_task

                        return ToolCall(
                            name=tool_name,
                            arguments=json.loads(args_json)
                        )

        raise ValueError("No tool prefix detected")

    async def _collect_arguments(self, token_stream) -> str:
        """Collect remaining argument tokens"""
        args = ""
        async for token in token_stream:
            args += token
        return args
```

### Performance Impact

```
Traditional: Generate all tokens → Parse → Execute
             [████████████████████] 80ms

Prefix-based: Detect prefix → Prepare → Stream args → Execute
              [██] 15ms       [██████████████] 50ms
                             ↑ Parallel!

Net latency: 50ms (37% reduction)
```

## Solution 3: Parallel Reasoning Model Coordination

### The Challenge

We have **two models running concurrently**:

1. **Router** (fast, specialized) - streaming tool predictions
2. **Reasoning** (slow, general) - streaming conversational response

How do they coordinate?

### Architecture: Push Model (Event-Driven) ⭐ **RECOMMENDED**

```python
class DualModelAgent:
    def __init__(self):
        self.router = SpeculativeRouter()
        self.reasoning = AnthropicClient()

    async def process_with_coordination(self, query: str):
        """
        Both models run in parallel.
        Router "pushes" tool calls to reasoning model when ready.
        """

        # Shared communication channel
        tool_channel = asyncio.Queue()
        tool_executed = asyncio.Event()

        # Task 1: Router generates tool calls
        async def router_task():
            try:
                tool_call = await self.router.predict_with_speculation(query)

                # Execute tool
                result = await self.execute_tool(tool_call)

                # Push result to reasoning model
                await tool_channel.put({
                    "status": "complete",
                    "tool": tool_call.name,
                    "result": result,
                    "latency_ms": tool_call.elapsed_ms
                })

                tool_executed.set()

            except Exception as e:
                await tool_channel.put({
                    "status": "error",
                    "error": str(e)
                })

        # Task 2: Reasoning model explains and synthesizes
        async def reasoning_task():
            # Give router a head start (100ms)
            try:
                await asyncio.wait_for(tool_executed.wait(), timeout=0.100)

                # Router won! Get the result
                tool_result = await tool_channel.get()

                # Stream explanation with tool result injected
                async for chunk in self.reasoning.synthesize(
                    query=query,
                    tool_result=tool_result
                ):
                    yield chunk

            except asyncio.TimeoutError:
                # Router too slow or uncertain
                # Reasoning model decides on its own
                async for chunk in self.reasoning.full_process(query):
                    # Still check if router finishes mid-stream
                    try:
                        tool_result = tool_channel.get_nowait()
                        chunk = self.inject_tool_result(chunk, tool_result)
                    except asyncio.QueueEmpty:
                        pass

                    yield chunk

        # Launch both tasks
        router_future = asyncio.create_task(router_task())

        # Stream reasoning output
        async for chunk in reasoning_task():
            yield chunk

        # Ensure router completes
        await router_future
```

### Timeline Example: Router Wins

```
t=0ms:    User query arrives
          ├─ Router starts streaming
          └─ Reasoning model starts processing

t=50ms:   Router: High confidence reached
          → Execute tool
          → Push result to channel
          → Set tool_executed event

t=100ms:  Reasoning model: Checks event
          → Event is SET! Router won!
          → Get tool result from channel
          → Start streaming explanation with result

t=150ms:  User sees: "I found 3 email scripts: ..."
          (Tool already executed!)

Result: Tool execution hidden from user, feels instant
```

### Timeline Example: Reasoning Wins

```
t=0ms:    User query arrives
          ├─ Router starts streaming
          └─ Reasoning model starts processing

t=100ms:  Reasoning timeout expires
          → Event is NOT set (router still unsure)
          → Start streaming without tool result

t=150ms:  Reasoning model: "Let me search for email scripts..."

t=180ms:  Router: Finally reaches high confidence
          → Execute tool
          → Push result to channel

t=200ms:  Reasoning model: Checks channel mid-stream
          → Result available!
          → Inject into current stream
          → "...found 3 scripts: ..."

Result: Graceful fallback, user doesn't notice delay
```

## Solution 4: Beam Search for Robust Predictions

### Concept

Track **multiple possible tool calls** simultaneously, execute when the best candidate crosses threshold.

### Implementation

```python
class BeamRouter:
    def __init__(self, beam_width: int = 3, threshold: float = 0.90):
        self.beam_width = beam_width
        self.threshold = threshold

    async def predict_with_beam_search(self, query: str) -> ToolCall:
        """Track top-k predictions, commit when confident"""

        # Initialize beams
        beams = [
            {
                "tokens": [],
                "text": "",
                "score": 0.0,
                "tool_call": None
            }
            for _ in range(self.beam_width)
        ]

        async for token_probs in self.model.stream_with_beam(query, k=self.beam_width):
            # Update each beam with its best next token
            for i, beam in enumerate(beams):
                best_token = token_probs[i].top_token

                beam["tokens"].append(best_token)
                beam["text"] += best_token.text
                beam["score"] += best_token.logprob

                # Try parsing
                try:
                    beam["tool_call"] = self._parse_partial(beam["text"])
                except:
                    pass

            # Find best beam
            best_beam = max(beams, key=lambda b: b["score"])

            # Calculate confidence for best beam
            confidence = math.exp(best_beam["score"] / len(best_beam["tokens"]))

            # Check if best beam is valid and confident
            if (confidence > self.threshold and
                best_beam["tool_call"] and
                best_beam["tool_call"].is_complete()):

                print(f"Beam search: {confidence:.0%} confident after {len(best_beam['tokens'])} tokens")

                return ToolCallResult(
                    tool_call=best_beam["tool_call"],
                    confidence=confidence,
                    alternative_beams=[b["tool_call"] for b in beams[1:]]
                )

        # All tokens generated, return best
        return ToolCallResult(
            tool_call=best_beam["tool_call"],
            confidence=1.0,
            early_exit=False
        )
```

### Benefits

- **More robust**: Less sensitive to single token errors
- **Confidence estimate**: Based on probability distribution across beams
- **Graceful degradation**: If top beam fails, try alternatives

### Cost

- **2-3x compute**: Running multiple beams in parallel
- **Still faster than full generation**: Early exit saves time

## Solution 5: Hook-Based Event System

### Concept

Emit events at key points during token generation, allowing external systems to react.

### Implementation

```python
class HookedRouter:
    def __init__(self):
        self.hooks = {
            "on_token": [],
            "on_confidence_threshold": [],
            "on_tool_detected": [],
            "on_completion": []
        }

    def register_hook(self, event: str, callback):
        """Register a callback for an event"""
        self.hooks[event].append(callback)

    async def stream_with_hooks(self, query: str) -> ToolCall:
        """Stream tokens and emit events at key points"""

        partial = ""
        confidence = 0.0
        detected_tool = None

        async for token_data in self.model.stream_with_logprobs(query):
            partial += token_data.text
            confidence = self._update_confidence(token_data.logprob)

            # Hook: Every token
            await self._emit("on_token", {
                "token": token_data.text,
                "confidence": confidence,
                "partial": partial
            })

            # Try parsing
            try:
                tool_call = self._parse_partial(partial)

                # Hook: Tool detected (first time)
                if tool_call.name and not detected_tool:
                    detected_tool = tool_call.name
                    await self._emit("on_tool_detected", {
                        "tool": detected_tool,
                        "confidence": confidence
                    })

                # Hook: Confidence threshold crossed
                if confidence > 0.85 and tool_call.is_complete():
                    await self._emit("on_confidence_threshold", {
                        "tool_call": tool_call,
                        "confidence": confidence
                    })

                    # Let hooks decide whether to commit
                    should_commit = await self._check_commit_approval()
                    if should_commit:
                        return tool_call

            except ParseError:
                continue

        # Hook: Generation complete
        final_call = json.loads(partial)
        await self._emit("on_completion", {
            "tool_call": final_call,
            "confidence": 1.0
        })

        return final_call

    async def _emit(self, event: str, data: Dict):
        """Emit event to all registered callbacks"""
        for callback in self.hooks.get(event, []):
            await callback(data)

# Usage example
router = HookedRouter()

@router.on("on_tool_detected")
async def start_preparation(data):
    print(f"Tool detected: {data['tool']}, starting prep...")
    await prepare_tool(data['tool'])

@router.on("on_confidence_threshold")
async def execute_immediately(data):
    print(f"Confident enough! Executing {data['tool_call'].name}")
    result = await execute_tool(data['tool_call'])
    return result

# Run with hooks
tool_call = await router.stream_with_hooks("find email scripts")
```

## Recommended Architecture for Production

### Hybrid Approach: Speculative + Prefix + Hooks

```python
class ProductionRouter:
    """
    Combines best practices:
    - Prefix detection for instant tool recognition
    - Speculative execution for fast commits
    - Hook system for external coordination
    - Fallback to reasoning model for edge cases
    """

    def __init__(
        self,
        speculation_threshold: float = 0.70,
        commitment_threshold: float = 0.88,
        reasoning_timeout_ms: float = 100
    ):
        self.speculation_threshold = speculation_threshold
        self.commitment_threshold = commitment_threshold
        self.reasoning_timeout = reasoning_timeout_ms / 1000

        self.hooks = HookSystem()

    async def route(
        self,
        query: str,
        reasoning_model: Optional[LLM] = None
    ) -> ToolCall:
        """
        Main routing logic with all optimizations
        """

        # Get router's current prediction (may already be warm)
        router_prediction = await self._stream_predict_with_speculation(query)
        
        # Check confidence
        if router_prediction.confidence > 0.85:
            # High confidence - use router prediction
            return router_prediction
        
        # Low confidence - fallback to reasoning model if available
        if reasoning_model:
            reasoning_prediction = await self._reasoning_fallback(
                query, 
                reasoning_model
            )
            return reasoning_prediction
        else:
            # No reasoning fallback, use router anyway
            return router_prediction

    async def _stream_predict_with_speculation(self, query: str) -> ToolCall:
        """Router prediction with all optimizations"""

        partial = ""
        confidence = 0.0
        detected_tool = None
        prep_task = None

        async for token in self.model.stream(query):
            partial += token.text
            confidence = self._calculate_confidence(token.logprobs)

            # Prefix detection
            if not detected_tool:
                detected_tool = self._detect_tool_prefix(partial)
                if detected_tool:
                    await self.hooks.emit("tool_detected", detected_tool)

            # Try parsing
            try:
                tool_call = self._parse_partial(partial)

                # Speculation threshold
                if confidence > self.speculation_threshold and not prep_task:
                    await self.hooks.emit("start_speculation", tool_call)
                    prep_task = asyncio.create_task(
                        self._prepare_execution(tool_call)
                    )

                # Commitment threshold
                if confidence > self.commitment_threshold and tool_call.is_complete():
                    await self.hooks.emit("commit", tool_call)

                    if prep_task:
                        await prep_task

                    return ToolCallResult(
                        tool_call=tool_call,
                        confidence=confidence,
                        method="early_commit"
                    )

            except ParseError:
                continue

        # Fallback: full generation
        return ToolCallResult(
            tool_call=json.loads(partial),
            confidence=1.0,
            method="full_generation"
        )
```

## Performance Summary

| Technique                 | Latency Reduction | Complexity | Best For                |
| ------------------------- | ----------------- | ---------- | ----------------------- |
| **Speculative Execution** | 20-40%            | Medium     | Production              |
| **Prefix Caching**        | 30-50%            | Low        | High-frequency patterns |
| **Beam Search**           | 10-20%            | High       | Accuracy-critical       |
| **Event Hooks**           | Varies            | Medium     | Complex workflows       |
| **Parallel Coordination** | 40-60%            | High       | Multi-model systems     |

## Next Steps

- [Deployment Guide](./Deployment%20Guide%20-%20Edge%20Device%20Implementation.md) - Deploy streaming router to edge devices
- [Integration Patterns](./Integration%20Patterns%20-%20Connecting%20All%20Components.md) - Integrate with Kiwi MCP and reasoning models
- Benchmarking - _Coming Soon_

---

**Key Takeaway**: Don't wait for complete token generation. Use confidence thresholds and speculative execution to reduce latency by 40-60%.
