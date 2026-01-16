# Multi-Net Agent Architecture

## The Vision: Intelligence Distributed Across Specialized Models

A radically different agent architecture where **no single model does everything**. Instead, intelligence is distributed across specialized nets, each doing what it does best:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERACTION                                 │
│                  "Hey, can you clean up my email scripts?"                   │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL FRONTEND (1-4B)                            │
│                                                                              │
│  • Small, fast, personality-focused                                          │
│  • Fine-tuned for YOUR communication style                                   │
│  • Generates [TOOL: natural language intent] markers                         │
│  • Does NOT do complex reasoning                                             │
│  • Does NOT know tool implementations                                        │
│                                                                              │
│  Output: "Sure! Let me find those. [TOOL: search for email scripts]"        │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ "[TOOL: search for email scripts]"
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FUNCTION ROUTER (FunctionGemma 270M)                      │
│                                                                              │
│  • Ultra-fast (40-80ms)                                                      │
│  • Translates intent → Kiwi MCP tool calls                                   │
│  • Predicts: search(item_type="script", query="email")                       │
│  • Routes to correct directive                                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           KIWI MCP EXECUTION                                 │
│                                                                              │
│  Directive: clean_email_scripts                                              │
│  Declares: <model_class tier="high-reasoning" />                             │
│                                                                              │
│  → Routes to high-reasoning model for complex steps                          │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              REASONING ENGINE (70B+ or Cloud Frontier/On-Demand)             │
│                                                                              │
│  • Only invoked when directives require it                                   │
│  • Handles complex multi-step reasoning                                      │
│  • Executes directive steps that need intelligence                           │
│  • Results flow back up the chain                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

## The Key Insight

**The frontend model doesn't need to be smart—it needs to be likeable.**

Traditional architecture:

```
One 70B model does EVERYTHING:
- Conversation ✓
- Reasoning ✓
- Tool calling ✓
- Personality ✓
- Planning ✓

Result: Slow (3-5s), expensive ($0.15/request), overkill for most interactions
```

Multi-Net architecture:

```
Small Frontend (1-4B): Conversation + Personality + Intent markers
Function Router (270M): Intent → Tool translation (40-80ms, highly deterministic)
Directives: Workflow orchestration
Reasoning (70B+ or cloud frontier): Only when truly needed

Result: Fast (50-200ms typical), cheap ($0.001 average), right-sized intelligence
```

---

## Component Deep Dive

### The 4-Model Architecture

Multi-Net uses **four specialized models** working in concert:

| Model                    | Size      | Role                                  | Runs                 |
| ------------------------ | --------- | ------------------------------------- | -------------------- |
| **Frontend**             | 3B-8B     | Conversation + personality            | Always               |
| **Predictive Context**   | 100-300M  | Search + score directives             | Continuously (background) |
| **FunctionGemma Router** | 270M      | Route to primitives (search/load/execute) | On [TOOL:] trigger |
| **Execution Models**     | Variable  | Execute directives with MCP access    | On primitive call    |

**Key insight**: Each model does ONE thing extremely well, then hands off. Execution models are selected dynamically based on directive requirements (model tags in XML).

**Execution Models** can be:
- Claude Sonnet 4 (for complex reasoning)
- Llama 3.3 70B (for local execution)
- Qwen 72B (for code-heavy tasks)
- Specialized fine-tuned models (for domain tasks)
- **Directive XML specifies which model to use**

---

### 1. Conversational Frontend (1-4B Parameters)

The **face** of the system. Small, fast, personalized.

#### What It Does

- **Natural conversation** - Friendly, contextual chat
- **Personality** - Your custom tone, humor, style
- **Intent detection** - Recognizes when tools are needed
- **Marker generation** - Outputs `[TOOL: intent]` markers
- **Result synthesis** - Incorporates tool results naturally

#### What It Does NOT Do

- ❌ Complex reasoning or planning
- ❌ Knowledge of tool schemas or APIs
- ❌ Multi-step problem solving
- ❌ Technical implementation details

#### Model Choices

| Model            | Parameters | Latency  | Notes                    |
| ---------------- | ---------- | -------- | ------------------------ |
| **Phi-3 Mini**   | 3.8B       | 80-150ms | Great reasoning for size |
| **Gemma 2 2B**   | 2B         | 50-100ms | Efficient, multilingual  |
| **Llama 3.2 3B** | 3B         | 70-130ms | Instruction-tuned        |
| **Qwen 2.5 3B**  | 3B         | 70-130ms | Strong chat ability      |
| **TinyLlama**    | 1.1B       | 30-60ms  | Ultra-fast, simpler      |

#### Training Focus

```python
FRONTEND_TRAINING_DATA = [
    # Personality and tone
    {
        "user": "hey whats up",
        "response": "Hey! Just here ready to help. What are we working on today? 🚀"
    },

    # Recognizing tool needs
    {
        "user": "I need to find those email scripts we made",
        "response": "On it! [TOOL: search for email scripts]"
    },

    # Natural marker integration
    {
        "user": "can you sync my stuff and check if theres new directives",
        "response": "Sure! Let me sync first. [TOOL: sync directives]\n\nNow checking for new ones. [TOOL: search for recent directives]"
    },

    # Handling results (injected by harness)
    {
        "context": "[RESULT: search | Found 3 scripts: email_enricher, email_validator, email_sender]",
        "response": "Found 3 email scripts:\n• email_enricher - enriches contact data\n• email_validator - validates addresses\n• email_sender - handles sending\n\nWhich one do you need?"
    },

    # Conversational without tools
    {
        "user": "thanks that was really helpful",
        "response": "Anytime! Let me know if you need anything else. 😊"
    },

    # Declining to reason (important!)
    {
        "user": "can you figure out why this algorithm is O(n²) and optimize it?",
        "response": "That's a meaty problem! Let me run it through our analysis directive. [TOOL: run code_analysis on current file]"
    }
]
```

### 2. Function Router (FunctionGemma 270M)

The **translator** between natural language and tool calls.

Achieves **~98% accuracy** with constrained output on trained tool schemas. Low-confidence predictions (< 0.85) automatically fall back to the reasoning engine for verification.

#### Architecture

```
Frontend Output: "[TOOL: search for email scripts]"
                              │
                              ▼
                 ┌────────────────────────┐
                 │   FunctionGemma 270M   │
                 │                        │
                 │  Input: Natural intent │
                 │  Output: Structured    │
                 │          tool call     │
                 │                        │
                 │  Latency: 40-80ms      │
                 └────────────────────────┘
                              │
                              ▼
        {
            "name": "search",
            "arguments": {
                "item_type": "script",
                "query": "email",
                "project_path": "/home/user/project"
            },
            "confidence": 0.95,
            "predicted_directive": "search_scripts"  // Optional: predict directive too
        }
```

#### Extended Capability: Directive Prediction

The router can predict not just the tool, but which **directive** the user likely wants:

```python
# Router predicts full execution path
{
    "intent": "clean up my email scripts",
    "prediction": {
        "name": "execute",
        "arguments": {
            "action": "run",
            "item_type": "directive",
            "item_id": "refactor_scripts"  # Predicted directive!
        },
        "confidence": 0.88,
        "directive_model_class": "high-reasoning"  # Knows this needs big brain
    }
}
```

### 3. Kiwi MCP Execution Layer

The **orchestrator** that runs directives and manages the workflow.

#### Directive Model Class Routing

Directives declare their computational requirements:

```xml
<directive>
  <metadata>
    <name>refactor_scripts</name>
    <model_class tier="high-reasoning" fallback="max" />
  </metadata>

  <process>
    <step id="1" reasoning="high">
      Analyze code quality and identify refactoring opportunities.
      This requires deep understanding of code patterns.
    </step>

    <step id="2" reasoning="low">
      Apply standard formatting using the formatter script.
    </step>

    <step id="3" reasoning="high">
      Review changes for correctness and suggest improvements.
    </step>
  </process>
</directive>
```

The execution layer routes steps to appropriate models:

```python
class DirectiveExecutor:
    def __init__(self):
        self.frontend = SmallFrontend()      # 3B - fast responses
        self.router = FunctionGemmaRouter()  # 270M - tool routing
        self.reasoner = HighReasoningModel() # 70B+ or cloud frontier - when needed

    async def execute_directive(self, directive: Directive):
        """Execute directive with intelligent model routing"""

        for step in directive.steps:
            if step.reasoning == "high":
                # Route to big brain
                result = await self.reasoner.execute_step(step)
            elif step.reasoning == "medium":
                # Use moderate model
                result = await self.frontend.execute_step(step)
            else:
                # Simple execution (scripts, etc)
                result = await self.execute_script(step)

            yield result
```

### 4. Reasoning Engine (On-Demand High Intelligence)

The **deep thinker** that only activates when truly needed.

#### When It's Invoked

1. **Directive requires it** - `model_class="high-reasoning"`
2. **Step requires it** - `reasoning="high"` on specific steps
3. **Confidence too low** - Router uncertain, needs verification
4. **Complex multi-step** - Planning required

#### What It Does

- Multi-step reasoning and planning
- Code analysis and generation
- Complex decision making
- Error recovery and debugging
- Learning and annealing

#### Cost Optimization

```
Without Multi-Net:
  Every request → 70B model → $0.15 average
  100 requests/day → $15/day → $450/month

With Multi-Net:
  90% handled by Frontend + Router → $0.001
  10% routed to Reasoning → $0.15
  Average: $0.016/request
  100 requests/day → $1.60/day → $48/month

Savings: 89% cost reduction
```

---

## The Harness: Wiring It All Together

### Intent Interception Harness

```python
class MultiNetHarness:
    """
    Orchestrates the multi-net architecture.
    Intercepts markers, routes through layers, handles interrupts.
    """

    def __init__(self):
        # The nets
        self.frontend = ConversationalFrontend("phi-3-mini-chat.gguf")
        self.router = FunctionGemmaRouter("kiwi-router.gguf")
        self.kiwi_mcp = KiwiMCPClient()
        self.reasoner = ReasoningClient()  # Cloud frontier or local 70B+

        # State
        self.conversation_history = []
        self.active_directive = None
        self.interrupt_requested = False

    async def chat(self, user_message: str) -> AsyncIterator[str]:
        """Main chat interface with full multi-net routing"""

        self.conversation_history.append({"role": "user", "content": user_message})

        # Stream from frontend
        buffer = ""

        async for token in self.frontend.stream(
            self.conversation_history,
            system_prompt=self._build_frontend_prompt()
        ):
            buffer += token

            # Check for intent markers
            while True:
                marker_match = re.search(r'\[TOOL:\s*([^\]]+)\]', buffer)

                if not marker_match:
                    # Yield safe portion
                    safe_idx = buffer.rfind('[TOOL:')
                    if safe_idx == -1:
                        yield buffer
                        buffer = ""
                    else:
                        yield buffer[:safe_idx]
                        buffer = buffer[safe_idx:]
                    break

                # Found marker! Extract intent
                intent = marker_match.group(1).strip()

                # Yield text before marker
                yield buffer[:marker_match.start()]

                # Route through FunctionGemma
                yield "\n🔧 "

                tool_call = await self.router.predict(intent)

                # Execute through Kiwi MCP
                result = await self._execute_with_routing(tool_call)

                # Format result for frontend to see
                result_marker = f"[RESULT: {tool_call['name']} | {self._summarize(result)}]"

                # Replace marker with result
                buffer = buffer[:marker_match.start()] + result_marker + buffer[marker_match.end():]

                yield "✓\n"

        # Yield remaining
        if buffer:
            yield buffer

        # Store response
        self.conversation_history.append({
            "role": "assistant",
            "content": buffer
        })

    async def _execute_with_routing(self, tool_call: Dict) -> Any:
        """
        Execute tool call with intelligent routing.
        May invoke reasoning engine if needed.
        """

        tool = tool_call["name"]
        params = tool_call["arguments"]

        if tool == "execute" and params.get("item_type") == "directive":
            # Running a directive - check if it needs reasoning
            directive = await self.kiwi_mcp.load_directive(params["item_id"])

            if directive.model_class == "high-reasoning":
                # Route to reasoning engine
                return await self._execute_with_reasoner(directive)
            else:
                # Execute normally
                return await self.kiwi_mcp.execute(**params)

        else:
            # Simple tool call
            return await self.kiwi_mcp.execute_tool(tool, params)

    async def _execute_with_reasoner(self, directive: Directive) -> Any:
        """Execute directive using reasoning engine for complex steps"""

        results = []

        for step in directive.steps:
            # Check for interrupt
            if self.interrupt_requested:
                yield {"interrupted": True, "completed_steps": len(results)}
                return

            if step.reasoning == "high":
                # Use reasoning engine
                step_result = await self.reasoner.execute(
                    step=step,
                    context=self._build_step_context(directive, results)
                )
            else:
                # Use frontend or direct execution
                step_result = await self._execute_simple_step(step)

            results.append(step_result)

            # Yield progress
            yield {
                "step": step.id,
                "result": step_result,
                "progress": len(results) / len(directive.steps)
            }

        return {"directive": directive.name, "results": results, "success": True}

    def interrupt(self):
        """Request interrupt of current operation"""
        self.interrupt_requested = True

    def _build_frontend_prompt(self) -> str:
        """System prompt for conversational frontend"""

        return """You are a friendly, helpful assistant.

# YOUR PERSONALITY
- Casual, warm, genuine
- Use natural language, occasional emojis
- Be concise but not curt
- Match the user's energy

# TOOL PROTOCOL
When you need to use Kiwi MCP tools, output:
[TOOL: natural language description of what you want]

Examples:
- [TOOL: search for email scripts]
- [TOOL: run the sync directive]
- [TOOL: create a new script called parser]

You'll see results as: [RESULT: tool | summary]
Incorporate these naturally in your response.

# IMPORTANT
You do NOT solve complex problems yourself.
For anything requiring deep thinking, use tools:
- [TOOL: analyze this code]
- [TOOL: run the debugging directive]
- [TOOL: optimize this function]

The system will route to appropriate models.
Your job is conversation and intent expression."""
```

---

## The Interrupt Mechanism

### Why Interrupts?

Long-running directives need graceful stopping:

```
User: "Actually wait, stop that"
      ↓
Agent Harness detects interrupt request
      ↓
Current directive step completes (or aborts)
      ↓
Control returns to frontend
      ↓
Frontend: "Stopped! We completed 3/7 steps. Want to resume or do something else?"
```

### Implementation

```python
class InterruptableHarness(MultiNetHarness):
    """Harness with interrupt support"""

    def __init__(self):
        super().__init__()
        self.interrupt_event = asyncio.Event()
        self.current_task = None

    async def chat(self, user_message: str) -> AsyncIterator[str]:
        """Chat with interrupt detection"""

        # Check if message is interrupt request
        if self._is_interrupt_request(user_message):
            await self._handle_interrupt()
            yield "Stopped! What would you like to do instead?"
            return

        # Normal processing with interruptability
        self.current_task = asyncio.current_task()

        try:
            async for chunk in super().chat(user_message):
                # Check for interrupt between chunks
                if self.interrupt_event.is_set():
                    self.interrupt_event.clear()
                    yield "\n⚠️ Interrupted. "
                    return

                yield chunk

        finally:
            self.current_task = None

    def _is_interrupt_request(self, message: str) -> bool:
        """Detect interrupt requests"""

        interrupt_phrases = [
            "stop", "wait", "hold on", "cancel", "nevermind",
            "abort", "pause", "actually no", "forget it"
        ]

        message_lower = message.lower().strip()
        return any(phrase in message_lower for phrase in interrupt_phrases)

    async def _handle_interrupt(self):
        """Handle interrupt request"""

        self.interrupt_event.set()

        # If there's an active directive, signal it
        if self.active_directive:
            self.active_directive.request_stop()

        # Wait briefly for current operation to notice
        await asyncio.sleep(0.1)


class InterruptableDirective:
    """Directive that can be interrupted mid-execution"""

    def __init__(self, directive: Directive):
        self.directive = directive
        self.stop_requested = False
        self.checkpoint = None

    def request_stop(self):
        """Request graceful stop"""
        self.stop_requested = True

    async def execute_steps(self) -> AsyncIterator[Dict]:
        """Execute steps with interrupt checking"""

        for i, step in enumerate(self.directive.steps):
            # Check for stop request
            if self.stop_requested:
                self.checkpoint = {
                    "directive": self.directive.name,
                    "completed_step": i - 1,
                    "remaining_steps": len(self.directive.steps) - i
                }
                yield {
                    "status": "interrupted",
                    "checkpoint": self.checkpoint
                }
                return

            # Execute step
            result = await self.execute_step(step)

            yield {
                "status": "step_complete",
                "step": i,
                "result": result
            }

        yield {"status": "complete", "success": True}

    async def resume_from_checkpoint(self, checkpoint: Dict):
        """Resume execution from checkpoint"""

        start_step = checkpoint["completed_step"] + 1

        for i in range(start_step, len(self.directive.steps)):
            if self.stop_requested:
                break

            step = self.directive.steps[i]
            result = await self.execute_step(step)
            yield result
```

---

## Training the Multi-Net System

### 1. Training the Conversational Frontend

**Goal**: Natural conversation + marker generation

```python
# frontend_training.py

FRONTEND_TRAINING_CONFIG = {
    "base_model": "microsoft/phi-3-mini-4k-instruct",
    "training_data": "frontend_conversations.jsonl",
    "lora_rank": 32,
    "epochs": 3,
    "learning_rate": 2e-4
}

# Training data categories
training_data = []

# Category 1: Pure conversation (40%)
training_data.extend([
    {"user": "hey!", "response": "Hey there! What's up?"},
    {"user": "thanks for the help", "response": "Anytime! 😊"},
    {"user": "good morning", "response": "Good morning! Ready to get things done today?"},
])

# Category 2: Tool marker generation (40%)
training_data.extend([
    {"user": "find my email scripts", "response": "On it! [TOOL: search for email scripts]"},
    {"user": "run the sync", "response": "Syncing now. [TOOL: run sync_directives]"},
    {"user": "create a new parser script", "response": "Creating it! [TOOL: create script called parser]"},
])

# Category 3: Deferring to reasoning (20%)
training_data.extend([
    {
        "user": "can you optimize this O(n²) function?",
        "response": "That's a meaty optimization problem! Let me analyze it properly. [TOOL: run code_analysis directive]"
    },
    {
        "user": "figure out why this is failing",
        "response": "I'll dig into it. [TOOL: run debugging directive on current file]"
    },
])

# Category 4: Result synthesis
training_data.extend([
    {
        "messages": [
            {"role": "user", "content": "find email scripts"},
            {"role": "assistant", "content": "On it! [TOOL: search for email scripts]"},
            {"role": "system", "content": "[RESULT: search | Found 3: email_enricher, email_validator, email_sender]"}
        ],
        "response": "Found 3 email scripts:\n\n• **email_enricher** - enriches contact data\n• **email_validator** - validates addresses\n• **email_sender** - handles sending\n\nWhich one do you need?"
    }
])
```

### 2. Training FunctionGemma Router

**Goal**: Intent → Tool translation + Directive prediction

```python
# router_training.py

ROUTER_TRAINING_CONFIG = {
    "base_model": "google/functiongemma-270m",
    "training_data": "router_intents.jsonl",
    "lora_rank": 16,
    "epochs": 5
}

# Training data includes directive predictions
training_data = [
    # Simple tool calls
    {
        "intent": "search for email scripts",
        "output": {
            "name": "search",
            "arguments": {"item_type": "script", "query": "email"},
            "predicted_directive": None  # No directive, just search
        }
    },

    # Directive predictions
    {
        "intent": "clean up my code",
        "output": {
            "name": "execute",
            "arguments": {"action": "run", "item_type": "directive", "item_id": "code_cleanup"},
            "predicted_directive": "code_cleanup",
            "model_class_hint": "standard"  # Router knows this is simple
        }
    },

    {
        "intent": "analyze and refactor this function",
        "output": {
            "name": "execute",
            "arguments": {"action": "run", "item_type": "directive", "item_id": "deep_refactor"},
            "predicted_directive": "deep_refactor",
            "model_class_hint": "high-reasoning"  # Router knows this needs big brain
        }
    },

    # Complex multi-step (router predicts orchestration directive)
    {
        "intent": "find email scripts, run the best one, and summarize results",
        "output": {
            "name": "execute",
            "arguments": {"action": "run", "item_type": "directive", "item_id": "search_run_summarize"},
            "predicted_directive": "search_run_summarize",
            "model_class_hint": "high-reasoning"
        }
    }
]
```

### 3. No Training Needed for Reasoning Engine

The reasoning engine (cloud frontier or local 70B+) doesn't need fine-tuning—it just receives:

- Clear directive steps
- Context from previous steps
- Structured output requirements

The directive itself tells the reasoner exactly what to do.

---

## Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    USER                                              │
│                        "analyze my code and fix the bugs"                           │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       CONVERSATIONAL FRONTEND (1-4B)                                 │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Phi-3 Mini (fine-tuned for personality + markers)                           │   │
│  │                                                                             │   │
│  │ Input: "analyze my code and fix the bugs"                                   │   │
│  │ Output: "I'll analyze that for you! [TOOL: run code_analysis_fix]"         │   │
│  │                                                                             │   │
│  │ Latency: 80-150ms | Cost: ~$0.0001                                         │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │ "[TOOL: run code_analysis_fix]"
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          FUNCTION ROUTER (FunctionGemma 270M)                        │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Intent: "run code_analysis_fix"                                             │   │
│  │                                                                             │   │
│  │ Prediction:                                                                 │   │
│  │   tool: "execute"                                                           │   │
│  │   params: {action: "run", item_id: "code_analysis_fix"}                    │   │
│  │   predicted_directive: "code_analysis_fix"                                  │   │
│  │   model_class_hint: "high-reasoning"  ← KNOWS THIS NEEDS BIG BRAIN         │   │
│  │   confidence: 0.94                                                          │   │
│  │                                                                             │   │
│  │ Latency: 40-80ms | Cost: $0                                                │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              KIWI MCP EXECUTION                                      │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Directive: code_analysis_fix                                                │   │
│  │ Model Class: high-reasoning                                                 │   │
│  │                                                                             │   │
│  │ Steps:                                                                      │   │
│  │   1. [high] Analyze code structure and identify issues                      │   │
│  │   2. [low]  Run linter script                                              │   │
│  │   3. [high] Generate fixes for identified bugs                              │   │
│  │   4. [low]  Apply formatting                                                │   │
│  │   5. [high] Review changes for correctness                                  │   │
│  │                                                                             │   │
│  │ Routing: Steps 1, 3, 5 → Reasoning Engine                                  │   │
│  │          Steps 2, 4 → Direct script execution                              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │ (Steps 1, 3, 5 only)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    REASONING ENGINE (70B+ or Cloud Frontier)                         │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Receives: Step context + Code + Previous results                            │   │
│  │                                                                             │   │
│  │ Performs: Deep analysis, bug identification, fix generation                 │   │
│  │                                                                             │   │
│  │ Returns: Structured analysis + Fixes                                        │   │
│  │                                                                             │   │
│  │ Latency: 1-5s per step | Cost: $0.10-0.30 total                            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            RESULT SYNTHESIS                                          │
│                                                                                     │
│  Results flow back up:                                                              │
│    Reasoning → Kiwi MCP → Harness → Frontend → User                               │
│                                                                                     │
│  Frontend synthesizes: "Done! Found and fixed 3 bugs:                              │
│                         • Null pointer on line 42                                  │
│                         • Off-by-one in loop on line 78                           │
│                         • Missing error handling on line 103                       │
│                         Want me to show you the changes?"                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Resource Requirements

### Minimum Viable Setup (Laptop)

```
Frontend (Phi-3 Mini):     ~2GB VRAM
Router (FunctionGemma):    ~500MB VRAM
Reasoning:                 Claude API (no local resources)

Total Local:               ~2.5GB VRAM
Works on:                  Any modern laptop with GPU
```

### Power User Setup (Gaming PC / Mac Studio)

```
Frontend (Phi-3 Mini):     ~2GB VRAM
Router (FunctionGemma):    ~500MB VRAM
Reasoning (70B+ Q4):       ~35GB VRAM

Total Local:               ~38GB VRAM
Works on:                  RTX 4090, Mac Studio M2 Ultra
Benefits:                  100% local, zero API costs
```

### Production Setup (Server)

```
Frontend (multiple):       4x Phi-3 Mini = ~8GB
Router (multiple):         4x FunctionGemma = ~2GB
Reasoning:                 70B+ or cloud frontier API

Handles:                   100+ concurrent users
Latency:                   <200ms typical
```

---

## Comparison: Multi-Net vs. Traditional

| Aspect                | Traditional (One Big Model) | Multi-Net Architecture         |
| --------------------- | --------------------------- | ------------------------------ |
| **Typical Latency**   | 2-5 seconds                 | 100-300ms (90% of requests)    |
| **Cost per Request**  | $0.10-0.20                  | $0.001-0.02 (average)          |
| **Personality**       | Generic                     | Custom fine-tuned              |
| **Tool Calling**      | Cloud API                   | Local, 40-80ms                 |
| **Complex Reasoning** | Always invoked              | Only when needed               |
| **Privacy**           | All data to cloud           | Local-first                    |
| **Offline Mode**      | ❌ No                       | ✅ Partial (frontend + router) |
| **Customization**     | Prompt engineering          | Fine-tune small models         |

---

## Training Pipeline

### Phase 1: Router (Week 1)

```bash
# Generate training data from existing patterns
python scripts/generate_router_data.py \
    --source agent.md \
    --output router_training.jsonl

# Fine-tune FunctionGemma
python scripts/train_router.py \
    --base google/functiongemma-270m \
    --data router_training.jsonl \
    --output kiwi-router
```

### Phase 2: Frontend (Week 2)

```bash
# Generate conversational training data
python scripts/generate_frontend_data.py \
    --personality_config personality.yaml \
    --tool_patterns tool_patterns.jsonl \
    --output frontend_training.jsonl

# Fine-tune Phi-3 Mini
python scripts/train_frontend.py \
    --base microsoft/phi-3-mini-4k-instruct \
    --data frontend_training.jsonl \
    --output kiwi-frontend
```

### Phase 3: Integration (Week 3)

```bash
# Test end-to-end
python scripts/test_multinet.py \
    --frontend kiwi-frontend \
    --router kiwi-router \
    --reasoner claude-sonnet-4-20250514

# Benchmark
python scripts/benchmark_multinet.py \
    --test_suite integration_tests.jsonl \
    --output benchmark_results.json
```

### Phase 4: Deploy (Week 4)

```bash
# Export models
python scripts/export_models.py \
    --frontend kiwi-frontend --format gguf
    --router kiwi-router --format gguf

# Start multi-net server
python -m kiwi.multinet.server \
    --frontend models/kiwi-frontend.gguf \
    --router models/kiwi-router.gguf \
    --reasoner-api claude
```

---

## The Vision: Intelligence at Every Scale

```
User Input
    │
    ▼
┌───────────────────┐
│  Conversational   │  1-4B params
│    Frontend       │  80ms latency
│  Personality +    │  $0.0001 cost
│  Intent Markers   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Function        │  270M params
│    Router         │  40-80ms latency
│  Intent → Tool    │  $0 cost (local)
│  + Directive      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Directive       │  Orchestration
│   Execution       │  Variable
│  Model Routing    │  Based on steps
└─────────┬─────────┘
          │ (only when needed)
          ▼
┌───────────────────┐
│   High           │  70B+ params
│   Reasoning      │  2-5s latency
│  Complex Tasks   │  $0.10-0.30 cost
└───────────────────┘
```

**Right-sized intelligence at every layer.**

The frontend is likeable.
The router is fast.
The directives are smart.
The reasoner is powerful.

**And together, they're unstoppable.**

---

## Next Steps

1. **Build the harness** - Implement `MultiNetHarness` with interrupt support
2. **Train the frontend** - Phi-3 Mini with personality + markers
3. **Train the router** - FunctionGemma with directive prediction
4. **Create test directives** - With model class annotations
5. **Benchmark end-to-end** - Measure latency, cost, quality
6. **Iterate on personality** - Fine-tune frontend for your style

---

_Document generated: 2026-01-17_
_Part of the Kiwi Fine-Tune documentation series_
_Related: MCP 2.0, FunctionGemma Training, Intent Marker Harness_
