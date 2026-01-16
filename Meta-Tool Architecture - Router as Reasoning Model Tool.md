# Zero-Shot Intent Marker Harness

> **Note:** This document describes a routing strategy built **on top of** MCP, not a replacement for the MCP protocol itself. When we reference "MCP 2.0" below, we mean this intent-based routing layer that enhances standard MCP usage.

## The Vision

A **harness** that wraps any frontier model (Claude, GPT-4o, Gemini) and teaches it to use intent markers through prompting alone. No fine-tuning required.

```
┌─────────────────────────────────────────────────────┐
│           ANY FRONTIER MODEL (No training)          │
│                                                     │
│   Claude Sonnet 4 / GPT-4o / Gemini 2.0 / ...      │
│                                                     │
│   Learns bracket syntax from system prompt only    │
└────────────┬────────────────────────────────────────┘
             │
             │ Streams: "Let me [search for email scripts] ..."
             ▼
┌─────────────────────────────────────────────────────┐
│              INTENT INTERCEPTOR HARNESS             │
│                                                     │
│  Watches stream → Detects markers → Routes to      │
│  FunctionGemma → Executes → Injects results        │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│         FUNCTIONGEMMA (Local Tool Router)           │
│                                                     │
│  Translates: "search for email scripts"            │
│  Into: search(item_type="script", query="email")   │
└─────────────────────────────────────────────────────┘
```

## The System Prompt Strategy

### Key Insight: Few-Shot Learning in System Prompt

Frontier models are **excellent** at picking up patterns from examples. We teach them the bracket syntax through the system prompt with:

1. **Clear protocol definition**
2. **Multiple examples** showing the pattern
3. **Explicit rules** about when/how to use it
4. **Result interpretation** examples

```python
INTENT_MARKER_SYSTEM_PROMPT = """You are a helpful assistant with access to Kiwi MCP tools through a special protocol.

# TOOL PROTOCOL

When you need to use Kiwi MCP functionality, use this exact format:
[TOOL: natural language description of what you want to do]

## Examples of Correct Usage

User: "find email scripts"
You: "I'll search for email scripts. [TOOL: search for email scripts]"

User: "show me the sync directive"
You: "Let me load that. [TOOL: load sync_directives from project]"

User: "run the backup"
You: "Running the backup now. [TOOL: run backup directive]"

User: "create a new script called parser"
You: "Creating the script. [TOOL: create script called parser]"

## How It Works

1. You output intent markers in natural language
2. The system intercepts them automatically
3. Tools are executed in the background
4. Results appear as: [RESULT: tool_name | summary | details]
5. You can reference the results naturally in your response

## Multi-Step Example

User: "find email scripts and load the enricher"
You: "I'll help with that. First, let me search: [TOOL: search for email scripts]

{System injects: [RESULT: search | Found 3 scripts | email_enricher.py, email_validator.py, email_sender.py]}

Great! I found 3 scripts. Now loading the enricher: [TOOL: load email_enricher from project]

{System injects: [RESULT: load | Loaded email_enricher.py | <content>]}

Here's the email enricher code..."

## Rules

1. Use [TOOL: ...] only for Kiwi MCP operations (search, load, run, create)
2. Describe your intent in natural, clear language inside the brackets
3. Be specific: mention item types (script/directive/knowledge) when relevant
4. Wait for [RESULT: ...] before referencing execution results
5. Stay conversational - markers are tools, not your personality

## Available Operations

- Search: [TOOL: search for X scripts/directives/knowledge]
- Load: [TOOL: load X from project/user/registry]
- Run: [TOOL: run X directive/script]
- Create: [TOOL: create script/directive called X]
- Sync: [TOOL: sync directives/scripts/knowledge]

Current project: {project_path}

Be natural and helpful. Use the protocol when needed, but stay conversational."""
```

## Implementation: Zero-Shot Harness

```python
# zero_shot_harness.py

import re
import asyncio
from typing import AsyncIterator, Dict, List, Optional, Any
from anthropic import Anthropic
from openai import AsyncOpenAI

class IntentMarkerHarness:
    """
    Universal harness that works with any LLM API.
    Teaches models the intent marker protocol through prompting.
    """

    INTENT_PATTERN = re.compile(r'\[TOOL:\s*([^\]]+)\]')
    RESULT_PATTERN = re.compile(r'\[RESULT:\s*([^\]]+)\]')

    def __init__(
        self,
        router_model_path: str,
        kiwi_mcp_client: Any,
        provider: str = "anthropic"  # anthropic, openai, google
    ):
        self.router = RouterFactory.create(router_model_path)
        self.kiwi = kiwi_mcp_client
        self.provider = provider

        # Initialize appropriate client
        if provider == "anthropic":
            self.client = Anthropic()
        elif provider == "openai":
            self.client = AsyncOpenAI()
        # Add more providers as needed

    async def chat(
        self,
        messages: List[Dict[str, str]],
        project_path: str,
        model: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Universal chat interface with intent interception.
        Works with any provider through the same interface.
        """

        # Add system prompt with protocol explanation
        system_prompt = self._build_system_prompt(project_path)

        if self.provider == "anthropic":
            async for chunk in self._chat_anthropic(messages, system_prompt, model):
                yield chunk

        elif self.provider == "openai":
            async for chunk in self._chat_openai(messages, system_prompt, model):
                yield chunk

    async def _chat_anthropic(
        self,
        messages: List[Dict],
        system_prompt: str,
        model: Optional[str]
    ) -> AsyncIterator[str]:
        """Handle Anthropic/Claude streaming with interception"""

        model = model or "claude-sonnet-4-20250514"
        buffer = ""

        async with self.client.messages.stream(
            model=model,
            max_tokens=4096,
            messages=messages,
            system=system_prompt
        ) as stream:

            async for text in stream.text_stream:
                buffer += text

                # Check for complete intent markers
                while True:
                    match = self.INTENT_PATTERN.search(buffer)

                    if not match:
                        # No complete marker - yield safe portion
                        safe_idx = buffer.rfind('[TOOL:')
                        if safe_idx == -1:
                            yield buffer
                            buffer = ""
                        else:
                            yield buffer[:safe_idx]
                            buffer = buffer[safe_idx:]
                        break

                    # Found intent marker!
                    intent = match.group(1).strip()

                    # Yield text before marker
                    yield buffer[:match.start()]

                    # Execute intent
                    yield "\n🔧 "  # Visual indicator

                    result = await self._execute_intent(intent, messages[-1]["content"])

                    # Format result marker
                    result_marker = self._format_result(result)

                    # Replace [TOOL: ...] with [RESULT: ...]
                    buffer = buffer[:match.start()] + result_marker + buffer[match.end():]

                    yield "✓\n"

        # Yield remaining
        if buffer:
            yield buffer

    async def _chat_openai(
        self,
        messages: List[Dict],
        system_prompt: str,
        model: Optional[str]
    ) -> AsyncIterator[str]:
        """Handle OpenAI streaming with interception"""

        model = model or "gpt-4o"

        # Add system message
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages
        ]

        buffer = ""

        stream = await self.client.chat.completions.create(
            model=model,
            messages=full_messages,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                buffer += text

                # Same interception logic as Anthropic
                while True:
                    match = self.INTENT_PATTERN.search(buffer)

                    if not match:
                        safe_idx = buffer.rfind('[TOOL:')
                        if safe_idx == -1:
                            yield buffer
                            buffer = ""
                        else:
                            yield buffer[:safe_idx]
                            buffer = buffer[safe_idx:]
                        break

                    intent = match.group(1).strip()
                    yield buffer[:match.start()]
                    yield "\n🔧 "

                    result = await self._execute_intent(intent, messages[-1]["content"])
                    result_marker = self._format_result(result)

                    buffer = buffer[:match.start()] + result_marker + buffer[match.end():]
                    yield "✓\n"

        if buffer:
            yield buffer

    async def _execute_intent(
        self,
        intent: str,
        original_query: str
    ) -> Dict:
        """
        Execute intent through FunctionGemma router.

        Args:
            intent: Natural language intent from [TOOL: ...] marker
            original_query: Original user query for context
        """

        # FunctionGemma translates intent to tool call
        tool_call = self.router.predict(
            query=intent,
            context={"original_query": original_query}
        )

        # Execute via Kiwi MCP
        result = await self.kiwi.execute(
            tool=tool_call['name'],
            params=tool_call['arguments']
        )

        return {
            "tool": tool_call['name'],
            "intent": intent,
            "result": result,
            "summary": self._summarize(result)
        }

    def _format_result(self, execution_result: Dict) -> str:
        """Format execution result as marker for model to see"""
        return f"[RESULT: {execution_result['tool']} | {execution_result['summary']}]"

    def _summarize(self, result: Any) -> str:
        """Summarize result for the marker"""
        if isinstance(result, list):
            return f"Found {len(result)} items"
        elif isinstance(result, dict):
            if 'status' in result:
                return result['status']
            return "Success"
        return "Completed"

    def _build_system_prompt(self, project_path: str) -> str:
        """Build comprehensive system prompt with examples"""

        return f"""You are a helpful assistant with access to Kiwi MCP tools.

# TOOL CALLING PROTOCOL

Use this exact syntax when you need Kiwi MCP functionality:
[TOOL: natural language description]

The system will automatically:
1. Intercept your [TOOL: ...] markers
2. Execute the appropriate Kiwi MCP operation
3. Inject results as [RESULT: tool | summary]
4. You can then reference results naturally

## Examples

Example 1 - Simple search:
User: "find email scripts"
You: "I'll search for those. [TOOL: search for email scripts]

{Result appears: [RESULT: search | Found 3 items]}

I found 3 email scripts: email_enricher.py, email_validator.py, and email_sender.py."

Example 2 - Load and inspect:
User: "show me the enricher code"
You: "Let me load that. [TOOL: load email_enricher from project]

{Result appears: [RESULT: load | Loaded successfully]}

Here's the code for email_enricher.py..."

Example 3 - Execute action:
User: "sync my directives"
You: "I'll sync them now. [TOOL: run sync_directives]

{Result appears: [RESULT: execute | Sync completed]}

Your directives have been synced successfully!"

Example 4 - Multi-step:
User: "find API directives and run the first one"
You: "I'll search first. [TOOL: search for API directives]

{Result appears: [RESULT: search | Found 2 items]}

Found 2 API directives. Running the first one: [TOOL: run api_test directive]

{Result appears: [RESULT: execute | Completed]}

The API test directive has been executed successfully!"

## Guidelines

✅ DO:
- Use natural, clear language in markers
- Mention item types when relevant (script/directive/knowledge)
- Be specific about operations (search/load/run/create)
- Wait for [RESULT: ...] before discussing execution outcomes
- Stay conversational around the markers

❌ DON'T:
- Try to format tool parameters yourself
- Use markers for non-Kiwi operations
- Reference execution results before they appear
- Over-explain the protocol to users

## Available Operations

Search: [TOOL: search for <query> scripts/directives/knowledge]
Load: [TOOL: load <item_name> from project/user/registry]
Run: [TOOL: run <directive/script name>]
Create: [TOOL: create script/directive called <name>]
Sync: [TOOL: sync directives/scripts/knowledge]

Current project: {project_path}

Remember: The markers are a tool, not your personality. Be helpful and natural."""


# Usage with any provider
harness = IntentMarkerHarness(
    router_model_path="kiwi-router.gguf",
    kiwi_mcp_client=kiwi_client,
    provider="anthropic"  # or "openai", "google"
)

# Works with Claude
async for chunk in harness.chat(
    messages=[{"role": "user", "content": "find email scripts"}],
    project_path="/home/user/project",
    model="claude-sonnet-4-20250514"
):
    print(chunk, end="", flush=True)

# Works with GPT-4o (same interface!)
harness_openai = IntentMarkerHarness(
    router_model_path="kiwi-router.gguf",
    kiwi_mcp_client=kiwi_client,
    provider="openai"
)

async for chunk in harness_openai.chat(
    messages=[{"role": "user", "content": "find email scripts"}],
    project_path="/home/user/project",
    model="gpt-4o"
):
    print(chunk, end="", flush=True)
```

## Advanced: In-Context Learning Reinforcement

For models that struggle with the protocol initially, use **progressive examples**:

```python
class AdaptiveHarness(IntentMarkerHarness):
    """
    Harness that adapts system prompt based on model performance.
    Adds more examples if model doesn't follow protocol.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol_failures = 0
        self.successful_uses = 0

    async def chat(
        self,
        messages: List[Dict],
        project_path: str,
        model: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Chat with adaptive prompting"""

        # Start with base system prompt
        system_prompt = self._build_adaptive_prompt(project_path)

        # ... streaming logic ...

        # Track if model used protocol correctly
        buffer = ""
        used_protocol = False

        async for chunk in self._stream_with_tracking(messages, system_prompt):
            buffer += chunk

            if '[TOOL:' in chunk:
                used_protocol = True
                self.successful_uses += 1

            yield chunk

        # If model didn't use protocol when it should have
        if self._should_have_used_protocol(messages[-1]["content"]) and not used_protocol:
            self.protocol_failures += 1

    def _build_adaptive_prompt(self, project_path: str) -> str:
        """Build system prompt with more examples if needed"""

        base_prompt = self._build_system_prompt(project_path)

        # If model struggling, add more examples
        if self.protocol_failures > 2:
            base_prompt += """

# ADDITIONAL EXAMPLES (for clarity)

User: "what scripts do we have about databases?"
You: "Let me check. [TOOL: search for database scripts]

{Result appears: [RESULT: search | Found 5 items]}

We have 5 database-related scripts..."

User: "can you run the migration script?"
You: "Sure, running it now. [TOOL: run migration script]

{Result appears: [RESULT: execute | Completed]}

The migration script has completed."

User: "I need to create a new validator"
You: "I'll create that. [TOOL: create script called validator]

{Result appears: [RESULT: execute | Created]}

Created validator script successfully!"

Remember: Use [TOOL: ...] for any Kiwi MCP operation."""

        return base_prompt

    def _should_have_used_protocol(self, message: str) -> bool:
        """Detect if message was asking for Kiwi MCP operation"""
        kiwi_keywords = [
            "find", "search", "show", "load", "run", "execute",
            "create", "sync", "script", "directive", "knowledge"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in kiwi_keywords)
```

## Chain-of-Thought Prompting Enhancement

Add reasoning steps to help model understand when to use protocol:

```python
COT_SYSTEM_PROMPT = """...

# DECISION PROCESS

Before responding, think through:

1. Does the user want a Kiwi MCP operation?
   → YES: Use [TOOL: ...] marker
   → NO: Respond normally

2. What operation? (search/load/run/create)
   → Be specific in your marker

3. What details matter? (item type, name, location)
   → Include in natural language

Example internal reasoning:
User: "show me email scripts"
→ Wants: Kiwi operation (YES)
→ Operation: Search
→ Details: email, scripts
→ Marker: [TOOL: search for email scripts]

Example 2:
User: "what's the weather?"
→ Wants: Kiwi operation (NO)
→ Respond normally (no marker)

You don't need to show this reasoning - just use it to decide."""
```

## Fallback: Protocol Correction

If model doesn't use protocol, gently correct:

```python
class SelfCorrectingHarness(IntentMarkerHarness):
    """
    Harness that detects when model should have used protocol
    and inserts a corrective message.
    """

    async def chat(
        self,
        messages: List[Dict],
        project_path: str,
        model: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Chat with protocol correction"""

        buffer = ""
        used_protocol = False

        # Stream response
        async for chunk in super().chat(messages, project_path, model):
            buffer += chunk
            if '[TOOL:' in chunk:
                used_protocol = True
            yield chunk

        # Check if model should have used protocol
        user_message = messages[-1]["content"]

        if self._is_kiwi_request(user_message) and not used_protocol:
            # Model forgot to use protocol!
            # Add corrective message to conversation

            correction = "\n\n(Note: For Kiwi MCP operations, use [TOOL: ...] markers. Let me try again.)\n\n"
            yield correction

            # Re-prompt with correction
            corrected_messages = messages + [
                {"role": "assistant", "content": buffer},
                {"role": "user", "content": "Please use the [TOOL: ...] protocol for that Kiwi operation."}
            ]

            async for chunk in super().chat(corrected_messages, project_path, model):
                yield chunk
```

## Why This Works Without Fine-Tuning

### 1. **Pattern Recognition**

Frontier models excel at few-shot learning. 3-5 examples in system prompt is often enough.

### 2. **Natural Syntax**

Brackets with natural language is intuitive. Models don't need to learn complex schemas.

### 3. **Immediate Feedback**

When markers work, model sees `[RESULT: ...]` injection. Reinforces correct usage.

### 4. **Context Window**

Modern models have huge context (200K+ tokens). Rich system prompts don't hurt.

### 5. **Instruction Following**

Frontier models are trained to follow system instructions. Clear protocol definition is sufficient.

## Testing Different Models

```python
# Test harness across providers
async def test_all_providers():
    query = "find email scripts and run sync"

    providers = [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),  # Test cheaper model
    ]

    for provider, model in providers:
        print(f"\n{'='*60}")
        print(f"Testing: {provider} / {model}")
        print('='*60)

        harness = IntentMarkerHarness(
            router_model_path="kiwi-router.gguf",
            kiwi_mcp_client=kiwi_client,
            provider=provider
        )

        async for chunk in harness.chat(
            messages=[{"role": "user", "content": query}],
            project_path="/home/user/project",
            model=model
        ):
            print(chunk, end="", flush=True)

        print("\n")

# Compare protocol adherence
asyncio.run(test_all_providers())
```

## Intent-Based Routing: The Evolution

This routing layer represents an evolution in how models interact with MCP tools (we informally call this pattern "MCP 2.0" but it builds on standard MCP, not replacing it):

### MCP 1.0 (Current)

```
Model → Sees raw tool schemas → Generates precise tool calls → Executes
```

- Model must understand JSON schemas
- Model sees all implementation details
- Model does formatting work

### Intent-Based Routing (This Design)

```
Model → Outputs natural language intents → Router translates → Executes
```

- Model uses natural language only
- Router handles all technical details
- Model focuses on conversation

**Benefits:**

- ✅ Works with any model (no tool calling API needed)
- ✅ Privacy (schemas stay local)
- ✅ Simpler prompts (no schema clutter)
- ✅ Faster (FunctionGemma router adds only 40-80ms latency)
- ✅ Flexible (easy to change tools without retraining)
