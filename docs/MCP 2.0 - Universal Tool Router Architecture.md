# MCP 2.0 - Universal Tool Router Architecture

## The Vision

**MCP 2.0**: A universal tool-calling layer where the reasoning model uses only natural language, and FunctionGemma handles ALL tool translation - not just Kiwi MCP, but weather APIs, databases, web search, everything.

> **Important Clarification:** MCP 2.0 is a **routing strategy** built on top of MCP 1.0 protocol. MCP servers remain completely unchanged. Only the client-side harness and routing layer changes. This is NOT a replacement for MCP protocol—it's an enhancement to how tool calls are routed.

```
┌──────────────────────────────────────────────────────┐
│        REASONING MODEL (Claude/GPT-4o/Gemini)        │
│                                                      │
│  Only knows: "Put what I want in [brackets]"        │
│                                                      │
│  Examples:                                           │
│  - [TOOL: check weather in San Francisco]           │
│  - [TOOL: search for email scripts in project]      │
│  - [TOOL: query database for user records]          │
│  - [TOOL: fetch latest news about AI]               │
│                                                      │
│  NO KNOWLEDGE OF:                                    │
│  ❌ Tool names (is it "get_weather" or "weather"?)  │
│  ❌ Parameters (temp in C or F? zip or city?)       │
│  ❌ APIs (which weather service?)                   │
│  ❌ Schemas (JSON structure?)                       │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ "[TOOL: check weather in San Francisco]"
                   ▼
┌──────────────────────────────────────────────────────┐
│      UNIVERSAL FUNCTION ROUTER (FunctionGemma)       │
│                                                      │
│  Trained on ALL tool domains:                        │
│  - Kiwi MCP tools (search, load, run)               │
│  - Weather APIs (OpenWeather, Weather.gov)          │
│  - Web tools (search, fetch, scrape)                │
│  - Database tools (query, insert, update)           │
│  - File system (read, write, list)                  │
│  - Email (send, read, search)                       │
│  - Calendar (create, update, search)                │
│  - Git (commit, push, status)                       │
│  - ... and more                                      │
│                                                      │
│  Decides:                                            │
│  ✅ Which tool domain (weather vs kiwi vs web)      │
│  ✅ Which specific tool in that domain              │
│  ✅ Exact parameters                                 │
│  ✅ Which provider (OpenWeather vs Weather.gov)     │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ weather_api.get_current(city="San Francisco", units="F")
                   ▼
           ┌───────────────────┐
           │   TOOL EXECUTOR   │
           │   (MCP Runtime)   │
           └───────────────────┘
```

## Architecture: Hierarchical vs. Monolithic

### Option 1: Monolithic FunctionGemma (Simpler) ⭐

**One model trained on ALL tools**

```python
# Training data includes ALL domains
training_data = [
    # Kiwi MCP
    {"intent": "search for email scripts",
     "name": "kiwi.search",
     "arguments": {"item_type": "script", "query": "email"}},

    # Weather
    {"intent": "check weather in SF",
     "name": "weather.get_current",
     "arguments": {"location": "San Francisco", "units": "fahrenheit"}},

    # Web search
    {"intent": "search web for AI news",
     "name": "web.search",
     "arguments": {"query": "AI news", "num_results": 10}},

    # Database
    {"intent": "find user records where age > 25",
     "name": "db.query",
     "arguments": {"table": "users", "filter": "age > 25"}},

    # Calendar
    {"intent": "create meeting tomorrow at 2pm",
     "name": "calendar.create_event",
     "arguments": {"title": "meeting", "start": "tomorrow 2pm"}},
]

# Single FunctionGemma model learns ALL domains
model = train_function_gemma(training_data)
```

**Pros:**

- ✅ Single model to deploy
- ✅ Can handle cross-domain queries
- ✅ Simpler architecture

**Cons:**

- ❌ Larger model (more training data)
- ❌ Harder to update one domain
- ❌ May have domain confusion

### Option 2: Hierarchical Router (Scalable) ⭐⭐

**Meta-router + specialized domain routers**

```
User Intent: "[check weather and search for weather-related scripts]"
        ↓
┌────────────────────────────────────────┐
│     META ROUTER (270M)                 │
│                                        │
│  Classifies intent into domains:      │
│  - "check weather" → weather domain    │
│  - "search scripts" → kiwi domain      │
└────┬─────────────────────────┬─────────┘
     │                         │
     ▼                         ▼
┌─────────────┐        ┌──────────────┐
│  Weather    │        │  Kiwi MCP    │
│  Router     │        │  Router      │
│  (270M)     │        │  (270M)      │
│             │        │              │
│  Specialist │        │  Specialist  │
│  for all    │        │  for Kiwi    │
│  weather    │        │  operations  │
│  APIs       │        │              │
└─────────────┘        └──────────────┘
```

**Benefits:**

- ✅ Easy to add new domains (train new specialist)
- ✅ Easy to update (retrain one specialist)
- ✅ Smaller individual models
- ✅ Specialists are more accurate
- ✅ Can run specialists in parallel

## Implementation: Monolithic Universal Router

```python
# mcp2_monolithic.py

from typing import Dict, List, Any, AsyncIterator
import json

class UniversalFunctionRouter:
    """
    Single FunctionGemma model trained on ALL tool domains.
    Replaces all traditional MCP servers.
    """

    def __init__(self, model_path: str):
        self.model = RouterFactory.create(model_path)

        # Tool registry - maps tool IDs to executors
        self.tool_executors = {}
        self.register_default_tools()

    def register_default_tools(self):
        """Register all available tool executors"""

        # Kiwi MCP
        self.register_tool_domain("kiwi", KiwiMCPExecutor())

        # Weather
        self.register_tool_domain("weather", WeatherAPIExecutor())

        # Web
        self.register_tool_domain("web", WebToolsExecutor())

        # Database
        self.register_tool_domain("database", DatabaseExecutor())

        # File system
        self.register_tool_domain("fs", FileSystemExecutor())

        # Calendar
        self.register_tool_domain("calendar", CalendarExecutor())

        # Email
        self.register_tool_domain("email", EmailExecutor())

        # Git
        self.register_tool_domain("git", GitExecutor())

    def register_tool_domain(self, domain: str, executor: Any):
        """Register a tool domain executor"""
        self.tool_executors[domain] = executor

    async def route_intent(
        self,
        intent: str,
        context: Dict = None
    ) -> Dict:
        """
        Route natural language intent to specific tool call.

        This is the CORE of MCP 2.0 - one function that handles
        ALL tool routing across ALL domains.
        """

        # FunctionGemma predicts tool from intent
        prediction = self.model.predict(
            query=intent,
            context=context or {}
        )

        # Parse tool call
        # Format: "domain.tool_name"
        tool_full_name = prediction.tool_call["name"]
        domain, tool_name = tool_full_name.split(".", 1)
        params = prediction.tool_call["arguments"]

        return {
            "domain": domain,
            "tool": tool_name,
            "params": params,
            "confidence": prediction.confidence
        }

    async def execute_intent(
        self,
        intent: str,
        context: Dict = None
    ) -> Any:
        """
        Route AND execute intent in one call.
        """

        # Route to tool
        route = await self.route_intent(intent, context)

        # Get executor for domain
        executor = self.tool_executors.get(route["domain"])
        if not executor:
            raise ValueError(f"No executor for domain: {route['domain']}")

        # Execute tool
        result = await executor.execute(
            tool=route["tool"],
            params=route["params"]
        )

        return {
            "route": route,
            "result": result
        }


class MCP2Harness:
    """
    MCP 2.0 harness that works with any reasoning model.
    Intercepts [intent markers] and routes through UniversalFunctionRouter.
    """

    def __init__(self, router_model_path: str):
        self.router = UniversalFunctionRouter(router_model_path)

    async def chat(
        self,
        messages: List[Dict],
        model_client: Any,  # Anthropic, OpenAI, etc.
        **model_kwargs
    ) -> AsyncIterator[str]:
        """
        Universal chat interface with MCP 2.0 routing.
        Works with ANY LLM + ANY tools.
        """

        # Add MCP 2.0 system prompt
        system_prompt = self._build_universal_system_prompt()

        # Stream model output
        buffer = ""

        async for chunk in model_client.stream(
            messages=messages,
            system=system_prompt,
            **model_kwargs
        ):
            buffer += chunk

            # Check for intent markers
            while True:
                match = re.search(r'\[([^\]]+)\]', buffer)

                if not match:
                    # Yield safe portion
                    safe_idx = buffer.rfind('[')
                    if safe_idx == -1:
                        yield buffer
                        buffer = ""
                    else:
                        yield buffer[:safe_idx]
                        buffer = buffer[safe_idx:]
                    break

                # Found intent marker
                intent = match.group(1).strip()

                # Yield text before marker
                yield buffer[:match.start()]

                # Execute through universal router
                yield "\n⚡ "

                execution = await self.router.execute_intent(intent)

                # Format result
                result_text = self._format_result(execution)

                # Replace marker with result
                buffer = buffer[:match.start()] + result_text + buffer[match.end():]

                yield "✓\n"

        if buffer:
            yield buffer

    def _build_universal_system_prompt(self) -> str:
        """
        System prompt for MCP 2.0 - teaches ANY tool usage
        through simple bracket syntax.
        """
        return """You have access to many tools through a simple protocol.

# MCP 2.0 PROTOCOL

Put your intent in square brackets: [what you want to do]

Examples across different tool domains:

Weather:
User: "What's the weather in Tokyo?"
You: "Let me check. [check weather in Tokyo] ..."

Web Search:
User: "Find recent AI news"
You: "Searching now. [search web for recent AI news] ..."

Kiwi MCP:
User: "Find email scripts"
You: "I'll search. [search for email scripts in project] ..."

Database:
User: "Show me active users"
You: "Querying. [query database for active users] ..."

Calendar:
User: "Schedule meeting tomorrow 2pm"
You: "Creating event. [create calendar event tomorrow at 2pm] ..."

File System:
User: "What's in my documents folder?"
You: "Let me check. [list files in documents folder] ..."

Git:
User: "Commit my changes"
You: "Committing. [git commit with message 'update'] ..."

Email:
User: "Send email to john@example.com"
You: "Sending. [send email to john@example.com saying hi] ..."

## Multi-Domain Example

User: "Check the weather and schedule a meeting if it's nice"
You: "Let me check the weather first. [check weather in current location]

{Result: sunny, 72°F}

It's sunny and 72°F - perfect! Let me schedule that meeting.
[create calendar event 'outdoor meeting' tomorrow at 10am]

{Result: event created}

Done! Your outdoor meeting is scheduled for tomorrow at 10am."

## Guidelines

- Be natural and conversational
- Use [brackets] for ANY tool operation
- Describe what you want in plain language
- The system handles ALL technical details
- You never need to know tool names, parameters, or APIs

Available tool domains:
- Weather (forecasts, current conditions, alerts)
- Web (search, fetch pages, scrape)
- Kiwi MCP (search, load, run scripts/directives)
- Database (query, insert, update, delete)
- File System (read, write, list, delete)
- Calendar (create, update, search events)
- Email (send, read, search)
- Git (commit, push, status, diff)
- And more...

Just describe your intent naturally. The router handles everything."""

    def _format_result(self, execution: Dict) -> str:
        """Format execution result for model to see"""

        route = execution["route"]
        result = execution["result"]

        # Create concise summary
        summary = self._summarize_result(result)

        return f"[RESULT: {route['domain']}.{route['tool']} | {summary}]"

    def _summarize_result(self, result: Any) -> str:
        """Create brief summary of any result type"""

        if isinstance(result, list):
            return f"{len(result)} items"
        elif isinstance(result, dict):
            if "temperature" in result:
                return f"{result['temperature']}°F, {result.get('condition', 'clear')}"
            elif "status" in result:
                return result["status"]
            elif "count" in result:
                return f"{result['count']} records"
        elif isinstance(result, str):
            return result[:50] + "..." if len(result) > 50 else result

        return "Success"


# Tool Executors

class WeatherAPIExecutor:
    """Executes weather tool calls"""

    async def execute(self, tool: str, params: Dict) -> Dict:
        if tool == "get_current":
            # Call actual weather API
            return await self._get_current_weather(
                params.get("location"),
                params.get("units", "fahrenheit")
            )
        elif tool == "get_forecast":
            return await self._get_forecast(params.get("location"))

    async def _get_current_weather(self, location: str, units: str) -> Dict:
        # Actual API call here
        return {
            "location": location,
            "temperature": 72,
            "condition": "sunny",
            "humidity": 45
        }


class WebToolsExecutor:
    """Executes web tool calls"""

    async def execute(self, tool: str, params: Dict) -> Any:
        if tool == "search":
            return await self._search_web(
                params.get("query"),
                params.get("num_results", 10)
            )
        elif tool == "fetch":
            return await self._fetch_page(params.get("url"))

    async def _search_web(self, query: str, num_results: int) -> List[Dict]:
        # Actual web search here
        return [
            {"title": "Result 1", "url": "...", "snippet": "..."},
            {"title": "Result 2", "url": "...", "snippet": "..."},
        ]


class KiwiMCPExecutor:
    """Executes Kiwi MCP tool calls - your existing implementation"""

    async def execute(self, tool: str, params: Dict) -> Any:
        # Your existing Kiwi MCP logic
        if tool == "search":
            return await kiwi_search(**params)
        elif tool == "load":
            return await kiwi_load(**params)
        # ... etc


# Usage Example
async def demo_mcp2():
    """Demo of MCP 2.0 in action"""

    harness = MCP2Harness(
        router_model_path="universal-function-router.gguf"
    )

    # Works with any model
    client = Anthropic()

    messages = [
        {"role": "user", "content": """
        Check the weather in San Francisco, and if it's nice,
        search for outdoor activity scripts in my project.
        """}
    ]

    async for chunk in harness.chat(
        messages=messages,
        model_client=client,
        model="claude-sonnet-4-20250514"
    ):
        print(chunk, end="", flush=True)

# Output would be:
# "Let me check the weather. ⚡ ✓
#
# It's 68°F and sunny in San Francisco - beautiful!
# Now let me search for outdoor activity scripts. ⚡ ✓
#
# I found 3 scripts related to outdoor activities: ..."
```

## Implementation: Hierarchical Router (More Scalable)

```python
# mcp2_hierarchical.py

class MetaRouter:
    """
    Top-level router that classifies intents into domains,
    then delegates to specialist routers.
    """

    def __init__(self, meta_model_path: str):
        self.meta_model = RouterFactory.create(meta_model_path)
        self.domain_routers = {}

    def register_domain_router(self, domain: str, router_model_path: str):
        """Register a specialist router for a domain"""
        self.domain_routers[domain] = RouterFactory.create(router_model_path)

    async def route_intent(self, intent: str) -> Dict:
        """Two-stage routing: domain classification → specialist routing"""

        # Stage 1: Meta-router classifies domain
        domain_prediction = self.meta_model.predict(
            query=intent,
            task="classify_domain"
        )

        domain = domain_prediction.tool_call["domain"]

        # Stage 2: Specialist router handles details
        specialist = self.domain_routers.get(domain)
        if not specialist:
            raise ValueError(f"No specialist for domain: {domain}")

        tool_call = specialist.predict(
            query=intent,
            context={"domain": domain}
        )

        return {
            "domain": domain,
            "tool": tool_call["name"],
            "params": tool_call["arguments"],
            "meta_confidence": domain_prediction.confidence,
            "specialist_confidence": tool_call.get("confidence", 1.0)
        }


# Training the Meta-Router

meta_training_data = [
    {"intent": "check weather in Tokyo", "domain": "weather"},
    {"intent": "search web for AI news", "domain": "web"},
    {"intent": "find email scripts", "domain": "kiwi"},
    {"intent": "query users table", "domain": "database"},
    {"intent": "create meeting tomorrow", "domain": "calendar"},
    {"intent": "list files in home", "domain": "filesystem"},
    {"intent": "commit my changes", "domain": "git"},
]

# Each domain gets its own specialist training data
kiwi_specialist_data = [
    {"intent": "search for email scripts", "name": "search", "arguments": {"item_type": "script", "query": "email"}},
    {"intent": "run sync directive", "name": "execute", "arguments": {"action": "run", "item_id": "sync_directives"}},
    # ... more Kiwi-specific examples
]

weather_specialist_data = [
    {"intent": "check weather in SF", "name": "get_current", "arguments": {"location": "San Francisco"}},
    {"intent": "forecast for next week", "name": "get_forecast", "arguments": {"days": 7}},
    # ... more weather-specific examples
]
```

## Training Strategy: Mergeable Models?

You asked about merging models. Interesting approaches:

### Approach 1: LoRA Merging (Possible!) ⭐

```python
# Train base FunctionGemma on common patterns
base_model = train_base_router(common_tool_patterns)

# Train domain-specific LoRAs
kiwi_lora = train_lora(base_model, kiwi_specific_data)
weather_lora = train_lora(base_model, weather_specific_data)
web_lora = train_lora(base_model, web_specific_data)

# At runtime, dynamically merge LoRAs based on detected domain
if domain == "kiwi":
    active_model = base_model + kiwi_lora
elif domain == "weather":
    active_model = base_model + weather_lora

# Or merge ALL LoRAs for universal model
universal_model = base_model + kiwi_lora + weather_lora + web_lora
```

**Benefits:**

- ✅ Small LoRA adapters (~10MB each)
- ✅ Easy to add new domains (train new LoRA)
- ✅ Can mix and match at runtime
- ✅ Base model handles common patterns

### Approach 2: Mixture of Experts (Advanced)

```python
class MixtureOfExpertsRouter:
    """
    Multiple specialist routers vote on best tool call.
    """

    def __init__(self, specialist_paths: Dict[str, str]):
        self.specialists = {
            domain: RouterFactory.create(path)
            for domain, path in specialist_paths.items()
        }

    async def route_intent(self, intent: str) -> Dict:
        """Get predictions from all specialists, pick best"""

        # Run all specialists in parallel
        predictions = await asyncio.gather(*[
            specialist.predict(intent)
            for specialist in self.specialists.values()
        ])

        # Pick highest confidence prediction
        best_prediction = max(predictions, key=lambda p: p.confidence)

        return best_prediction.tool_call
```

## Training Data Generation at Scale

```python
# generate_universal_training_data.py

class UniversalTrainingDataGenerator:
    """
    Generate training data for ALL tool domains.
    """

    def __init__(self):
        self.domains = [
            "kiwi", "weather", "web", "database",
            "filesystem", "calendar", "email", "git"
        ]

    def generate_all_domains(self) -> List[Dict]:
        """Generate comprehensive training data"""

        all_data = []

        # Kiwi MCP patterns
        all_data.extend(self._generate_kiwi_patterns())

        # Weather patterns
        all_data.extend(self._generate_weather_patterns())

        # Web patterns
        all_data.extend(self._generate_web_patterns())

        # ... etc for each domain

        # Cross-domain patterns
        all_data.extend(self._generate_cross_domain_patterns())

        return all_data

    def _generate_kiwi_patterns(self) -> List[Dict]:
        """Generate Kiwi MCP training examples"""

        patterns = []

        # Search variations
        for item_type in ["script", "directive", "knowledge"]:
            for query in ["email", "api", "database", "test"]:
                patterns.append({
                    "intent": f"search for {query} {item_type}s",
                    "name": "kiwi.search",
                    "arguments": {
                        "item_type": item_type,
                        "query": query
                    }
                })

        # Load variations
        for item in ["email_enricher", "sync_directives", "api_test"]:
            patterns.append({
                "intent": f"load {item} from project",
                "name": "kiwi.load",
                "arguments": {
                    "item_id": item,
                    "source": "project"
                }
            })

        return patterns

    def _generate_weather_patterns(self) -> List[Dict]:
        """Generate weather API training examples"""

        patterns = []

        cities = ["San Francisco", "Tokyo", "London", "New York"]

        for city in cities:
            # Current weather
            patterns.append({
                "intent": f"check weather in {city}",
                "name": "weather.get_current",
                "arguments": {"location": city, "units": "fahrenheit"}
            })

            # Forecast
            patterns.append({
                "intent": f"weather forecast for {city}",
                "name": "weather.get_forecast",
                "arguments": {"location": city, "days": 7}
            })

        return patterns

    def _generate_cross_domain_patterns(self) -> List[Dict]:
        """
        Generate patterns that might involve multiple domains.
        Important for teaching the router to handle complex intents.
        """

        return [
            {
                "intent": "check weather and schedule outdoor meeting if nice",
                "tools": [
                    {"name": "weather.get_current", "arguments": {"location": "current"}},
                    {"name": "calendar.create_event", "arguments": {"title": "outdoor meeting"}}
                ]
            },
            {
                "intent": "search web for python tutorials then save to notes",
                "tools": [
                    {"name": "web.search", "arguments": {"query": "python tutorials"}},
                    {"name": "filesystem.write", "arguments": {"path": "notes.txt"}}
                ]
            }
        ]
```

## Deployment: Universal MCP 2.0 Server

### Note: Optional HTTP Gateway

The server below is an **optional convenience wrapper** that exposes intent routing via HTTP.
It does NOT replace MCP servers—those continue to run unchanged. This gateway simply
provides an HTTP interface for applications that prefer REST over stdio/SSE.

```python
# mcp2_server.py

class MCP2Server:
    """
    Optional HTTP gateway for MCP 2.0 routing layer.
    Exposes intent routing via REST API while MCP servers run unchanged.
    """

    def __init__(self, router_model_path: str):
        self.router = UniversalFunctionRouter(router_model_path)
        self.app = self._create_server()

    def _create_server(self):
        """Create server with MCP 2.0 protocol"""

        from flask import Flask, request, jsonify, stream_with_context

        app = Flask(__name__)

        @app.route("/v2/execute", methods=["POST"])
        async def execute_intent():
            """Execute natural language intent"""

            data = request.json
            intent = data.get("intent")
            context = data.get("context", {})

            result = await self.router.execute_intent(intent, context)

            return jsonify(result)

        @app.route("/v2/stream", methods=["POST"])
        async def stream_with_interception():
            """Stream LLM output with intent interception"""

            data = request.json
            messages = data.get("messages")
            model_config = data.get("model", {})

            harness = MCP2Harness(self.router)

            async def generate():
                async for chunk in harness.chat(messages, **model_config):
                    yield chunk

            return app.response_class(
                stream_with_context(generate()),
                mimetype="text/event-stream"
            )

        return app

    def run(self, host="localhost", port=3000):
        """Start MCP 2.0 server"""
        self.app.run(host=host, port=port)


# Start server
server = MCP2Server("universal-router.gguf")
server.run()

# Now ANY LLM can use it:
# POST /v2/execute
# {"intent": "check weather in Tokyo"}
# → {domain: "weather", result: {...}}
```

## Benefits of MCP 2.0

> **Note:** MCP 2.0 is a client-side routing enhancement. MCP servers remain unchanged.

| Feature             | MCP 1.0 (Traditional)          | MCP 2.0 (Routing Layer)       |
| ------------------- | ------------------------------ | ----------------------------- |
| **MCP Servers**     | Run as-is                      | Run as-is (unchanged)         |
| **Model Knowledge** | Needs tool schemas             | Just bracket syntax           |
| **Privacy**         | Schemas sent to cloud          | Schemas stay local            |
| **Flexibility**     | Change schema → change prompts | Change tools → retrain router |
| **Speed**           | Cloud tool calling API         | Local router (40-80ms)        |
| **Universality**    | One protocol per provider      | One protocol for ALL          |
| **Debugging**       | Opaque tool calls              | Clear intent → tool flow      |
| **Fine-tuning**     | Per-model                      | Just router                   |

## This IS the Evolution

**MCP 1.0:**

```
"Here are 50 tool schemas in JSON. Generate valid tool calls."
→ Complex, error-prone, leaks implementation details
```

**MCP 2.0:**

```
"Put what you want in [brackets]. We handle everything."
→ Simple, private, universal
```

This is genuinely innovative. You've identified that **the tool-calling layer should be separate from reasoning** and can be handled by a specialized, local, fast model that acts as a universal translator between natural language and ANY tool API.

Want me to create:

1. Complete training pipeline for universal router?
2. Hierarchical routing implementation?
3. LoRA merging strategy?
4. Deployment guide for MCP 2.0 server?
