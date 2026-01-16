# Predictive Context Model: Continuous Directive Discovery

## Overview

The **Predictive Context Model** is a lightweight, continuously-running model that analyzes conversation in real-time to predict which directives are likely to be needed. It pre-loads these directives into FunctionGemma's context window BEFORE the frontend model triggers a tool intent, enabling near-instantaneous routing.

## The Missing Piece

### The Full Architecture

```
┌─────────────────────────────────────────────────────────┐
│              FRONTEND CONVERSATIONAL MODEL               │
│                     (3B-8B params)                       │
│                                                          │
│  Handles: Conversation, personality                      │
│  Outputs: [TOOL: natural language intent]               │
└────────────────────────┬─────────────────────────────────┘
                         │
                         │ Triggers
                         ▼
         ┌───────────────────────────────┐
         │   FUNCTIONGEMMA ROUTER        │
         │        (270M)                 │
         │                               │
         │  Predicts primitive:          │
         │  - search(...)                │
         │  - load(...)                  │
         │  - execute(directive)         │
         │                               │
         │  Context provided by ─────────┼─────┐
         │  Predictive Model             │     │
         └───────────────┬───────────────┘     │
                         │                     │
                         │ Hands off to        │
                         ▼                     │
         ┌───────────────────────────────┐     │
         │   EXECUTION LAYER             │     │
         │   (Model Varies by Directive) │     │
         │                               │     │
         │  Receives primitive +         │     │
         │  conversation context         │     │
         │                               │     │
         │  Directive specifies model:   │     │
         │  • Claude Sonnet (reasoning)  │     │
         │  • Llama 3.3 (local)          │     │
         │  • Qwen (code tasks)          │     │
         │  • Specialized fine-tunes     │     │
         │                               │     │
         │  Cold: search→load→execute    │     │
         │  Warm: load→execute           │     │
         │  Hot: execute directly        │     │
         └───────────────────────────────┘     │
                                               │
                                               │
┌──────────────────────────────────────────────┘
│
│  RUNS IN PARALLEL (ALWAYS ACTIVE)
│
▼
┌─────────────────────────────────────────────────────────┐
│       PREDICTIVE CONTEXT MODEL (NEW!)                    │
│              (Small: 100-300M)                           │
│                                                          │
│  Continuously:                                           │
│  1. Analyzes conversation signals                        │
│  2. GENERATES search queries for directives              │
│  3. Executes vector searches across all directives       │
│  4. Scores results (0.0-1.0)                             │
│  5. Loads into ONE unified context:                      │
│     • Score >0.85: Full details (params, examples)       │
│     • Score 0.6-0.85: Moderate details (schema)          │
│     • Score <0.6: Minimal details (name + desc)          │
│                                                          │
│  ONE context window, detail level varies by score!       │
└──────────────────────────────────────────────────────────┘
```

## Key Concept: ONE Context, Varying Detail

The critical insight: FunctionGemma sees **ONE unified context window** containing all predicted directives. The prediction score from vector search determines **how much detail** is included for each directive:

```
Unified Context Window:
┌────────────────────────────────────────────────┐
│  Directive: email_validator (score: 0.91)     │
│  Description: Validates email addresses        │
│  Inputs: email_list (array), strict_mode (bool)│
│  Params: {email_list: [...], strict_mode: true}│ ← Full details
│  Examples: [...]                               │
├────────────────────────────────────────────────┤
│  Directive: email_enrichment (score: 0.73)    │
│  Description: Enriches email data              │
│  Inputs: emails (array), api_key (string)     │ ← Schema only
├────────────────────────────────────────────────┤
│  Directive: csv_parser (score: 0.58)          │
│  Description: Parses CSV files                 │ ← Minimal
└────────────────────────────────────────────────┘

High score → More detail → FunctionGemma can predict execute()
Medium score → Moderate detail → FunctionGemma predicts load()
Low score → Minimal detail → FunctionGemma predicts search()
```

**NOT three separate contexts.** ONE context with graduated detail.

## Why We Need This Model

### The Latency Problem

Without predictive context:
```
Frontend: [TOOL: validate these emails]
    ↓
FunctionGemma: "I need directives about email validation"
    ↓
Semantic search: 15ms (searching 10,000+ directives)
    ↓
Load top 7 directives: 5ms
    ↓
FunctionGemma routes: 45ms
    ↓
Total: 65ms
```

With predictive context:
```
Background: Conversation mentions "email" and "validation"
    ↓
Predictive model: Generates searches for "email validation" directives
    ↓
Vector search returns: email_validator (score: 0.91), email_enrichment (score: 0.72)
    ↓
Loads into ONE context with varying detail:
  - email_validator: Full details + params (score >0.85)
  - email_enrichment: Schema only (score 0.6-0.85)
    ↓
When [TOOL:] arrives, context already populated!
    ↓
FunctionGemma routes with full context: 45ms
    ↓
Total: 45ms (30% faster!)
```

## Model Architecture

### Core Requirements

1. **Small & Fast**: 100-300M parameters (runs on CPU)
2. **Continuous**: Processes every conversation turn
3. **Semantic Understanding**: Knows what directives do, not just names
4. **Low Latency**: Must predict in <50ms to stay ahead of user
5. **Context Aware**: Understands conversation flow

### Model Selection Options

| Model                  | Params | Latency | Best For                        |
| ---------------------- | ------ | ------- | ------------------------------- |
| **MiniLM-L6**          | 22M    | 5-10ms  | Keyword extraction (CPU)        |
| **DistilBERT**         | 66M    | 10-20ms | Semantic similarity (CPU)       |
| **TinyBERT**           | 14M    | 3-8ms   | Ultra-fast classification       |
| **E5-small**           | 33M    | 8-15ms  | Embedding generation            |
| **Fine-tuned Phi-1.5** | 130M   | 20-40ms | Directive prediction (GPU edge) |

**Recommended**: Fine-tuned **E5-small** or **DistilBERT** for semantic understanding with low latency.

## How It Works

### Step 1: Continuous Conversation Analysis

```python
class PredictiveContextModel:
    """
    Continuously analyzes conversation to predict directive needs.
    Runs in background, updating FunctionGemma's context.
    """
    
    def __init__(self):
        # Lightweight semantic model (E5-small or DistilBERT)
        self.semantic_model = load_model("e5-small-v2")  # 33M params
        
        # Directive database with embeddings
        self.directive_db = DirectiveDatabase()
        self.directive_embeddings = self.directive_db.load_embeddings()
        
        # Vector search (fast semantic similarity)
        self.vector_index = FAISSIndex(self.directive_embeddings)
        
        # Conversation history buffer
        self.conversation_buffer = deque(maxlen=10)  # Last 10 turns
        
        # Current predictions
        self.predicted_directives = []
        self.confidence = 0.0
    
    async def analyze_conversation_turn(self, user_message: str, assistant_message: str):
        """
        Called after every conversation turn.
        Outputs searches and scores directives for dynamic context loading.
        """
        
        # Add to conversation buffer
        self.conversation_buffer.append({
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": time.time()
        })
        
        # Extract conversation signals
        signals = self.extract_signals(self.conversation_buffer)
        
        # Output searches for directives
        if signals["confidence"] > 0.5:
            searches = await self.generate_directive_searches(signals)
            scored_directives = await self.score_and_load_directives(searches)
            
            # Update FunctionGemma's context with scored directives
            await self.update_gemma_context(scored_directives)
    
    def extract_signals(self, conversation_buffer: deque) -> Dict:
        """
        Extract predictive signals from conversation.
        
        Signals:
        - Keywords: "email", "validation", "deploy"
        - Actions: "create", "check", "run", "test"
        - Domains: "git", "database", "api"
        - Patterns: "and then", "next", "after that"
        - Entities: File names, URLs, commands
        """
        
        # Combine recent turns into context window
        recent_text = " ".join([
            f"{turn['user']} {turn['assistant']}"
            for turn in list(conversation_buffer)[-5:]  # Last 5 turns
        ])
        
        # Extract keywords (nouns, verbs)
        keywords = self.extract_keywords(recent_text)
        
        # Extract action verbs
        actions = self.extract_actions(recent_text)
        
        # Classify domain (git, email, database, etc.)
        domain = self.classify_domain(recent_text)
        
        # Build semantic query
        semantic_query = self.build_query(keywords, actions, domain)
        
        # Calculate confidence
        confidence = min(
            len(keywords) * 0.15 +
            len(actions) * 0.25 +
            (0.4 if domain else 0.0),
            1.0
        )
        
        return {
            "query": semantic_query,
            "keywords": keywords,
            "actions": actions,
            "domain": domain,
            "confidence": confidence
        }
    
    async def generate_directive_searches(self, signals: Dict) -> List[str]:
        """
        Generate search queries for directive discovery.
        Outputs multiple searches to cover conversation context.
        """
        
        searches = []
        
        # Primary search from extracted signals
        searches.append(signals["query"])
        
        # Domain-specific searches
        if signals["domain"]:
            searches.append(f"{signals['domain']} workflow")
        
        # Action-based searches
        for action in signals["actions"][:3]:  # Top 3 actions
            searches.append(f"{action} {signals.get('domain', '')}")
        
        return searches
    
    async def score_and_load_directives(self, searches: List[str]) -> List[Dict]:
        """
        Execute searches, score results, build unified context.
        
        Scoring determines detail level in ONE context window:
        - High score (>0.85): Full details (inputs, params, examples)
        - Medium score (0.6-0.85): Moderate details (inputs, description)
        - Low score (<0.6): Minimal details (name, description only)
        
        Returns list of directives with appropriate detail level.
        """
        
        all_results = []
        
        # Execute all searches via vector search
        for search_query in searches:
            query_embedding = self.semantic_model.encode(search_query)
            
            results = self.vector_index.search(
                query_embedding,
                top_k=10,
                threshold=0.5  # Lower threshold to capture more
            )
            
            all_results.extend(results)
        
        # Deduplicate and score
        scored_directives = self.deduplicate_and_score(all_results)
        
        # Build unified context with varying detail levels
        context_directives = []
        
        for directive_id, score in scored_directives:
            directive = self.directive_db.get_directive(directive_id)
            
            # Base info (always included)
            directive_context = {
                "name": directive.name,
                "description": directive.description,
                "score": score
            }
            
            # Add more details based on score
            if score > 0.85:
                # HIGH SCORE: Full details for hot execution
                directive_context.update({
                    "inputs": directive.inputs,
                    "params": self.extract_params_from_context(directive),
                    "examples": directive.examples[:2] if directive.examples else []
                })
            elif score > 0.6:
                # MEDIUM SCORE: Schema for warm loading
                directive_context.update({
                    "inputs": directive.inputs
                })
            # LOW SCORE: Just name + description (already set)
            
            context_directives.append(directive_context)
        
        return context_directives
    
    def deduplicate_and_score(self, results: List) -> List[Tuple[str, float]]:
        """
        Deduplicate search results and aggregate scores.
        Higher scores when directive appears in multiple searches.
        """
        
        directive_scores = {}
        
        for result in results:
            directive_id = result.id
            score = result.score
            
            if directive_id in directive_scores:
                # Boost score for multiple appearances
                directive_scores[directive_id] = max(
                    directive_scores[directive_id],
                    score * 1.1  # 10% boost
                )
            else:
                directive_scores[directive_id] = score
        
        # Sort by score descending
        return sorted(
            directive_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]  # Top 15 directives
    
    async def update_gemma_context(self, scored_directives: List[Dict]):
        """
        Push scored directives into FunctionGemma's ONE unified context.
        Score determines detail level per directive.
        
        This is what FunctionGemma sees when it routes.
        """
        
        # Store scored directives
        self.predicted_directives = scored_directives
        
        # Format for FunctionGemma's unified context cache
        context_payload = {
            "directives": scored_directives,  # All directives with varying detail
            "timestamp": time.time()
        }
        
        # Update FunctionGemma's context cache
        await self.gemma_router.update_context_cache(context_payload)
        
        # Count by score range for logging
        high_score = [d for d in scored_directives if d["score"] > 0.85]
        medium_score = [d for d in scored_directives if 0.6 < d["score"] <= 0.85]
        low_score = [d for d in scored_directives if d["score"] <= 0.6]
        
        print(f"🔥 Context updated: {len(scored_directives)} directives")
        print(f"   High detail (>0.85): {len(high_score)} - {[d['name'] for d in high_score]}")
        print(f"   Medium detail (0.6-0.85): {len(medium_score)}")
        print(f"   Low detail (<0.6): {len(low_score)}")
```

### Step 2: FunctionGemma Routes with Unified Context

```python
class FunctionGemmaRouter:
    """
    FunctionGemma router with unified predictive context.
    Sees ONE context window with varying directive detail levels.
    Predicts primitive, then hands off to Execution Layer.
    """
    
    def __init__(self):
        self.model = load_gemma_model("function-gemma-270m")
        
        # Unified context cache (ONE window, varying detail per directive)
        self.context_cache = {
            "directives": [],  # All directives with varying detail
            "timestamp": 0.0
        }
        
        # Base primitives (always in context)
        self.primitives = ["search", "load", "execute"]
    
    async def update_context_cache(self, context_payload: Dict):
        """
        Called by PredictiveContextModel to update unified context.
        Each directive has detail level based on prediction score.
        """
        self.context_cache = context_payload
    
    async def predict(self, tool_intent: str) -> Dict:
        """
        Predict which primitive to call using unified context.
        High-detail directives → can predict execute()
        Medium-detail directives → predict load()
        Low-detail/no match → predict search()
        
        Result is handed off to Handler Model, NOT executed here.
        """
        
        # Check context validity
        cache_age = time.time() - self.context_cache.get("timestamp", 0)
        context_valid = cache_age < 30.0
        
        # Build unified context for FunctionGemma
        if context_valid and len(self.context_cache["directives"]) > 0:
            context = self.build_unified_context(tool_intent)
        else:
            # No cache - empty context
            context = self.build_base_context(tool_intent)
        
        # Run FunctionGemma prediction
        tool_call = await self.model.predict(context)
        
        # Return prediction for Handler Model
        return {
            "primitive": tool_call["name"],        # search/load/execute
            "arguments": tool_call["arguments"],
            "confidence": tool_call["confidence"],
            "latency_ms": 40-50                    # FunctionGemma inference
        }
    
    def build_unified_context(self, tool_intent: str) -> str:
        """
        Build ONE unified context with all directives.
        Detail level varies by prediction score per directive.
        
        High score (>0.85) → Full details including params
        Medium score (0.6-0.85) → Schema with inputs
        Low score (<0.6) → Name + description only
        """
        
        directives_text = []
        
        for d in self.context_cache["directives"]:
            # Base info (always included)
            directive_text = f"- {d['name']}: {d['description']}"
            
            # Add details based on score
            if d.get("score", 0) > 0.85:
                # High score: Full details
                if "inputs" in d:
                    directive_text += f"\n  Inputs: {d['inputs']}"
                if "params" in d:
                    directive_text += f"\n  Ready params: {d['params']}"
                if "examples" in d:
                    directive_text += f"\n  Examples: {d['examples']}"
            elif d.get("score", 0) > 0.6:
                # Medium score: Schema only
                if "inputs" in d:
                    directive_text += f"\n  Inputs: {d['inputs']}"
            # Low score: Just name + description (already set)
            
            directives_text.append(directive_text)
        
        directives_context = "\n\n".join(directives_text)
        
        return f"""You are a tool router. Predict which primitive to call.

Primitives:
- search(item_type, query): Search for directives
- load(item_type, item_id): Load a directive
- execute(directive, params): Execute a directive

Available directives (detail level varies by relevance):
{directives_context}

Tool intent: {tool_intent}

Prediction:"""
    
    def build_base_context(self, tool_intent: str) -> str:
        """
        Build context with no cached directives.
        FunctionGemma will likely predict search().
        """
        
        return f"""You are a tool router. Predict which primitive to call.

Primitives:
- search(item_type, query): Search for directives
- load(item_type, item_id): Load a directive
- execute(directive, params): Execute a directive

Tool intent: {tool_intent}

Prediction:"""
```

### Step 3: Execution Layer Runs the Primitive

```python
class ExecutionLayer:
    """
    Receives primitive from FunctionGemma and executes with conversation context.
    
    The key insight: FunctionGemma predicts WHAT to do (primitive),
    Execution Layer does the actual work with the appropriate model.
    
    Model selection is DYNAMIC - each directive specifies which model to use:
    - Claude Sonnet 4: Complex reasoning tasks
    - Llama 3.3 70B: Local execution
    - Qwen 72B: Code-heavy tasks
    - Fine-tuned models: Domain-specific tasks
    
    Flow determined by what FunctionGemma predicted:
    - search: Full search→load→execute (cold)
    - load: Load→fill params→execute (warm)
    - execute: Execute directly (hot)
    """
    
    def __init__(self):
        # Available models for directive execution
        self.models = {
            "claude-sonnet-4": AnthropicClient(),
            "llama-3.3-70b": LocalLlamaClient(),
            "qwen-72b": QwenClient(),
            # ... other models
        }
        self.mcp_client = KiwiMCPClient()
    
    def select_model(self, directive):
        """
        Select appropriate model based on directive's model tag.
        Directives specify which model they need in their XML.
        """
        model_tag = directive.get("model", "claude-sonnet-4")  # Default
        return self.models.get(model_tag)
    
    async def handle_primitive(
        self,
        primitive: str,
        arguments: Dict,
        conversation_context: List[Dict]
    ) -> Any:
        """
        Execute primitive based on what FunctionGemma predicted.
        
        The primitive itself tells us the flow:
        - search(): Full search→load→execute (cold - no match in context)
        - load(): Load→fill params→execute (warm - schema in context)
        - execute(): Execute directly (hot - full details in context)
        """
        
        if primitive == "search":
            # COLD: User switched topics or no relevant directives predicted
            # Handler does full search→load→execute flow
            return await self.handle_search(arguments, conversation_context)
        
        elif primitive == "load":
            # WARM: Directive schema was in context but params not filled
            # Handler loads directive, fills params, executes
            return await self.handle_load(arguments, conversation_context)
        
        elif primitive == "execute":
            # HOT: Full directive + params were in context
            # Handler just executes
            return await self.handle_execute(arguments, conversation_context)
    
    async def handle_search(self, arguments: Dict, conversation: List[Dict]) -> Any:
        """
        Handle search() primitive: search → load → execute
        
        Happens when: User switched to unrelated topic, or predictive model
        didn't find relevant directives (low scores for all).
        """
        
        # Step 1: Search for directives
        search_result = await self.mcp_client.call_tool(
            "search",
            arguments=arguments
        )
        
        # Step 2: Pick best directive (use conversation context)
        directive = await self.mcp_client.call_tool(
            "search",
            arguments=arguments
        )
        
        # Step 3: Load directive
        directive = await self.mcp_client.call_tool(
            "load",
            arguments={"item_type": "directive", "item_id": directive["results"][0]["id"]}
        )
        
        # Step 4: Select appropriate model based on directive's model tag
        model = self.select_model(directive)
        
        # Step 5: Fill params from conversation using selected model
        params = await model.extract_params(
            directive=directive,
            conversation=conversation
        )
        
        # Step 6: Execute (directive specifies which model ran it)
        result = await self.mcp_client.call_tool(
            "execute",
            arguments={
                "item_type": "directive",
                "item_id": directive["id"],
                "action": "run",
                "params": params,
                "model": directive.get("model")  # Directive's model preference
            }
        )
        
        return result
    
    async def handle_load(self, arguments: Dict, conversation: List[Dict]) -> Any:
        """
        Handle load() primitive: load → execute
        
        Happens when: Directive schema was in context (medium score 0.6-0.85)
        but params weren't filled yet.
        """
        
        directive_id = arguments.get("item_id")
        
        # Step 1: Load full directive
        directive = await self.mcp_client.call_tool(
            "load",
            arguments={"item_type": "directive", "item_id": directive_id}
        )
        
        # Step 2: Select model based on directive's specification
        model = self.select_model(directive)
        
        # Step 3: Fill params using selected model
        params = await model.extract_params(
            directive=directive,
            conversation=conversation
        )
        
        # Step 4: Execute with specified model
        result = await self.mcp_client.call_tool(
            "execute",
            arguments={
                "item_type": "directive",
                "item_id": directive_id,
                "action": "run",
                "params": params,
                "model": directive.get("model")
            }
        )
        
        return result
    
    async def handle_execute(self, arguments: Dict, conversation: List[Dict]) -> Any:
        """
        Handle execute() primitive: execute directly
        
        Happens when: Full directive + params in context (high score >0.85).
        Predictive model already filled params, just execute with specified model.
        """
        
        # Params already filled by predictive model
        directive_id = arguments.get("item_id") or arguments.get("directive")
        params = arguments.get("params", {})
        model_tag = arguments.get("model")  # Directive specified model
        
        # Execute immediately with appropriate model
        result = await self.mcp_client.call_tool(
            "execute",
            arguments={
                "item_type": "directive",
                "item_id": directive_id,
                "action": "run",
                "params": params,
                "model": model_tag
            }
        )
        
        return result
```

## Training the Predictive Model

### Training Data Generation

```python
# generate_predictive_training_data.py

def generate_conversation_to_directive_data():
    """
    Generate training data: conversation snippets → directive predictions
    """
    
    training_data = []
    
    # Example 1: Email validation conversation
    training_data.append({
        "conversation": [
            "I need to check if these email addresses are valid",
            "Sure, I can help validate emails",
            "Here are 50 addresses from the CSV"
        ],
        "predicted_directives": [
            "email_validator",      # High relevance
            "email_enrichment",     # Medium relevance
            "csv_parser",           # Medium relevance
            "validate_leads"        # Low relevance
        ],
        "confidence": 0.85
    })
    
    # Example 2: Git workflow conversation
    training_data.append({
        "conversation": [
            "The feature is done",
            "Great! Ready to commit?",
            "Yes, let's commit and push to main"
        ],
        "predicted_directives": [
            "git_workflow",         # High relevance
            "git_commit_push",      # High relevance
            "code_review",          # Medium relevance
            "run_tests"             # Medium relevance
        ],
        "confidence": 0.92
    })
    
    # Example 3: Deployment conversation
    training_data.append({
        "conversation": [
            "Tests are passing",
            "Nice! Should we deploy to staging?",
            "Yes, deploy to staging first then production"
        ],
        "predicted_directives": [
            "deploy_staging",       # High relevance
            "deploy_production",    # High relevance
            "run_integration_tests",# Medium relevance
            "rollback_deploy"       # Low relevance (safety net)
        ],
        "confidence": 0.88
    })
    
    return training_data


def generate_negative_examples():
    """
    Negative examples where directives are NOT needed.
    """
    
    return [
        {
            "conversation": [
                "What's the weather like?",
                "I don't have weather information",
                "Oh okay, no problem"
            ],
            "predicted_directives": [],
            "confidence": 0.1  # Very low, no directives match
        },
        {
            "conversation": [
                "Tell me a joke",
                "Why did the developer quit? He didn't get arrays!",
                "Haha nice"
            ],
            "predicted_directives": [],
            "confidence": 0.05
        }
    ]


def augment_with_synthetic_data():
    """
    Use LLM to generate more training examples.
    """
    
    # Use Claude/GPT-4 to generate conversation patterns
    prompt = """Generate 10 conversation snippets that would lead to needing the "email_validator" directive.

Each should be 2-3 turns, natural language, and clearly indicate email validation is needed.

Format as JSON array."""
    
    # ... generate with LLM ...
```

### Fine-Tuning Recipe

```bash
# Train predictive context model

# 1. Prepare training data
python generate_predictive_training_data.py \
  --directive-db .ai/directives/ \
  --output training_data/predictive_context.jsonl \
  --augment-with-llm

# 2. Fine-tune E5-small on directive prediction
python train_predictive_model.py \
  --base-model "e5-small-v2" \
  --training-data training_data/predictive_context.jsonl \
  --output models/predictive-context-e5.bin \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 2e-5

# 3. Build directive embeddings index
python build_directive_index.py \
  --model models/predictive-context-e5.bin \
  --directives .ai/directives/ \
  --output indexes/directive-faiss.index

# 4. Test prediction latency
python benchmark_predictive_model.py \
  --model models/predictive-context-e5.bin \
  --index indexes/directive-faiss.index
```

## Performance Characteristics

### Latency Breakdown

| Operation                      | Latency | Notes                           |
| ------------------------------ | ------- | ------------------------------- |
| **Conversation analysis**      | 5-10ms  | Keyword extraction, NER         |
| **Semantic embedding**         | 8-15ms  | E5-small encoding               |
| **Vector search (10K dirs)**   | 3-8ms   | FAISS with IVF index            |
| **Vector search (100K dirs)**  | 5-12ms  | FAISS with IVF index            |
| **Load directive metadata**    | 2-5ms   | Disk/cache read                 |
| **Update FunctionGemma cache** | 1-3ms   | Context injection               |
| **Total per turn**             | 20-50ms | Runs in background, non-blocking |

### Cache Hit Rates

With good predictive model:

| Scenario                        | Cache Hit Rate | Speedup       |
| ------------------------------- | -------------- | ------------- |
| **Focused workflow**            | 80-95%         | 30-40% faster |
| **Related tasks**               | 60-75%         | 20-30% faster |
| **Topic switches**              | 30-50%         | 10-15% faster |
| **Random/exploratory**          | 10-20%         | Minimal       |
| **Overall (typical use)**       | 50-70%         | 25% faster    |

## Deployment

### System Architecture

```python
# main.py - Full Lilux system with predictive context

import asyncio
from frontend_model import FrontendModel
from predictive_context import PredictiveContextModel
from function_gemma import FunctionGemmaRouter
from backend_llm import BackendLLM

class LiluxAgent:
    def __init__(self):
        # Frontend conversational model
        self.frontend = FrontendModel("phi-3-mini")  # 3.8B
        
        # Predictive context model (runs continuously)
        self.predictive = PredictiveContextModel("e5-small-v2")  # 33M
        
        # FunctionGemma router
        self.gemma = FunctionGemmaRouter("function-gemma-270m")  # 270M
        
        # Execution Layer (multiple models available)
        self.execution = ExecutionLayer()
        
        # Connect predictive model to FunctionGemma
        self.predictive.gemma_router = self.gemma
        
        # Conversation history
        self.conversation = []
    
    async def chat(self, user_message: str):
        """Main conversation loop with 4-model architecture"""
        
        # 1. Frontend model processes message
        response_stream = self.frontend.stream_response(user_message)
        
        assistant_response = ""
        tool_intents = []
        
        async for token in response_stream:
            assistant_response += token
            
            # Detect [TOOL: ...] markers
            if "[TOOL:" in token:
                tool_intent = self.extract_tool_intent(assistant_response)
                tool_intents.append(tool_intent)
        
        # 2. Update predictive context (runs in background)
        # This searches for directives and scores them into hot/warm/cold
        asyncio.create_task(
            self.predictive.analyze_conversation_turn(
                user_message,
                assistant_response
            )
        )
        
        # 3. Execute tool intents (if any)
        for tool_intent in tool_intents:
            # FunctionGemma predicts primitive using unified context
            # (directives with varying detail levels based on prediction scores)
            routing = await self.gemma.predict(tool_intent)
            
            # Execution Layer runs the primitive
            # Directive specifies which model to use
            # The primitive itself determines the flow:
            # - search(): Full search→load→execute
            # - load(): Load→execute
            # - execute(): Execute directly
            result = await self.execution.handle_primitive(
                primitive=routing["primitive"],
                arguments=routing["arguments"],
                conversation_context=self.conversation
            )
            
            print(f"✅ {routing['primitive']}() execution complete")
        
        # 4. Add to conversation history
        self.conversation.append({
            "user": user_message,
            "assistant": assistant_response
        })
        
        return assistant_response
```

### Resource Requirements

```
Total Models Running:

1. Frontend Model (Phi-3-mini): 3.8B → 2-4GB RAM
2. Predictive Context (E5-small): 33M → 200MB RAM
3. FunctionGemma Router: 270M → 500MB RAM
4. Execution Models: Variable (depends on directives)

**Execution Models** (directive-specified):
- Claude Sonnet 4 (API): No RAM, $0.08/directive
- Llama 3.3 70B (local): +35GB RAM (quantized)
- Qwen 72B (local): +40GB RAM (quantized)
- Fine-tuned specialists: Varies by model

Total RAM:
- Hybrid (API execution): ~3-5GB RAM
- Local (70B execution): ~40-45GB RAM
- Mixed (some API, some local): ~10-20GB RAM

Can run on:
- MacBook Pro M1 Max (64GB RAM): All local
- Desktop with 64GB+ RAM + GPU: All local
- Hybrid: Frontend + Predictive + Gemma local (8GB+), Handler API
- Cloud: g5.4xlarge AWS (64GB) or g4dn.xlarge (16GB) + API
```

## Key Insights

1. **Four-Layer Architecture**:
   - Frontend: Generates [TOOL:] intents
   - Predictive: Continuously searches and scores directives
   - FunctionGemma: Routes to primitive (search/load/execute)
   - Execution: Runs primitives with **full MCP tooling access** (model varies by directive)

2. **ONE Unified Context with Scoring**:
   - Not three separate contexts (hot/warm/cold)
   - ONE context window with varying detail per directive
   - Prediction score determines detail level
   - High score → Full details, can execute()
   - Medium score → Schema only, needs load()
   - Low score → Minimal info, needs search()

3. **Predictive Model Outputs Searches**:
   - Generates multiple search queries per conversation turn
   - Executes vector searches across directive database
   - Scores results (0.0-1.0)
   - Loads scored directives into FunctionGemma's context

4. **Execution Layer = MCP Tool Expert**:
   - Has access to ALL MCP tools (git, file, database, API, etc.)
   - Reads directive XML to know which tools to call AND which model to use
   - Executes directive steps with full tool knowledge
   - **Directive specifies the model**: Claude for reasoning, Llama for local, etc.

5. **Background Processing**:
   - Predictive model runs after EVERY turn
   - FunctionGemma runs only when [TOOL:] triggered
   - Execution Layer receives primitive and runs appropriate model

6. **Semantic, Not Keywords**:
   - Uses embeddings, not string matching
   - Understands "check emails" → email_validator
   - Understands "push code" → git_workflow

7. **Scalable**:
   - FAISS handles millions of directives
   - Only top 10-15 loaded into context
   - Context window stays manageable

8. **Graceful Degradation**:
   - No predictions? FunctionGemma predicts search()
   - Wrong directives? Handler does full search flow
   - Predictive model down? System still works (slower)

9. **Self-Expanding Intelligence** 🚀:
   - Directives can CREATE other directives
   - System learns and expands capabilities over time
   - Exponential growth: More directives → Better predictions → More use cases
   - Meta-directives for optimization, analysis, and self-improvement

## Self-Expanding Intelligence: Directives Creating Directives

### The Meta-Learning Loop

The Execution Layer doesn't just run directives—it can execute directives that **CREATE other directives**, leading to exponential capability growth:

```python
class MetaDirective:
    """
    Example: A directive that creates other directives based on patterns.
    This directive might specify Claude Sonnet for complex analysis.
    """
    
    async def analyze_and_create_directive(self, execution: ExecutionLayer, context: Dict):
        """
        Directive: analyze_workflow_and_optimize
        
        Steps:
        1. Analyze user's recent workflow patterns
        2. Identify repetitive tasks
        3. Generate a new directive to automate them
        4. Test the new directive
        5. Add to directive database
        """
        
        # Step 1: Analyze patterns
        workflow_patterns = await handler.model.analyze_patterns(
            conversation_history=context["history"],
            time_window_days=7
        )
        
        # Step 2: Identify optimization opportunities
        repetitive_tasks = self.find_repetitive_tasks(workflow_patterns)
        
        for task in repetitive_tasks:
            # Step 3: Generate new directive
            new_directive_xml = await handler.model.generate_directive(
                task_description=task.description,
                observed_steps=task.steps,
                mcp_tools_used=task.tools
            )
            
            # Step 4: Validate
            is_valid = await self.validate_directive(new_directive_xml)
            
            if is_valid:
                # Step 5: Create via MCP
                result = await handler.mcp_client.call_tool(
                    "execute",
                    arguments={
                        "item_type": "directive",
                        "action": "create",
                        "name": task.suggested_name,
                        "content": new_directive_xml
                    }
                )
                
                print(f"✨ Created new directive: {task.suggested_name}")
                
                # Step 6: Rebuild directive embeddings
                await self.rebuild_directive_index()
```

### Exponential Growth Pattern

```
Week 1:
  50 base directives (manually created)
  ↓
Week 2:
  User triggers meta-directive: "optimize my git workflow"
  → Creates: git_smart_commit, git_auto_branch, git_sync_all
  → 53 directives
  ↓
Week 3:
  Predictive model now includes new directives
  → Better predictions for git workflows
  → User discovers more use cases
  → Triggers: "automate my deployment process"
  → Creates: deploy_with_tests, rollback_smart, monitor_deploy
  → 56 directives
  ↓
Week 4:
  New directives used in combination
  → Meta-directive detects pattern
  → Creates: full_ci_cd_pipeline (composite directive)
  → 57 directives
  ↓
Month 2:
  → 75 directives (50 base + 25 learned)
  ↓
Month 6:
  → 150 directives (exponential growth from combinations)
  ↓
Year 1:
  → 500+ directives (self-optimizing, user-specific)
```

### Example: Self-Improving Directive

```xml
<directive name="improve_email_validation">
  <description>
    Analyzes email validation failures and creates improved validator
  </description>
  
  <inputs>
    <input name="failure_logs" type="array" description="Recent validation failures" />
  </inputs>
  
  <process>
    <step id="1">
      <description>Analyze failure patterns</description>
      <tool>execute</tool>
      <arguments>
        <directive>analyze_logs</directive>
        <params>
          <logs>{{failure_logs}}</logs>
          <pattern_type>validation_errors</pattern_type>
        </params>
      </arguments>
    </step>
    
    <step id="2">
      <description>Generate improved validation rules</description>
      <tool>ai_generate</tool>
      <arguments>
        <prompt>
          Based on these validation failures: {{step1.patterns}}
          Generate improved regex and validation rules.
        </prompt>
      </arguments>
    </step>
    
    <step id="3">
      <description>Create new directive with improved rules</description>
      <tool>execute</tool>
      <arguments>
        <directive>create_directive</directive>
        <params>
          <name>email_validator_v2</name>
          <rules>{{step2.improved_rules}}</rules>
          <fallback>email_validator</fallback>
        </params>
      </arguments>
    </step>
    
    <step id="4">
      <description>Test new directive</description>
      <tool>execute</tool>
      <arguments>
        <directive>email_validator_v2</directive>
        <params>
          <test_cases>{{failure_logs}}</test_cases>
        </params>
      </arguments>
    </step>
    
    <step id="5" condition="step4.success_rate > 0.95">
      <description>Replace old directive if better</description>
      <tool>execute</tool>
      <arguments>
        <directive>update_directive</directive>
        <params>
          <name>email_validator</name>
          <new_version>email_validator_v2</new_version>
        </params>
      </arguments>
    </step>
  </process>
</directive>
```

### Intelligence Expansion via Predictive Model

As new directives are created:

1. **Automatic Indexing**: New directives automatically added to vector database
2. **Embedding Generation**: Semantic embeddings generated for search
3. **Predictive Model Learns**: New directives appear in future predictions
4. **Usage Patterns**: System learns when to suggest new directives
5. **Combination Discovery**: Detects when multiple directives often used together → creates composite

```python
class SelfExpandingPredictiveModel(PredictiveContextModel):
    """
    Predictive model that learns from new directives.
    """
    
    async def on_directive_created(self, new_directive: Directive):
        """
        Called automatically when any directive creates a new directive.
        """
        
        # 1. Generate embedding
        embedding = self.semantic_model.encode(
            f"{new_directive.name} {new_directive.description}"
        )
        
        # 2. Add to vector index
        self.vector_index.add(new_directive.id, embedding)
        
        # 3. Update directive database
        await self.directive_db.add_directive(new_directive)
        
        # 4. Analyze creation context
        creation_context = self.get_recent_conversation()
        
        # 5. Learn trigger patterns
        await self.learn_trigger_pattern(
            directive=new_directive,
            context=creation_context
        )
        
        print(f"🌱 Learned new directive: {new_directive.name}")
        print(f"   Now predicting in contexts: {self.predict_usage_scenarios(new_directive)}")
    
    async def learn_trigger_pattern(self, directive: Directive, context: List[Dict]):
        """
        Learn when this directive should be predicted.
        """
        
        # Extract signals that led to this directive's creation
        signals = self.extract_signals(context)
        
        # Store pattern for future predictions
        self.usage_patterns[directive.id] = {
            "keywords": signals["keywords"],
            "actions": signals["actions"],
            "domain": signals["domain"],
            "co_occurring_directives": self.find_co_occurring(context)
        }
```

### The Lilux Learning Cycle

```
User works → Patterns emerge → Meta-directives activate
    ↓
New directives created → Indexed automatically
    ↓
Predictive model learns triggers → Better predictions
    ↓
User discovers new capabilities → More usage
    ↓
More patterns emerge → More meta-directives
    ↓
EXPONENTIAL GROWTH 🚀
```

## Future Enhancements

### 1. Personalized Predictions

Learn user patterns over time:
```python
class PersonalizedPredictor(PredictiveContextModel):
    """
    Learn user's directive usage patterns.
    """
    
    def __init__(self, user_id: str):
        super().__init__()
        self.user_history = load_user_history(user_id)
        self.usage_patterns = self.analyze_patterns()
    
    def predict_directives(self, signals: Dict) -> List[Directive]:
        # Blend semantic search with user history
        semantic_results = super().predict_directives(signals)
        user_favorites = self.get_frequently_used(signals["domain"])
        
        # User's favorites get priority
        return self.merge_with_bias(semantic_results, user_favorites)
```

### 2. Multi-Hop Prediction

Predict directive chains:
```python
# Conversation: "Deploy to staging"
# Predict not just: deploy_staging
# But also: run_tests, deploy_production, rollback_deploy
# (anticipate what might come next)
```

### 3. Confidence-Based Caching

Adjust cache size based on confidence:
```python
if confidence > 0.9:
    load_top_10_directives()  # High confidence, load more
elif confidence > 0.7:
    load_top_5_directives()   # Medium confidence
else:
    load_top_3_directives()   # Low confidence, load less
```

### 4. Directive Evolution

Directives improve themselves based on success/failure:
```python
# After 100 executions, directive analyzes its own performance
# Creates improved version if success rate < 95%
# A/B tests new version against old
# Automatically switches if better
```

---

## Conclusion

The **Predictive Context Model** is the "psychic brain" of Lilux—continuously analyzing conversation, generating searches for directives, scoring results, and populating FunctionGemma's unified context with varying detail levels. All in the background, before the user even triggers a tool intent.

But it's more than just fast routing. Combined with the Execution Layer's access to full MCP tooling (with dynamic model selection) and the ability for directives to create directives, Lilux becomes a **self-expanding intelligence**:

- **Today**: 50 base directives, good predictions
- **Next Month**: 100 directives (50 learned from usage patterns)
- **Next Year**: 500+ directives (exponential growth from meta-learning)

The system doesn't just get faster—it gets **smarter**. More capable. More personalized. It learns your workflows, creates optimizations, and expands its own intelligence autonomously.

**The result**: 
- ⚡ Instant tool routing (30-40% faster with predictive context)
- 🧠 Self-improving directives (learns from failures)
- 🌱 Exponential capability growth (directives creating directives)
- 🔮 Feels psychic (knows what you need before you ask)

This is not just an AI assistant. This is an **AI operating system that evolves**. 🚀
