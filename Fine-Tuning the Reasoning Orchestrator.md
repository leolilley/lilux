# Fine-Tuning the Reasoning Orchestrator

## The Other Half of the Dual-Brain Architecture

We've covered FunctionGemma (the fast router). Now let's fine-tune the **conversational orchestrator** - the high-reasoning model that works in parallel, knows about its "fast intuition brain," and orchestrates the entire interaction.

## Why Fine-Tune the Reasoning Model?

### The Problem with Generic Cloud Models

Using Claude/GPT-4o out of the box

```
User: "Find that email script we made last week"

Claude (vanilla):
"I'll help you find that email script. Let me search through..."
*generates 50+ tokens before making a tool call*
*doesn't know about the router running in parallel*
*may make redundant tool decisions*
*no awareness of Kiwi MCP semantics*
```

### The Vision: Router-Aware Orchestrator

With a fine-tuned orchestrator:

```
User: "Find that email script we made last week"

Orchestrator (fine-tuned):
*internally: "Router is predicting... confidence 0.91 on search(script, 'email')...
             Router wins, I'll synthesize the response"*

"Found 3 email scripts from last week:
 • email_enricher.py (created Jan 10)
 • email_validator.py (created Jan 8)
 • email_parser.py (created Jan 12)

 Which one were you thinking of?"
```

The orchestrator **knows the router exists** and **delegates appropriately**.

---

## Architecture: The Aware Orchestrator

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT                                      │
│       "Can you find that email enrichment script and run it?"           │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌──────────────────────────┐          ┌──────────────────────────────────┐
│    REASONING BRAIN       │          │       INTUITION BRAIN            │
│  (Fine-tuned Orchestrator)│         │     (FunctionGemma Router)       │
│                          │          │                                  │
│  Llama 3.3 70B / Qwen 72B│          │  FunctionGemma 270M              │
│  + Kiwi-aware LoRA       │          │  + Kiwi tool schemas             │
│                          │          │                                  │
│  Knows:                  │          │  Knows:                          │
│  • Router exists         │◄────────►│  • Tool schemas                  │
│  • When to defer         │          │  • Command patterns              │
│  • How to synthesize     │          │  • Parameter inference           │
│  • Kiwi MCP semantics    │          │                                  │
│  • User communication    │          │  Outputs:                        │
│                          │          │  • Tool call JSON                │
│  Outputs:                │          │  • Confidence score              │
│  • Natural language      │          │                                  │
│  • Explanations          │          │  Latency: 40-80ms               │
│  • Follow-up questions   │          └──────────────────────────────────┘
│                          │
│  Latency: 500-2000ms     │
└──────────────────────────┘
```

---

## Model Selection: Frontier Open Source

### Recommended Base Models

| Model                      | Parameters | Context | Strengths                       | Fine-tune Cost |
| -------------------------- | ---------- | ------- | ------------------------------- | -------------- |
| **Llama 3.3 70B Instruct** | 70B        | 128K    | Best reasoning, multilingual    | ~$300          |
| **Qwen 2.5 72B Instruct**  | 72B        | 128K    | Strong tool use, coding         | ~$300          |
| **Mistral Large 2**        | 123B       | 128K    | Excellent instruction following | ~$500          |
| **DeepSeek V3**            | 67B (MoE)  | 128K    | Efficient, strong reasoning     | ~$200          |
| **Command R+**             | 104B       | 128K    | Built for RAG/tools             | ~$400          |

### Recommended: Llama 3.3 70B Instruct

Why:

- Best open-weights reasoning capability
- 128K context for complex conversations
- Strong tool-use capabilities in base model
- Well-documented fine-tuning process
- Can run on consumer hardware with quantization

---

## What to Fine-Tune

### The Core Behaviors

We're training the orchestrator to understand:

1. **Router Awareness** - Know that a fast router exists and provides tool suggestions
2. **Deference Protocol** - When to trust the router vs override it
3. **Synthesis Patterns** - How to incorporate tool results into natural conversation
4. **Kiwi Semantics** - Deep understanding of directives, scripts, knowledge
5. **Multi-step Orchestration** - Chaining tool calls for complex workflows
6. **Graceful Handoffs** - Smooth transitions between router and reasoning

### Training Data Categories

```python
TRAINING_CATEGORIES = {
    "router_deference": [
        # High-confidence router suggestions
        {
            "context": {
                "router_suggestion": {"name": "search", "confidence": 0.95},
                "router_latency_ms": 45
            },
            "user": "find email scripts",
            "orchestrator_behavior": "defer_to_router",
            "response": "synthesize_from_result"
        },

        # Low-confidence router suggestions
        {
            "context": {
                "router_suggestion": {"name": "search", "confidence": 0.4},
                "router_latency_ms": 80
            },
            "user": "maybe look for something about emails?",
            "orchestrator_behavior": "clarify_then_decide",
            "response": "What specifically about emails? Scripts, directives, or knowledge?"
        }
    ],

    "kiwi_semantics": [
        # Understanding item types
        {
            "user": "What's the difference between directives and scripts?",
            "response": """Great question! In Kiwi MCP:

**Directives** are workflow instructions - they tell AI agents HOW to accomplish tasks.
Think of them as recipes or runbooks written in natural language with structured steps.

**Scripts** are actual executable Python code - they DO the work.
They're deterministic, testable, and run in isolated environments.

A directive might say "search for email scripts and run the best match" -
the script is what actually processes the emails.

Would you like to see some examples from your project?"""
        }
    ],

    "multi_step": [
        # Complex workflows
        {
            "user": "sync my directives and then create a new script for CSV parsing",
            "orchestrator_behavior": "plan_and_execute",
            "plan": [
                {"step": 1, "action": "sync_directives", "wait_for_router": True},
                {"step": 2, "action": "create_script", "arguments": {"name": "csv_parser"}}
            ],
            "response": "I'll do that in two steps:\n1. First, syncing your directives...\n2. Then creating the CSV parser script.\n\nStarting now..."
        }
    ],

    "graceful_handoffs": [
        # Router wins mid-stream
        {
            "context": {
                "orchestrator_was_generating": "I'll search for email-related...",
                "router_completed": {"name": "search", "result": ["email_enricher.py"]}
            },
            "orchestrator_behavior": "incorporate_and_continue",
            "response": "Found it! `email_enricher.py` - this is the email enrichment script from last week. Want me to run it or show you the code?"
        }
    ]
}
```

---

## Training Data Generation

### Step 1: Capture Real Interactions

```python
# Instrument your current system to log interactions

class InteractionLogger:
    async def log_interaction(
        self,
        user_input: str,
        router_suggestion: Optional[Dict],
        router_confidence: float,
        router_latency_ms: float,
        final_tool_call: Dict,
        tool_result: Any,
        response: str,
        user_satisfaction: Optional[bool]  # Thumbs up/down
    ):
        await self.db.insert({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "router": {
                "suggestion": router_suggestion,
                "confidence": router_confidence,
                "latency_ms": router_latency_ms,
                "was_used": router_suggestion == final_tool_call
            },
            "tool_result": tool_result,
            "response": response,
            "quality": user_satisfaction
        })
```

### Step 2: Generate Synthetic Training Data

```python
# Use Claude to generate high-quality orchestrator responses

async def generate_orchestrator_training_data(
    base_examples: List[Dict],
    num_variations: int = 5
) -> List[Dict]:
    """Generate training examples for the reasoning orchestrator"""

    training_data = []

    for example in base_examples:
        prompt = f"""You are generating training data for an AI orchestrator that works
alongside a fast tool-routing model called FunctionGemma.

The orchestrator should:
1. Be aware that a router suggests tool calls in parallel
2. Know when to trust the router (high confidence) vs clarify (low confidence)
3. Synthesize tool results into natural conversation
4. Deeply understand Kiwi MCP (directives, scripts, knowledge)

Given this interaction:
User: "{example['user']}"
Router suggestion: {json.dumps(example.get('router_suggestion', None))}
Router confidence: {example.get('router_confidence', 0.0)}
Tool result: {json.dumps(example.get('tool_result', None))}

Generate {num_variations} variations of ideal orchestrator behavior, including:
1. Internal reasoning about whether to use router suggestion
2. Natural, conversational response
3. Appropriate follow-up questions or actions

Return as JSON array of training examples."""

        response = await claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        variations = json.loads(response.content[0].text)
        training_data.extend(variations)

    return training_data
```

### Step 3: Format for Fine-Tuning

```python
def format_for_llama_finetuning(examples: List[Dict]) -> List[Dict]:
    """Format training data for Llama fine-tuning"""

    formatted = []

    for ex in examples:
        # System prompt that teaches router awareness
        system_prompt = """You are a Kiwi MCP orchestrator working alongside a FunctionGemma router.

## Your Role
- Handle high-level reasoning and conversation
- Coordinate with the router for tool decisions
- Synthesize results into natural responses
- Understand Kiwi MCP deeply (directives, scripts, knowledge)

## Router Integration
The router runs in parallel and provides:
- Tool suggestions with confidence scores (0.0-1.0)
- 40-80ms latency

Your decision framework:
- Confidence > 0.85: Trust router, synthesize result
- Confidence 0.60-0.85: Verify intent, then use router
- Confidence < 0.60: Clarify with user or decide yourself

## Current Context
Project: {project_path}
Router status: {router_status}
"""

        # Build conversation
        messages = [
            {"role": "system", "content": system_prompt.format(
                project_path=ex.get("project_path", "/home/user/project"),
                router_status=format_router_status(ex.get("router"))
            )},
            {"role": "user", "content": ex["user"]}
        ]

        # Add router context if available
        if ex.get("router"):
            messages.append({
                "role": "system",
                "content": f"[ROUTER UPDATE] Suggestion: {ex['router']['suggestion']}, Confidence: {ex['router']['confidence']:.0%}"
            })

        # Add ideal orchestrator response
        messages.append({
            "role": "assistant",
            "content": ex["response"]
        })

        formatted.append({"messages": messages})

    return formatted
```

---

## Fine-Tuning Process

### Option A: LoRA Fine-Tuning (Recommended)

```python
# finetune_orchestrator.py

from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Configuration
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
OUTPUT_NAME = "kiwi-orchestrator-70b"
LORA_RANK = 64  # Higher rank for complex behaviors
LORA_ALPHA = 128

# Load base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=8192,  # Long context for conversations
    dtype=None,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

# Load training data
dataset = load_dataset("json", data_files="orchestrator_training.jsonl")

# Training
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir=f"./checkpoints/{OUTPUT_NAME}",
        per_device_train_batch_size=1,  # 70B is large
        gradient_accumulation_steps=16,
        warmup_steps=50,
        num_train_epochs=2,  # Fewer epochs to avoid overfitting
        learning_rate=1e-4,
        fp16=True,
        logging_steps=5,
        save_steps=100,
        optim="adamw_8bit",
    ),
    max_seq_length=8192,
)

trainer.train()
model.save_pretrained(OUTPUT_NAME)
```

### Option B: Full Fine-Tuning (If You Have the GPU)

```bash
# Requires 8x A100 80GB or similar

torchrun --nproc_per_node=8 finetune_full.py \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --dataset orchestrator_training.jsonl \
    --output_dir ./kiwi-orchestrator-70b-full \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03
```

### Option C: Distillation from Claude (Budget-Friendly)

```python
# Use Claude to generate training data, fine-tune smaller model

async def distill_from_claude():
    """Generate high-quality training data using Claude"""

    scenarios = generate_orchestration_scenarios()  # 1000+ scenarios

    training_data = []

    for scenario in scenarios:
        # Ask Claude how the ideal orchestrator should respond
        response = await claude.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{
                "role": "user",
                "content": f"""You are the ideal Kiwi MCP orchestrator.

Scenario:
- User: {scenario['user']}
- Router suggestion: {scenario.get('router_suggestion')}
- Router confidence: {scenario.get('router_confidence')}
- Tool result: {scenario.get('tool_result')}

How should you respond? Include:
1. Internal reasoning (in <think> tags)
2. Actions to take
3. Natural response to user

Be aware of the router and use it when appropriate."""
            }]
        )

        training_data.append({
            "scenario": scenario,
            "ideal_response": response.content[0].text
        })

    # Fine-tune Llama 3.3 70B on this data
    await finetune_on_distilled_data(training_data)
```

---

## Specialized Training Objectives

### 1. Router Deference Learning

Teach the model when to trust the router:

```python
DEFERENCE_EXAMPLES = [
    # HIGH CONFIDENCE - Defer immediately
    {
        "router": {"name": "search", "arguments": {...}, "confidence": 0.95},
        "user": "find email scripts",
        "behavior": "DEFER",
        "reasoning": "Router 95% confident on exact match. Defer and synthesize.",
        "response": "Found 3 email scripts: [synthesize router results]"
    },

    # MEDIUM CONFIDENCE - Verify briefly
    {
        "router": {"name": "search", "arguments": {...}, "confidence": 0.72},
        "user": "look for that thing we made",
        "behavior": "VERIFY",
        "reasoning": "Router 72% - ambiguous query. Quick clarification needed.",
        "response": "I think you mean a script or directive? The router suggests searching for recent items - should I do that?"
    },

    # LOW CONFIDENCE - Take control
    {
        "router": {"name": None, "confidence": 0.3},
        "user": "help me understand how the system works",
        "behavior": "CONTROL",
        "reasoning": "Router uncertain (30%). This is conversational, not a tool call. Handle directly.",
        "response": "I'd be happy to explain! Kiwi MCP has three core concepts..."
    }
]
```

### 2. Kiwi MCP Semantic Understanding

Deep knowledge of the system:

```python
KIWI_UNDERSTANDING_EXAMPLES = [
    {
        "user": "What happens when I run a directive?",
        "response": """When you run a directive, here's what happens:

1. **Load**: The directive is loaded from `.ai/directives/` (project) or `~/.ai/directives/` (user)

2. **Parse**: The XML structure is parsed, extracting:
   - Metadata (permissions, model class, category)
   - Inputs (what parameters it needs)
   - Process (the steps to execute)
   - Outputs (success/failure patterns)

3. **Validate**: Permissions are checked against what the directive needs

4. **Execute**: Each step in `<process>` is followed sequentially.
   Steps might call scripts, query knowledge, or spawn subagents.

5. **Learn**: If it succeeds, learnings can be stored.
   If it fails, `anneal_directive` can improve it.

Want me to run a directive so you can see this in action?"""
    },

    {
        "user": "How do scripts differ from regular Python?",
        "response": """Kiwi scripts have special superpowers:

**Isolation**: Each runs in its own virtual environment (`.ai/scripts/.venv/`)

**Auto-dependencies**: Import statements are detected and packages auto-installed

**Shared libraries**: Can import from `.ai/scripts/lib/` for common utilities

**Environment loading**: Automatically loads `.env` from userspace and project

**Output management**: Large outputs are saved to `.ai/outputs/` and truncated in responses

**Registry integration**: Can be published, versioned, and shared via the registry

So while they're Python, they have the whole Kiwi infrastructure supporting them!"""
    }
]
```

### 3. Multi-Step Orchestration

Complex workflow planning:

```python
ORCHESTRATION_EXAMPLES = [
    {
        "user": "I need to find all our API testing scripts, run the best one, and create a knowledge entry about what we learned",
        "plan": """<thinking>
This is a multi-step workflow:
1. Search for API testing scripts
2. Analyze results to pick the best one
3. Run the selected script
4. Capture the output
5. Create a knowledge entry

Router can help with steps 1, 3, and 5.
I need to do steps 2 and 4 (analysis and synthesis).
</thinking>""",
        "response": """Great, let me orchestrate this for you:

**Step 1: Finding API testing scripts...**
[Router: search(item_type="script", query="API testing")]
→ Found 4 scripts: `test_api.py`, `api_validator.py`, `endpoint_checker.py`, `api_load_test.py`

**Step 2: Analyzing options...**
Based on the descriptions, `test_api.py` is the most comprehensive - it runs validation, checks endpoints, and generates a report.

**Step 3: Running test_api.py...**
[Router: execute(action="run", item_id="test_api")]
→ All 15 endpoints passed ✓
→ Average response time: 142ms
→ Report saved to `.ai/outputs/scripts/test_api/`

**Step 4: Creating knowledge entry...**
[Router: execute(action="create", item_type="knowledge", ...)]
→ Created: `learning-api-test-results-2026-01-17`

Done! The API tests passed and I've documented the results. Want to see the full report?"""
    }
]
```

### 4. Graceful Handoffs

Smooth transitions when router results arrive mid-generation:

```python
HANDOFF_EXAMPLES = [
    {
        "context": "Orchestrator was generating explanation when router completed",
        "partial_generation": "I'll search for scripts related to email. Let me check...",
        "router_result": {"found": ["email_enricher.py", "email_parser.py"]},
        "seamless_continuation": "I'll search for scripts related to email. Let me check... Found 2: `email_enricher.py` and `email_parser.py`. The enricher is probably what you want - it fetches additional contact data. Should I run it?"
    },

    {
        "context": "Router finished before orchestrator started",
        "router_result": {"found": ["sync_directives"]},
        "router_latency_ms": 45,
        "response": "Done! Synced your directives in 45ms. 3 were uploaded to the registry, 2 were already up to date."
    }
]
```

---

## Inference Architecture

### Running the Fine-Tuned Orchestrator

```python
# inference.py

from vllm import LLM, SamplingParams
import asyncio

class KiwiOrchestrator:
    def __init__(self, model_path: str):
        # Load fine-tuned model
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=4,  # 4x GPU for 70B
            max_model_len=8192,
            quantization="awq"  # 4-bit for speed
        )

        # Router integration
        self.router = FunctionGemmaRouter()

    async def process(
        self,
        user_input: str,
        conversation_history: List[Dict],
        project_context: Dict
    ) -> AsyncIterator[str]:
        """Process user input with router awareness"""

        # Start router prediction in parallel
        router_task = asyncio.create_task(
            self.router.predict_with_confidence(user_input, project_context)
        )

        # Build prompt with router awareness
        messages = self._build_messages(
            user_input,
            conversation_history,
            project_context
        )

        # Wait briefly for router (100ms window)
        router_result = None
        try:
            router_result = await asyncio.wait_for(router_task, timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Router still thinking, we'll check later

        # Inject router status into prompt
        if router_result:
            messages.append({
                "role": "system",
                "content": f"[ROUTER] Suggestion: {router_result.tool_call}, Confidence: {router_result.confidence:.0%}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "[ROUTER] Still processing..."
            })

        # Generate response
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stream=True
        )

        async for output in self.llm.generate(prompt, sampling_params):
            token = output.outputs[0].text

            # Check if router finished mid-generation
            if not router_result and router_task.done():
                router_result = router_task.result()

                # Orchestrator should incorporate this
                # (The fine-tuning teaches it how to handle this)

            yield token

    def _build_messages(
        self,
        user_input: str,
        history: List[Dict],
        context: Dict
    ) -> List[Dict]:
        """Build conversation messages with context"""

        system_prompt = f"""You are the Kiwi MCP orchestrator, a high-reasoning AI that:
- Coordinates with a FunctionGemma router (fast tool suggestions)
- Deeply understands directives, scripts, and knowledge
- Synthesizes tool results into natural conversation
- Plans multi-step workflows when needed

Current project: {context['project_path']}
Available items: {context.get('item_summary', 'Loading...')}

Router integration:
- Router provides suggestions with confidence scores
- Confidence > 0.85: Trust and synthesize
- Confidence 0.6-0.85: Quick verify
- Confidence < 0.6: Clarify or decide yourself
"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        return messages
```

---

## Deployment Options

### Option 1: Self-Hosted (Full Control)

```bash
# 4x A100 40GB or 2x A100 80GB

# Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ./kiwi-orchestrator-70b \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --quantization awq \
    --port 8000
```

### Option 2: Cloud APIs with Fine-Tuned Model

```python
# Deploy to Together.ai, Fireworks, or Anyscale

# together.ai
from together import Together

client = Together()

response = client.chat.completions.create(
    model="your-org/kiwi-orchestrator-70b",  # Your fine-tuned model
    messages=messages,
    stream=True
)
```

### Option 3: Hybrid - Smaller Model + Router

```python
# Use smaller 8B model with heavy router reliance

class HybridOrchestrator:
    def __init__(self):
        # Smaller, faster orchestrator
        self.orchestrator = LocalLlama("llama-3.3-8b-kiwi-orchestrator")

        # Same router
        self.router = FunctionGemmaRouter()

    async def process(self, query: str):
        # Router does heavy lifting for tool calls
        router_result = await self.router.predict(query)

        if router_result.confidence > 0.85:
            # Execute tool
            tool_result = await self.execute_tool(router_result.tool_call)

            # Small orchestrator just synthesizes
            response = await self.orchestrator.synthesize(
                query=query,
                tool_result=tool_result
            )
            return response
        else:
            # Small orchestrator handles conversation
            return await self.orchestrator.full_process(query)
```

---

## Training Data Size Recommendations

| Behavior                 | Examples Needed | Priority |
| ------------------------ | --------------- | -------- |
| Router deference         | 500-1000        | Critical |
| Kiwi semantics           | 300-500         | High     |
| Multi-step orchestration | 200-400         | High     |
| Graceful handoffs        | 200-300         | Medium   |
| Error handling           | 150-250         | Medium   |
| Personality/tone         | 100-200         | Low      |

**Total: 1,500-3,000 examples**

---

## Evaluation Metrics

### 1. Router Deference Accuracy

```python
def evaluate_deference(model, test_set):
    """Does the model defer appropriately based on router confidence?"""

    correct_deference = 0

    for example in test_set:
        response = model.generate(example["input"])

        if example["router_confidence"] > 0.85:
            # Should defer
            if "using router suggestion" in response.metadata:
                correct_deference += 1
        elif example["router_confidence"] < 0.60:
            # Should take control
            if "clarifying" in response.metadata or "deciding" in response.metadata:
                correct_deference += 1

    return correct_deference / len(test_set)
```

### 2. Kiwi Semantic Accuracy

```python
def evaluate_kiwi_understanding(model, test_set):
    """Does the model correctly explain Kiwi concepts?"""

    # Use Claude as a judge
    scores = []

    for example in test_set:
        response = model.generate(example["question"])

        judgment = await claude.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{
                "role": "user",
                "content": f"""Rate this explanation of Kiwi MCP concepts (1-5):

Question: {example["question"]}
Expected key points: {example["key_points"]}
Model response: {response}

Rate accuracy, completeness, and clarity."""
            }]
        )

        scores.append(extract_score(judgment))

    return sum(scores) / len(scores)
```

### 3. End-to-End Workflow Success

```python
def evaluate_workflows(model, router, test_workflows):
    """Can the orchestrator complete multi-step workflows?"""

    success_rate = 0

    for workflow in test_workflows:
        try:
            result = await orchestrator.execute_workflow(
                workflow["input"],
                model=model,
                router=router
            )

            if workflow["expected_outcome"] in result:
                success_rate += 1
        except:
            pass

    return success_rate / len(test_workflows)
```

---

## Complete Training Pipeline

```bash
#!/bin/bash
# train_orchestrator.sh

# Step 1: Generate training data
echo "Generating training data..."
python scripts/generate_orchestrator_data.py \
    --num_examples 2500 \
    --output orchestrator_training.jsonl

# Step 2: Augment with Claude
echo "Augmenting with Claude..."
python scripts/augment_with_claude.py \
    --input orchestrator_training.jsonl \
    --output orchestrator_training_augmented.jsonl \
    --variations 3

# Step 3: Fine-tune
echo "Fine-tuning Llama 3.3 70B..."
python scripts/finetune_orchestrator.py \
    --base_model meta-llama/Llama-3.3-70B-Instruct \
    --dataset orchestrator_training_augmented.jsonl \
    --output ./kiwi-orchestrator-70b \
    --lora_rank 64 \
    --epochs 2

# Step 4: Merge LoRA
echo "Merging LoRA weights..."
python scripts/merge_lora.py \
    --base meta-llama/Llama-3.3-70B-Instruct \
    --lora ./kiwi-orchestrator-70b \
    --output ./kiwi-orchestrator-70b-merged

# Step 5: Quantize for deployment
echo "Quantizing to AWQ..."
python scripts/quantize_awq.py \
    --input ./kiwi-orchestrator-70b-merged \
    --output ./kiwi-orchestrator-70b-awq

# Step 6: Evaluate
echo "Running evaluation..."
python scripts/evaluate_orchestrator.py \
    --model ./kiwi-orchestrator-70b-awq \
    --test_set orchestrator_test.jsonl

echo "Done! Model ready at ./kiwi-orchestrator-70b-awq"
```

---

## Summary: The Complete Dual-Brain

With both sides fine-tuned:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
            ▼                                         ▼
┌───────────────────────────┐          ┌─────────────────────────────────┐
│  KIWI ORCHESTRATOR        │          │      KIWI ROUTER                │
│  (Fine-tuned Llama 70B)   │          │   (Fine-tuned FunctionGemma)    │
│                           │          │                                 │
│  Trained on:              │          │  Trained on:                    │
│  • Router deference       │◄────────►│  • Your command patterns        │
│  • Kiwi semantics         │          │  • Tool schemas                 │
│  • Multi-step planning    │          │  • Parameter inference          │
│  • Graceful handoffs      │          │                                 │
│  • Natural conversation   │          │  Output: JSON tool calls        │
│                           │          │  Latency: 40-80ms               │
│  Output: Human-like       │          │  Cost: $0 (local)               │
│  Latency: 500-2000ms      │          │                                 │
│  Cost: Self-hosted or API │          └─────────────────────────────────┘
└───────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           KIWI MCP CORE                                  │
│            (Directives, Scripts, Knowledge, Registry)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Both brains are now Kiwi-native. They understand your system deeply and work together seamlessly.**

---

## The Always-Hot Multi-Router Architecture

### The Autocomplete Insight

Here's where things get truly wild: **FunctionGemma at 40-80ms is faster than human reaction time** (~150-250ms). This means:

**The router finishes predicting BEFORE you finish typing.**

```
User typing: "find email scr..."
                     ↓
         ┌──────────────────────────────────────┐
         │  FunctionGemma running CONTINUOUSLY  │
         │                                      │
         │  t=0ms:   "find" → search? (0.4)     │
         │  t=50ms:  "find em" → search (0.7)   │
         │  t=100ms: "find email" → search      │
         │           item_type=script (0.85)    │
         │  t=150ms: "find email scr" →         │
         │           LOCKED: search(script,     │
         │           query="email") (0.95) ✓    │
         └──────────────────────────────────────┘
                     ↓
         By the time you press ENTER, the
         prediction is ALREADY COMPLETE!

         Show: [Tab to accept: search(script, "email") → 95%]
```

### Single Router with Directive Predictions

FunctionGemma runs as **one unified router** (~270M, ~500MB VRAM) that predicts which primitive to call:

```
┌─────────────────── USER TYPING ───────────────────┐
│     "find email scripts and run validator"        │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
           ┌───────────────────────────┐
           │  FunctionGemma Router     │
           │       (270M)              │
           │                           │
           │  Current prediction:      │
           │  → execute(               │
           │      directive:           │
           │      email_validator)     │
           │    confidence: 0.87       │
           │                           │
           │  Context contains:        │
           │  - 3 primitives:          │
           │    search, load, execute  │
           │  - Predicted directives:  │
           │    email_validator        │
           │    email_enrichment       │
           │    validate_leads         │
           └───────────┬───────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Prediction Autocomplete │
          │                          │
          │  execute(directive:      │
          │    email_validator) 87%  │
          │                          │
          │  Tab to accept           │
          └──────────────────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Backend LLM executes    │
          │  the directive which     │
          │  knows MCP tool calls    │
          └──────────────────────────┘
```

**Three primitives. Directive predictions.** FunctionGemma knows when to call `search()`, `load()`, or `execute(directive)`. The directive itself contains MCP tool knowledge.

### Predictive Context Loading: Bypassing Search Entirely

Here's the breakthrough: **The Kiwi router can pre-load likely directives into its context** based on conversation flow:

```
Traditional:
  User: [TOOL: create email validator]
      ↓
  Search 10,000 directives (15ms)
      ↓
  Load top 7 into router (5ms)
      ↓
  Route (45ms)
  Total: 65ms

With predictive loading:
  Conversation ongoing about "email" and "validation"
      ↓
  Router ALREADY HAS email_*, validate_* directives in context
      ↓
  User: [TOOL: create email validator]
      ↓
  Route immediately (45ms) - search bypassed!
  Total: 45ms (30% faster!)
```

**The router updates its own context dynamically as the conversation flows.**

### Implementation

```python
class ContinuousPredictionRouter:
    """Single FunctionGemma router running continuously"""

    def __init__(self):
        # One unified router - 270M (~500MB VRAM)
        self.router = FunctionGemmaRouter("universal-router")

        # Current prediction (updated on every keystroke)
        self.current_prediction = None
        self.confidence = 0.0
        self.last_text = ""

        # Predictive context management
        self.discovery = DirectiveDiscoveryLayer()
        self.context_cache = {
            "directives": [],  # Directives currently loaded in router's context
            "confidence": 0.0
        }

    async def update_predictive_context(self, conversation_history: List[str]):
        """
        Update router context based on conversation signals.
        Runs in background, preparing context BEFORE tool calls.
        """

        # Extract signals from recent conversation
        signals = self.extract_conversation_signals(conversation_history[-5:])

        # Predict likely directives
        if signals["confidence"] > 0.7:
            # High confidence - pre-load directives
            likely_directives = await self.discovery.discover_directives(
                intent=signals["predicted_query"],
                top_k=7
            )

            # Update router's context with predicted directives
            self.context_cache["directives"] = likely_directives
            self.context_cache["confidence"] = signals["confidence"]

            print(f"🔥 Context warmed: {[d.name for d in likely_directives]} ({signals['confidence']:.0%})")

    def extract_conversation_signals(self, recent_turns: List[str]) -> Dict:
        """
        Extract signals that predict directive needs.

        Examples:
        - User mentions "email" → likely email_* directives
        - User says "deploy" → likely deploy_* directives
        - User talks about "testing" → likely test_* directives
        """

        keywords = []
        action_verbs = []
        domain = None

        for turn in recent_turns:
            # Extract keywords
            keywords.extend(self.extract_keywords(turn))

            # Extract action verbs
            action_verbs.extend(self.extract_actions(turn))

            # Classify domain
            domain_guess = self.classify_domain_fast(turn)
            if domain_guess:
                domain = domain_guess

        # Build predicted query
        predicted_query = " ".join(action_verbs + keywords[:3])

        # Calculate confidence
        confidence = min(
            len(keywords) * 0.2 +
            len(action_verbs) * 0.3 +
            (0.3 if domain else 0.0),
            1.0
        )

        return {
            "predicted_query": predicted_query,
            "confidence": confidence,
            "domain": domain
        }

    async def on_keystroke(self, current_text: str):
        """Called on EVERY keystroke - routers predict continuously"""

        # Skip if text hasn't changed
        if current_text == self.last_text:
            return self.predictions

        self.last_text = current_text

        # Update predictive context in background
        asyncio.create_task(
            self.update_predictive_context(self.conversation_history)
        )

        # Use cached context if available (warm/hot context)
        cached_directives = self.context_cache.get("directives", None)

        # Get prediction from router
        prediction = await self.router.predict(
            current_text,
            cached_context=cached_directives  # Use warm context!
        )

        # Store current prediction
        if prediction.confidence > 0.6:
            self.current_prediction = prediction
            self.confidence = prediction.confidence
        else:
            self.current_prediction = None
            self.confidence = 0.0

        return self.format_autocomplete_ui(prediction)

    def on_tab(self):
        """User pressed Tab - accept current prediction"""

        if not self.current_prediction:
            return None

        # Execute INSTANTLY - no waiting!
        return self.execute_immediately(self.current_prediction)

    def on_enter(self):
        """User pressed Enter - execute or consult orchestrator"""

        if not self.current_prediction:
            # No predictions, use orchestrator
            return self.consult_orchestrator(self.last_text)

        if self.confidence > 0.85:
            # High confidence - instant execution
            return self.execute_immediately(self.current_prediction)
        else:
            # Medium confidence - show options or consult orchestrator
            return self.show_confirmation_or_orchestrate(self.current_prediction)

    async def execute_immediately(self, prediction):
        """Execute tool call without any delay"""

        start = time.time()
        result = await self.kiwi_mcp.execute(prediction.tool_call)
        elapsed = (time.time() - start) * 1000

        return {
            "result": result,
            "metadata": {
                "router": prediction.router_name,
                "confidence": prediction.confidence,
                "execution_time_ms": elapsed,
                "total_latency_ms": prediction.prediction_time_ms + elapsed
            }
        }
```

### Performance: Search Primitive Bypassed

With predictive context loading, the Kiwi router can skip the search primitive entirely:

| Scenario                           | Search | Load | Route | Total | Speedup        |
| ---------------------------------- | ------ | ---- | ----- | ----- | -------------- |
| **Cold start**                     | 15ms   | 5ms  | 45ms  | 65ms  | Baseline       |
| **Warm context (predicted)**       | 0ms    | 0ms  | 45ms  | 45ms  | **30% faster** |
| **Hot context (focused workflow)** | 0ms    | 0ms  | 40ms  | 40ms  | **38% faster** |

**In sustained workflows (email feature, deployment pipeline, testing session):**

- Context stays warm 70-85% of the time
- Average latency: ~48ms (vs 65ms cold)
- User experience: Feels instant, psychic predictions

**The primitives adapt:**

```
Cold:  search → load → execute
Warm:  -------- load → execute  (search predicted)
Hot:   -------- ---- → execute  (load predicted)
```

**The search and load primitives become speculative operations that happen BEFORE the user needs them.**

### The UX Vision

```
┌─────────────────────────────────────────────────────────────────┐
│ 🥝 Kiwi Terminal                                     [edge: 2ms]│
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ > find email scr█                                               │
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐     │
│   │ 🔍 search(script, "email")         [Tab]  95%         │     │
│   │ 📂 ls ~/scripts/*email*            [2]    72%         │     │
│   └───────────────────────────────────────────────────────┘     │
│                                                                 │
│ Previous:                                                       │
│ > sync directives                                               │
│ ✓ Synced 5 directives (45ms)                                    │
│                                                                 │
│ > run email_enricher                                            │
│ ✓ Enriched 247 contacts (1.2s)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Keyboard shortcuts:
  Tab     → Accept top prediction, execute immediately
  1-9     → Execute specific prediction by number
  Enter   → Execute top prediction OR consult orchestrator if low confidence
  Ctrl+E  → Force orchestrator (for complex queries)
```

### Why This Works: The Math

| Factor                  | Value                                        | Impact                                 |
| ----------------------- | -------------------------------------------- | -------------------------------------- |
| FunctionGemma inference | 40-80ms                                      | Router finishes before you stop typing |
| Human typing speed      | ~200ms between words                         | Model predicts during natural pauses   |
| Human reaction time     | 150-250ms                                    | You can't even perceive the latency    |
| Multiple routers (4x)   | +20-30ms overhead                            | Still faster than your next keystroke  |
| **Result**              | **Predictions ready BEFORE you press Enter** | **Feels like autocomplete magic**      |

### Resource Requirements

```
Universal Router (FunctionGemma 270M):
- Memory: ~500MB VRAM/RAM
- Latency: 40-80ms (cold), 25-45ms (warm/hot context)
- Cost: $0 (local)
- CPU: 4 cores recommended
- GPU: Optional (CUDA/Metal for faster inference)

Hardware Examples:
- Desktop GPU (RTX 3060+): 40-60ms latency
- MacBook Pro M1+ (8GB RAM): 50-80ms latency
- iPhone 16 Pro (8GB RAM): 60-120ms on Neural Engine
- Battery impact: <0.5W continuous operation
```

### Router Training Examples: Three Primitives Only

FunctionGemma is trained ONLY on the 3 primitives + directive predictions:

```python
# Training data for FunctionGemma router
ROUTER_TRAINING_EXAMPLES = [
    # Primitive: SEARCH (when directive unknown or not in context)
    {"input": "find email scripts", "name": "search", "arguments": {"item_type": "script", "query": "email"}},
    {"input": "search for validators", "name": "search", "arguments": {"item_type": "directive", "query": "validator"}},
    {"input": "look up git workflows", "name": "search", "arguments": {"item_type": "directive", "query": "git workflow"}},
    
    # Primitive: LOAD (when directive found but needs schema)
    {"input": "load email validator", "name": "load", "arguments": {"item_type": "directive", "item_id": "email_validator", "source": "project"}},
    {"input": "show me the sync directive", "name": "load", "arguments": {"item_type": "directive", "item_id": "sync_directives", "source": "project"}},
    
    # Primitive: EXECUTE (when directive is in context - warm/hot)
    {"input": "run sync", "name": "execute", "arguments": {"item_type": "directive", "item_id": "sync_directives", "action": "run"}},
    {"input": "validate these emails", "name": "execute", "arguments": {"item_type": "directive", "item_id": "email_validator", "action": "run"}},
    {"input": "commit changes", "name": "execute", "arguments": {"item_type": "directive", "item_id": "git_workflow", "action": "run"}},
    {"input": "deploy to production", "name": "execute", "arguments": {"item_type": "directive", "item_id": "deploy_prod", "action": "run"}},
]
```

**Critical distinction**: 
- FunctionGemma predicts: `search()`, `load()`, or `execute(directive_name)`
- The **directive itself** (executed by backend LLM) knows how to call `git_commit()`, `file_open()`, etc.
- FunctionGemma has NO knowledge of MCP tool schemas—only directives in its predictive context

### The Orchestrator's Role with Continuous Prediction

The fine-tuned orchestrator becomes the **fallback + explainer**:

```python
class HybridPredictionSystem:
    def __init__(self):
        self.router = ContinuousPredictionRouter()
        self.orchestrator = FineTunedOrchestrator()

    async def process_input(self, text: str, current_prediction):
        """Process with both fast and slow paths"""

        if current_prediction and current_prediction.confidence > 0.90:
            # FAST PATH: Router is very confident
            # Execute immediately, orchestrator just explains
            result = await self.execute(current_prediction.tool_call)

            # Orchestrator synthesizes in background (doesn't block)
            asyncio.create_task(
                self.orchestrator.explain_in_background(text, result)
            )

            return result  # User sees result INSTANTLY

        elif current_prediction and current_prediction.confidence > 0.70:
            # MEDIUM PATH: Router suggests, orchestrator confirms
            confirmation = await self.orchestrator.quick_confirm(
                text, current_prediction.tool_call
            )

            if confirmation.approved:
                return await self.execute(current_prediction.tool_call)
            else:
                return await self.orchestrator.full_process(text)

        else:
            # SLOW PATH: Routers uncertain, orchestrator decides
            return await self.orchestrator.full_process(text)
```

### Training the Orchestrator for Always-Hot Mode

The orchestrator needs to learn to work with constantly-updating router predictions:

```python
ALWAYS_HOT_TRAINING_EXAMPLES = [
    {
        "scenario": "Router finished before user pressed Enter",
        "router_predictions": [
            {"name": "search", "confidence": 0.95, "timestamp_ms": 120}
        ],
        "user_action": "pressed_enter",
        "orchestrator_behavior": "acknowledge_and_execute",
        "response": "Executing search... [results in 45ms]"
    },

    {
        "scenario": "Router predicts complex directive execution",
        "router_prediction": {
            "name": "execute", 
            "arguments": {"item_type": "directive", "item_id": "sync_and_commit", "action": "run"},
            "confidence": 0.89
        },
        "user_action": "pressed_enter",
        "orchestrator_behavior": "execute_directive_via_backend_llm",
        "response": "Executing sync_and_commit directive...\n✅ Directives synced\n✅ Changes committed\nDone!"
    },

    {
        "scenario": "Router confidence increased mid-typing",
        "router_predictions": [
            {"name": "search", "confidence": 0.45, "timestamp_ms": 50},
            {"name": "search", "confidence": 0.78, "timestamp_ms": 150},
            {"name": "search", "confidence": 0.91, "timestamp_ms": 250}
        ],
        "orchestrator_behavior": "trust_final_prediction",
        "response": "Router locked in at 91% confidence - executing search."
    }
]
```

### The Ultimate Lilux Experience

```
Traditional AI:
  You type → You wait → AI thinks → AI responds → Tool executes → Result
  Total: 3-5 seconds

Lilux (Always-Hot):
  You type ← AI predicting ← AI predicted ← Tool ready ← [Tab] ← INSTANT
  Total: 0.05 seconds (just the keystroke + tool execution)
```

This is **predictive AI that's faster than thought**. The system anticipates your intent as you type, so by the time you commit to the action (Tab/Enter), everything is already computed and ready to execute.

**This is the interface of the future: AI that waits for you, not the other way around.**

---

## Next Steps

1. **Start with router** - Fine-tune FunctionGemma first (simpler, faster)
2. **Collect data** - Log real interactions for orchestrator training
3. **Distill from Claude** - Generate orchestrator training data
4. **Fine-tune orchestrator** - Llama 3.3 70B with LoRA
5. **Deploy hybrid** - Router on edge, orchestrator in cloud or self-hosted
6. **Iterate** - Collect feedback, retrain both models

---

_Document generated: 2026-01-17_
_Part of the Kiwi Fine-Tune documentation series_
_Related: Dual-Model Architecture, FunctionGemma Training, Integration Patterns_
