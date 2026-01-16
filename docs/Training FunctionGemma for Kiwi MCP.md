# Training FunctionGemma for Kiwi MCP

## Overview

This guide walks through fine-tuning Google's FunctionGemma 270M model to become a specialized router for Kiwi MCP tool calls.

## Prerequisites

### Hardware Requirements

**Minimum**:

- GPU with 8GB VRAM (e.g., RTX 3060)
- 16GB system RAM
- 20GB free disk space

**Recommended**:

- GPU with 16GB+ VRAM (e.g., RTX 4080, A6000)
- 32GB system RAM
- 50GB free disk space (SSD)

**Training time**:

- Minimum setup: ~4 hours
- Recommended setup: ~1.5 hours

### Software Requirements

```bash
# Python 3.10+
python --version

# Install training framework (choose one)
pip install unsloth  # Recommended - optimized for small models
# OR
pip install axolotl  # Alternative - more features

# Install dependencies
pip install transformers accelerate datasets peft bitsandbytes
```

## Step 1: Generate Training Data

### 1.1 Extract Patterns from agent.md

The current `agent.md` contains our dispatch table. We'll convert this into training examples:

```python
# scripts/generate_training_data.py

import json
from typing import List, Dict
from itertools import product

# Extract from your agent.md dispatch table
COMMAND_PATTERNS = {
    "search directives {X}": {
        "name": "search",
        "arguments": {
            "item_type": "directive",
            "query": "{X}",
            "source": "local"
        }
    },
    "search scripts {X}": {
        "name": "search",
        "arguments": {
            "item_type": "script",
            "query": "{X}",
            "source": "local"
        }
    },
    "search knowledge {X}": {
        "name": "search",
        "arguments": {
            "item_type": "knowledge",
            "query": "{X}",
            "source": "local"
        }
    },
    "run directive {X}": {
        "name": "execute",
        "arguments": {
            "item_type": "directive",
            "action": "run",
            "item_id": "{X}"
        }
    },
    "run script {X}": {
        "name": "execute",
        "arguments": {
            "item_type": "script",
            "action": "run",
            "item_id": "{X}"
        }
    },
    "load directive {X}": {
        "name": "load",
        "arguments": {
            "item_type": "directive",
            "item_id": "{X}",
            "source": "project"
        }
    },
    "create directive {X}": {
        "name": "execute",
        "arguments": {
            "item_type": "directive",
            "action": "create",
            "item_id": "{X}"
        }
    },
    "create script {X}": {
        "name": "execute",
        "arguments": {
            "item_type": "script",
            "action": "create",
            "item_id": "{X}"
        }
    },
    "sync directives": {
        "name": "execute",
        "arguments": {
            "item_type": "directive",
            "action": "run",
            "item_id": "sync_directives"
        }
    },
    "bootstrap {X}": {
        "name": "execute",
        "arguments": {
            "item_type": "directive",
            "action": "run",
            "item_id": "bootstrap",
            "parameters": {"project_type": "{X}"}
        }
    }
}

# Natural language variations for each pattern
PHRASINGS = {
    "search": [
        "search {type} {query}",
        "find {type} {query}",
        "look for {type} {query}",
        "show me {type} {query}",
        "search for {query} {type}",
        "find {query} in {type}",
        "lookup {query} {type}"
    ],
    "run": [
        "run {item}",
        "execute {item}",
        "run the {item}",
        "execute the {item}",
        "use {item}",
        "trigger {item}"
    ],
    "load": [
        "load {item}",
        "show {item}",
        "display {item}",
        "get {item}",
        "read {item}",
        "inspect {item}"
    ],
    "create": [
        "create {type} {name}",
        "new {type} {name}",
        "make {type} called {name}",
        "add {type} {name}",
        "create a {type} named {name}"
    ]
}

# Common query topics
QUERY_TOPICS = [
    "email", "csv", "api", "testing", "deployment", "database",
    "authentication", "validation", "parsing", "enrichment",
    "sync", "backup", "migration", "cleanup", "reporting"
]

# Common item names
ITEM_NAMES = [
    "create_script", "sync_directives", "test_api", "deploy_app",
    "email_enricher", "csv_parser", "data_validator", "backup_tool"
]

def generate_training_examples() -> List[Dict]:
    """Generate comprehensive training dataset"""
    examples = []
    project_path = "/home/user/project"  # Default

    # 1. Generate search variations
    for query in QUERY_TOPICS:
        for item_type in ["directive", "script", "knowledge"]:
            for phrasing in PHRASINGS["search"]:
                user_input = phrasing.format(type=item_type, query=query)
                examples.append({
                    "user": user_input,
                    "context": {"project_path": project_path},
                    "function_call": {
                        "name": "search",
                        "arguments": {
                            "item_type": item_type,
                            "query": query,
                            "source": "local",
                            "project_path": project_path
                        }
                    }
                })

    # 2. Generate run variations
    for item_name in ITEM_NAMES:
        for phrasing in PHRASINGS["run"]:
            user_input = phrasing.format(item=item_name)

            # Infer item type from name
            if "directive" in item_name or item_name in ["sync_directives", "bootstrap"]:
                item_type = "directive"
            else:
                item_type = "script"

            examples.append({
                "user": user_input,
                "context": {"project_path": project_path},
                "function_call": {
                    "name": "execute",
                    "arguments": {
                        "item_type": item_type,
                        "action": "run",
                        "item_id": item_name,
                        "project_path": project_path
                    }
                }
            })

    # 3. Generate create variations
    for name in ["my_script", "test_tool", "new_parser", "api_client"]:
        for item_type in ["directive", "script"]:
            for phrasing in PHRASINGS["create"]:
                user_input = phrasing.format(type=item_type, name=name)
                examples.append({
                    "user": user_input,
                    "context": {"project_path": project_path},
                    "function_call": {
                        "name": "execute",
                        "arguments": {
                            "item_type": item_type,
                            "action": "create",
                            "item_id": name,
                            "project_path": project_path
                        }
                    }
                })

    # 4. Add modifiers (to user, to project, from registry)
    modifier_examples = []
    for ex in examples[:50]:  # Add modifiers to subset
        if ex["function_call"]["name"] == "load":
            # "load directive X from registry"
            modified = ex.copy()
            modified["user"] = ex["user"] + " from registry"
            modified["function_call"]["arguments"]["source"] = "registry"
            modifier_examples.append(modified)

            # "load directive X to project"
            modified = ex.copy()
            modified["user"] = ex["user"] + " to project"
            modified["function_call"]["arguments"]["destination"] = "project"
            modifier_examples.append(modified)

    examples.extend(modifier_examples)

    return examples

def convert_to_training_format(examples: List[Dict]) -> List[Dict]:
    """Convert to format expected by Unsloth/Axolotl"""
    formatted = []

    for ex in examples:
        formatted.append({
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a Kiwi MCP function router. Current project: {ex['context']['project_path']}"
                },
                {
                    "role": "user",
                    "content": ex["user"]
                },
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": ex["function_call"]
                }
            ]
        })

    return formatted

if __name__ == "__main__":
    # Generate examples
    print("Generating training examples...")
    examples = generate_training_examples()
    print(f"Generated {len(examples)} examples")

    # Convert format
    formatted = convert_to_training_format(examples)

    # Save
    with open("kiwi_training_data.jsonl", "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    print(f"Saved training data to kiwi_training_data.jsonl")

    # Show sample
    print("\nSample training example:")
    print(json.dumps(formatted[0], indent=2))
```

### 1.2 Generate Synthetic Variations with GPT-4o

```python
# scripts/augment_training_data.py

import asyncio
from anthropic import Anthropic
import json

client = Anthropic()

async def generate_variations(base_example: Dict, num_variations: int = 5) -> List[Dict]:
    """Use Claude to generate natural variations"""

    prompt = f"""Generate {num_variations} natural language variations of this command:

Original: "{base_example['user']}"

Requirements:
- Same intent and tool call
- Different phrasing (casual, formal, abbreviated)
- Include typos/shortcuts (e.g., "sync dirs", "find email stuff")
- Natural speech patterns

Return JSON array of strings only."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    variations = json.loads(response.content[0].text)

    # Create new examples with same function call
    return [
        {
            **base_example,
            "user": variation
        }
        for variation in variations
    ]

async def augment_dataset(input_file: str, output_file: str):
    """Augment dataset with variations"""
    with open(input_file) as f:
        examples = [json.loads(line) for line in f]

    augmented = []
    for ex in examples[:100]:  # Augment subset
        variations = await generate_variations(ex, num_variations=3)
        augmented.extend(variations)

    # Combine original + augmented
    all_examples = examples + augmented

    with open(output_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Augmented dataset: {len(examples)} → {len(all_examples)} examples")

if __name__ == "__main__":
    asyncio.run(augment_dataset(
        "kiwi_training_data.jsonl",
        "kiwi_training_data_augmented.jsonl"
    ))
```

## Step 2: Fine-Tune with Unsloth

### 2.1 Training Script

```python
# scripts/train_functiongemma.py

from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

# Configuration
MODEL_NAME = "google/gemma-2-2b-function-calling"  # Base FunctionGemma
OUTPUT_NAME = "kiwi-function-router"
MAX_SEQ_LENGTH = 512
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Use 4-bit quantization for training
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing=True,
)

# Load dataset
dataset = load_dataset("json", data_files="kiwi_training_data_augmented.jsonl")
train_dataset = dataset["train"].train_test_split(test_size=0.1)

# Format function for training
def format_for_training(examples):
    texts = []
    for messages in examples["messages"]:
        # Convert to FunctionGemma format
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(format_for_training, batched=True)

# Training arguments
from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset["train"],
    eval_dataset=train_dataset["test"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir=f"./checkpoints/{OUTPUT_NAME}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
    ),
)

# Train!
print("Starting training...")
trainer.train()

# Save
print(f"Saving model to {OUTPUT_NAME}...")
model.save_pretrained(OUTPUT_NAME)
tokenizer.save_pretrained(OUTPUT_NAME)

print("Training complete!")
```

### 2.2 Run Training

```bash
# Generate training data
python scripts/generate_training_data.py

# Optionally augment with variations
python scripts/augment_training_data.py

# Train model (will take 1-4 hours depending on GPU)
python scripts/train_functiongemma.py
```

## Step 3: Evaluate Performance

### 3.1 Test Inference

```python
# scripts/test_router.py

from unsloth import FastLanguageModel
import json

# Load trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="kiwi-function-router",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)  # Enable inference mode

def predict_tool_call(query: str, project_path: str = "/home/user/project"):
    """Predict tool call for a query"""

    messages = [
        {
            "role": "system",
            "content": f"You are a Kiwi MCP function router. Current project: {project_path}"
        },
        {
            "role": "user",
            "content": query
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.1,  # Low temperature for consistency
        do_sample=False,  # Greedy decoding
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse function call from response
    # (Format depends on FunctionGemma's output structure)
    return parse_function_call(response)

# Test cases
test_queries = [
    "find email scripts",
    "run sync_directives",
    "create a new script called csv_parser",
    "search for api testing directives",
    "load the email_enricher script",
]

print("Testing router predictions:\n")
for query in test_queries:
    result = predict_tool_call(query)
    print(f"Query: {query}")
    print(f"Prediction: {json.dumps(result, indent=2)}\n")
```

### 3.2 Benchmark Accuracy

```python
# scripts/benchmark_accuracy.py

from typing import List, Dict
import json
from collections import defaultdict

def load_test_set(file_path: str) -> List[Dict]:
    """Load test examples"""
    with open(file_path) as f:
        return [json.loads(line) for line in f]

def evaluate_accuracy(model, test_set: List[Dict]) -> Dict:
    """Evaluate model accuracy on test set"""

    results = {
        "total": len(test_set),
        "correct": 0,
        "tool_name_correct": 0,
        "params_correct": 0,
        "errors": []
    }

    for example in test_set:
        query = example["messages"][1]["content"]
        expected = example["messages"][2]["function_call"]

        try:
            predicted = predict_tool_call(query)

            # Check tool name
            if predicted["name"] == expected["name"]:
                results["tool_name_correct"] += 1

                # Check parameters
                if predicted["arguments"] == expected["arguments"]:
                    results["correct"] += 1
                    results["params_correct"] += 1
                else:
                    results["errors"].append({
                        "query": query,
                        "expected": expected,
                        "predicted": predicted,
                        "error": "parameter_mismatch"
                    })
            else:
                results["errors"].append({
                    "query": query,
                    "expected": expected,
                    "predicted": predicted,
                    "error": "tool_mismatch"
                })
        except Exception as e:
            results["errors"].append({
                "query": query,
                "expected": expected,
                "error": str(e)
            })

    results["accuracy"] = results["correct"] / results["total"]
    results["tool_accuracy"] = results["tool_name_correct"] / results["total"]
    results["param_accuracy"] = results["params_correct"] / results["total"]

    return results

# Run benchmark
test_set = load_test_set("kiwi_test_set.jsonl")
results = evaluate_accuracy(model, test_set)

print("Benchmark Results:")
print(f"Overall Accuracy: {results['accuracy']:.2%}")
print(f"Tool Name Accuracy: {results['tool_accuracy']:.2%}")
print(f"Parameter Accuracy: {results['param_accuracy']:.2%}")
print(f"\nErrors: {len(results['errors'])}")

# Show errors
if results['errors']:
    print("\nSample errors:")
    for error in results['errors'][:5]:
        print(json.dumps(error, indent=2))
```

### 3.3 Benchmark Speed

```python
# scripts/benchmark_speed.py

import time
import numpy as np

def benchmark_inference_speed(model, num_runs: int = 100):
    """Benchmark inference latency"""

    test_queries = [
        "find email scripts",
        "run sync_directives",
        "create script my_parser",
        "search api testing directives",
        "load email_enricher from registry"
    ]

    latencies = []

    print(f"Running {num_runs} inference calls...")

    for i in range(num_runs):
        query = test_queries[i % len(test_queries)]

        start = time.perf_counter()
        _ = predict_tool_call(query)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    # Statistics
    results = {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": np.min(latencies),
        "max": np.max(latencies)
    }

    print("\nLatency Statistics (ms):")
    for key, value in results.items():
        print(f"  {key.upper()}: {value:.2f}ms")

    return results

# Run benchmark
speed_results = benchmark_inference_speed(model, num_runs=100)
```

## Step 4: Export for Deployment

### 4.1 Convert to GGUF (for llama.cpp)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert model
python convert.py ../kiwi-function-router \
  --outfile ../kiwi-router.gguf \
  --outtype q8_0  # 8-bit quantization
```

### 4.2 Convert to CoreML (for iOS/macOS)

```python
# scripts/convert_to_coreml.py

import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("kiwi-function-router")
tokenizer = AutoTokenizer.from_pretrained("kiwi-function-router")

# Convert to CoreML
mlmodel = ct.convert(
    model,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine
)

# Save
mlmodel.save("KiwiFunctionRouter.mlpackage")
print("CoreML model saved!")
```

### 4.3 Convert to ONNX (cross-platform)

```python
# scripts/convert_to_onnx.py

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# Load and convert
model = ORTModelForCausalLM.from_pretrained(
    "kiwi-function-router",
    export=True
)

tokenizer = AutoTokenizer.from_pretrained("kiwi-function-router")

# Save
model.save_pretrained("kiwi-router-onnx")
tokenizer.save_pretrained("kiwi-router-onnx")

print("ONNX model saved!")
```

## Expected Results

After training on ~1,000-2,000 examples, you should achieve:

- **Accuracy**: 96-99% on test set
- **Inference speed**: 40-80ms on typical hardware
- **Model size**: ~600MB (quantized)
- **Memory usage**: 1-2GB RAM during inference

## Troubleshooting

### Low Accuracy (<95%)

- **Solution**: Generate more training examples, especially for edge cases
- Add more natural language variations
- Increase training epochs (try 5-10)

### Slow Inference (>150ms)

- **Solution**: Use smaller quantization (4-bit or 8-bit)
- Ensure model is in inference mode: `FastLanguageModel.for_inference(model)`
- Use GPU/NPU if available

### High Memory Usage (>4GB)

- **Solution**: Use 4-bit quantization during inference
- Reduce `max_seq_length` to 256 or 384
- Use GGUF format with llama.cpp for CPU inference

## Next Steps

- [Streaming Architecture](./Streaming%20Architecture%20%26%20Concurrent%20Execution.md) - Handle concurrent predictions
- [Deployment Guide](./Deployment%20Guide%20-%20Edge%20Device%20Implementation.md) - Deploy to edge devices
- [Integration Patterns](./Integration%20Patterns%20-%20Connecting%20All%20Components.md) - Integrate with reasoning model

---

**Estimated Training Time**: 1-4 hours  
**Estimated Training Cost**: $5-50 (depending on GPU rental)  
**Model Size**: ~600MB quantized  
**Target Accuracy**: 98%+
