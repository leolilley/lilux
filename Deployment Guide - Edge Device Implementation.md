# Deployment Guide - Edge Device Implementation

## Overview

This guide covers deploying FunctionGemma-based routers to edge devices (phones, laptops, embedded systems) for fast, private, offline-capable tool routing.

## Platform Support Matrix

| Platform    | Runtime           | GPU/NPU       | Latency   | Setup Difficulty |
| ----------- | ----------------- | ------------- | --------- | ---------------- |
| **macOS**   | llama.cpp / Metal | Metal (M1+)   | 30-50ms   | Easy             |
| **iOS**     | CoreML            | Neural Engine | 25-40ms   | Medium           |
| **Windows** | ONNX / DirectML   | GPU           | 40-80ms   | Medium           |
| **Linux**   | llama.cpp / CUDA  | CUDA          | 30-60ms   | Easy             |
| **Android** | TFLite / NNAPI    | GPU/NPU       | 50-100ms  | Hard             |
| **Web**     | ONNX / WebGPU     | WebGPU        | 100-200ms | Medium           |

## Deployment Option 1: macOS (Metal) 🍎

### Why macOS First?

- **Excellent hardware**: M1/M2/M3 chips have powerful Neural Engines
- **Easy setup**: Metal acceleration works out of the box
- **Great for development**: Most developers use Macs

### Setup with llama.cpp

```bash
# 1. Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_METAL=1  # Enable Metal acceleration

# 2. Convert your model to GGUF (if not already done)
python convert.py /path/to/kiwi-function-router \
  --outfile kiwi-router-q8_0.gguf \
  --outtype q8_0

# 3. Test inference
./main \
  -m kiwi-router-q8_0.gguf \
  -p "find email scripts" \
  -n 128 \
  --temp 0.1 \
  -ngl 99  # Offload all layers to GPU
```

### Python Integration

```python
# router_llama.py

from llama_cpp import Llama
import json
from typing import Dict, Optional

class LlamaCppRouter:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,          # Context window
            n_threads=8,        # CPU threads
            n_gpu_layers=99,    # Use Metal
            verbose=False
        )

    def predict(self, query: str, project_path: str) -> Dict:
        """Predict tool call for query"""

        prompt = f"""<system>You are a Kiwi MCP function router. Current project: {project_path}</system>
<user>{query}</user>
<assistant>"""

        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["</assistant>"],
            echo=False
        )

        # Parse function call from response
        text = response["choices"][0]["text"]
        return self._parse_function_call(text)

    async def stream_predict(self, query: str, project_path: str):
        """Stream predictions with early exit"""

        prompt = f"""<system>You are a Kiwi MCP function router. Current project: {project_path}</system>
<user>{query}</user>
<assistant>"""

        partial = ""

        for chunk in self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stream=True
        ):
            token = chunk["choices"][0]["text"]
            partial += token

            # Try parsing partial JSON
            try:
                tool_call = self._parse_partial(partial)
                if tool_call and tool_call["is_complete"]:
                    yield tool_call
                    return  # Early exit!
            except:
                continue

    def _parse_function_call(self, text: str) -> Dict:
        """Parse function call from model output"""
        # Assuming format: search|{"item_type":"script","query":"email"}
        if "|" in text:
            tool_name, args_json = text.split("|", 1)
            return {
                "name": tool_name.strip(),
                "arguments": json.loads(args_json)
            }
        else:
            # Fallback: try parsing as pure JSON
            return json.loads(text)

# Usage
router = LlamaCppRouter("kiwi-router-q8_0.gguf")
result = router.predict("find email scripts", "/Users/me/project")
print(result)
```

### Performance Tuning

```python
# Optimize for latency
router = LlamaCppRouter(
    model_path="kiwi-router-q8_0.gguf",
    n_ctx=256,          # Smaller context = faster
    n_threads=4,        # Fewer threads = less contention
    n_gpu_layers=99,    # Full GPU
    n_batch=8           # Smaller batch for responsiveness
)

# Optimize for throughput
router = LlamaCppRouter(
    model_path="kiwi-router-q8_0.gguf",
    n_ctx=512,
    n_threads=8,
    n_gpu_layers=99,
    n_batch=512         # Large batch for throughput
)
```

### Expected Performance (M1 Mac)

```
Model: kiwi-router-q8_0.gguf (270M params, 8-bit)
Hardware: M1 Pro, 16GB RAM

First token: 28ms
Subsequent tokens: 3-4ms per token
Total (avg 40 tokens): 28 + (40 × 3.5) = 168ms

With early exit (20 tokens): 28 + (20 × 3.5) = 98ms
```

## Deployment Option 2: iOS (CoreML) 📱

### Why iOS?

- **Neural Engine**: Dedicated AI hardware on A15+ chips
- **Energy efficient**: <1W power consumption
- **Always available**: Run in background

### Setup

```bash
# 1. Convert to CoreML format
python scripts/convert_to_coreml.py

# This creates: KiwiFunctionRouter.mlpackage
```

### Swift Integration

```swift
// KiwiRouter.swift

import CoreML
import NaturalLanguage

class KiwiRouter {
    private let model: KiwiFunctionRouter
    private let tokenizer: Tokenizer

    init() throws {
        // Load CoreML model
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine

        self.model = try KiwiFunctionRouter(configuration: config)
        self.tokenizer = Tokenizer()
    }

    func predict(query: String, projectPath: String) async throws -> ToolCall {
        // Tokenize input
        let tokens = tokenizer.encode(
            "System: Kiwi MCP router. Project: \(projectPath)\nUser: \(query)\nAssistant:"
        )

        // Create input
        let input = KiwiFunctionRouterInput(
            input_ids: tokens,
            attention_mask: Array(repeating: 1, count: tokens.count)
        )

        // Run inference
        let output = try await model.prediction(input: input)

        // Decode output
        let responseTokens = output.logits.argmax(dim: -1)
        let responseText = tokenizer.decode(responseTokens)

        // Parse function call
        return try parseFunctionCall(responseText)
    }

    func streamPredict(query: String, projectPath: String) -> AsyncStream<PartialToolCall> {
        AsyncStream { continuation in
            Task {
                // CoreML doesn't support streaming natively
                // So we run full inference and emit early if possible

                let result = try await self.predict(
                    query: query,
                    projectPath: projectPath
                )

                continuation.yield(.complete(result))
                continuation.finish()
            }
        }
    }
}

// Usage in SwiftUI
struct KiwiAgentView: View {
    @StateObject private var router = KiwiRouter()
    @State private var query = ""
    @State private var result: ToolCall?

    var body: some View {
        VStack {
            TextField("Enter query", text: $query)

            Button("Route") {
                Task {
                    result = try await router.predict(
                        query: query,
                        projectPath: "/Users/me/project"
                    )
                }
            }

            if let result = result {
                Text("Tool: \(result.name)")
                Text("Args: \(result.arguments)")
            }
        }
    }
}
```

### Expected Performance (iPhone 15 Pro)

```
Model: KiwiFunctionRouter.mlpackage (270M params)
Hardware: A17 Pro Neural Engine

Cold start: 150ms (first inference)
Warm inference: 22-35ms
Power draw: 0.3-0.5W
Battery impact: ~2-3%/hour continuous use
```

## Deployment Option 3: Linux (CUDA) 🐧

### Setup with ONNX Runtime

```bash
# 1. Install ONNX Runtime with CUDA
pip install onnxruntime-gpu

# 2. Convert model to ONNX (if not done)
python scripts/convert_to_onnx.py

# 3. Test
python scripts/test_onnx_router.py
```

### Python Integration

```python
# router_onnx.py

import onnxruntime as ort
import numpy as np
from typing import Dict

class ONNXRouter:
    def __init__(self, model_path: str):
        # Configure for CUDA
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'DEFAULT',
            }),
            'CPUExecutionProvider'
        ]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        # Warm up
        self._warmup()

    def _warmup(self):
        """Warm up model with dummy input"""
        dummy_input = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        }
        self.session.run(None, dummy_input)

    def predict(self, query: str, project_path: str) -> Dict:
        """Predict tool call"""

        # Tokenize
        input_ids, attention_mask = self.tokenize(query, project_path)

        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

        # Decode
        logits = outputs[0]
        tokens = np.argmax(logits, axis=-1)
        text = self.detokenize(tokens)

        return self.parse_function_call(text)

    def tokenize(self, query: str, project_path: str):
        # Implementation depends on your tokenizer
        pass

    def detokenize(self, tokens):
        # Implementation depends on your tokenizer
        pass

# Usage
router = ONNXRouter("kiwi-router.onnx")
result = router.predict("find email scripts", "/home/user/project")
```

### Expected Performance (RTX 4080)

```
Model: kiwi-router.onnx (270M params, FP16)
Hardware: RTX 4080, 16GB VRAM

First inference: 45ms
Subsequent: 12-18ms
Memory: ~800MB VRAM
Power: 30-50W (shared with other tasks)
```

## Deployment Option 4: Android 🤖

### Setup with TensorFlow Lite

```bash
# 1. Convert to TFLite
python scripts/convert_to_tflite.py

# Output: kiwi-router.tflite
```

### Kotlin Integration

```kotlin
// KiwiRouter.kt

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

class KiwiRouter(private val context: Context) {
    private val interpreter: Interpreter
    private val tokenizer: Tokenizer

    init {
        // Load model
        val modelFile = loadModelFile("kiwi-router.tflite")

        // Configure GPU delegate
        val options = Interpreter.Options()
        options.addDelegate(GpuDelegate())
        options.setNumThreads(4)

        interpreter = Interpreter(modelFile, options)
        tokenizer = Tokenizer(context)
    }

    fun predict(query: String, projectPath: String): ToolCall {
        // Tokenize
        val tokens = tokenizer.encode(
            "System: Kiwi MCP router. Project: $projectPath\n" +
            "User: $query\n" +
            "Assistant:"
        )

        // Prepare input
        val inputBuffer = ByteBuffer.allocateDirect(tokens.size * 4)
            .order(ByteOrder.nativeOrder())

        tokens.forEach { inputBuffer.putInt(it) }

        // Prepare output
        val outputBuffer = ByteBuffer.allocateDirect(256 * 4)
            .order(ByteOrder.nativeOrder())

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Decode output
        val outputTokens = IntArray(256)
        outputBuffer.rewind()
        for (i in outputTokens.indices) {
            outputTokens[i] = outputBuffer.int
        }

        val responseText = tokenizer.decode(outputTokens)

        // Parse function call
        return parseFunctionCall(responseText)
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)
        val fileChannel = inputStream.channel
        val startOffset = 0L
        val declaredLength = inputStream.available().toLong()
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
    }
}

// Usage
val router = KiwiRouter(context)
val result = router.predict("find email scripts", "/storage/emulated/0/project")
```

### Expected Performance (Pixel 8 Pro)

```
Model: kiwi-router.tflite (270M params, INT8)
Hardware: Tensor G3, GPU

First inference: 120ms
Subsequent: 60-80ms
Power: 1-2W
Battery: ~5-8%/hour continuous
```

## Deployment Option 5: Web (ONNX + WebGPU) 🌐

### Setup with Transformers.js

```bash
# No conversion needed - use ONNX model directly
npm install @xenova/transformers
```

### JavaScript Integration

```javascript
// router-web.js

import { pipeline } from "@xenova/transformers";

class WebRouter {
  constructor() {
    this.pipeline = null;
  }

  async init() {
    // Load model (auto-downloads from HuggingFace)
    this.pipeline = await pipeline(
      "text-generation",
      "your-username/kiwi-function-router",
      {
        device: "webgpu", // Use WebGPU if available
        dtype: "fp16",
      },
    );
  }

  async predict(query, projectPath) {
    const prompt = `System: Kiwi MCP router. Project: ${projectPath}
User: ${query}
Assistant:`;

    const result = await this.pipeline(prompt, {
      max_new_tokens: 128,
      temperature: 0.1,
      do_sample: false,
    });

    // Parse function call
    return this.parseFunctionCall(result[0].generated_text);
  }

  parseFunctionCall(text) {
    // Extract tool call from response
    if (text.includes("|")) {
      const [name, argsJson] = text.split("|", 2);
      return {
        name: name.trim(),
        arguments: JSON.parse(argsJson),
      };
    }
    return JSON.parse(text);
  }
}

// Usage
const router = new WebRouter();
await router.init();

const result = await router.predict("find email scripts", "/home/user/project");

console.log(result);
```

### Expected Performance (Chrome on M1 Mac)

```
Model: kiwi-function-router (270M params, FP16)
Runtime: WebGPU

First inference: 250ms (includes download)
Subsequent: 100-150ms
Memory: ~1.2GB
```

## Cross-Platform Wrapper

### Unified API for All Platforms

```python
# kiwi_router/__init__.py

from abc import ABC, abstractmethod
from typing import Dict, Optional
import platform
import sys

class RouterBase(ABC):
    @abstractmethod
    def predict(self, query: str, project_path: str) -> Dict:
        pass

    @abstractmethod
    async def stream_predict(self, query: str, project_path: str):
        pass

class RouterFactory:
    @staticmethod
    def create(
        model_path: str,
        device: Optional[str] = None
    ) -> RouterBase:
        """Create optimal router for current platform"""

        if device:
            return RouterFactory._create_for_device(model_path, device)

        # Auto-detect platform
        system = platform.system()

        if system == "Darwin":
            # macOS - use Metal
            from .routers.llama_cpp import LlamaCppRouter
            return LlamaCppRouter(model_path)

        elif system == "Linux":
            # Linux - try CUDA, fallback to CPU
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    from .routers.onnx import ONNXRouter
                    return ONNXRouter(model_path, device='cuda')
            except:
                pass

            # Fallback to llama.cpp
            from .routers.llama_cpp import LlamaCppRouter
            return LlamaCppRouter(model_path)

        elif system == "Windows":
            # Windows - try DirectML, fallback to CPU
            try:
                import onnxruntime as ort
                if 'DmlExecutionProvider' in ort.get_available_providers():
                    from .routers.onnx import ONNXRouter
                    return ONNXRouter(model_path, device='dml')
            except:
                pass

            from .routers.llama_cpp import LlamaCppRouter
            return LlamaCppRouter(model_path)

        else:
            raise RuntimeError(f"Unsupported platform: {system}")

# Usage
from kiwi_router import RouterFactory

# Auto-detect best backend
router = RouterFactory.create("kiwi-router.gguf")

# Or specify device
router = RouterFactory.create("kiwi-router.onnx", device="cuda")

result = router.predict("find email scripts", "/home/user/project")
```

## Production Deployment Checklist

### Pre-Deployment

- [ ] Train model on comprehensive dataset (1000+ examples)
- [ ] Benchmark accuracy (target: 98%+)
- [ ] Benchmark latency (target: <100ms)
- [ ] Test on target hardware
- [ ] Optimize model size (quantization)
- [ ] Set up monitoring/logging

### Deployment

- [ ] Convert to target format (GGUF/CoreML/ONNX/TFLite)
- [ ] Package model with application
- [ ] Implement graceful fallbacks
- [ ] Add error handling
- [ ] Set up over-the-air model updates
- [ ] Configure resource limits (memory/GPU)

### Post-Deployment

- [ ] Monitor latency in production
- [ ] Track accuracy metrics
- [ ] Collect edge cases for retraining
- [ ] A/B test model versions
- [ ] Optimize based on real-world usage

## Troubleshooting

### Issue: High Latency (>200ms)

**Possible causes:**

- Model not using GPU/NPU
- Too large context window
- Not using optimized format

**Solutions:**

```python
# 1. Verify GPU usage
print(router.device)  # Should show GPU/Metal/CUDA

# 2. Reduce context
router = Router(model_path, n_ctx=256)  # vs 512

# 3. Use quantized model
# INT8 is 4x smaller than FP32, much faster
```

### Issue: High Memory Usage (>4GB)

**Solutions:**

```python
# Use smaller quantization
router = Router(
    "kiwi-router-q4_0.gguf"  # 4-bit vs 8-bit
)

# Limit context
router = Router(model_path, n_ctx=256)

# Unload model when not in use
router.unload()
```

### Issue: Inaccurate Predictions

**Solutions:**

- Collect examples of failures
- Add to training set
- Retrain model
- Increase model size (try 1B param version)
- Use beam search for better confidence

## Next Steps

- [Integration Patterns](./Integration%20Patterns%20-%20Connecting%20All%20Components.md) - Integrate router with Kiwi MCP
- Monitoring & Optimization - Coming Soon
- Model Updates - Coming Soon

---

**Platform Recommendation**: Start with macOS (Metal) for development, then expand to iOS and Linux (CUDA) for production.
