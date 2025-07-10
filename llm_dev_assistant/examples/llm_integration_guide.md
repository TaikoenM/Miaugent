# docs/llm_integration_guide.md
# LLM Integration and Selection Guide

This guide describes how to use the LLM selection and integration system in the LLM Development Assistant project.

## Overview

The LLM selection system allows you to dynamically choose the most appropriate LLM for different tasks based on:

1. **Task Purpose** - What the LLM will be used for (code generation, planning, vision analysis, etc.)
2. **Task Complexity** - How complex the task is (LOW, NORMAL, HARD, NIGHTMARE)

The system automatically selects the best available LLM based on these factors, following these rules:
- Vision tasks → Vision-capable models (like GPT-4 Vision)
- Hard/Nightmare tasks → Online powerful models (like GPT-4)
- Normal/Low tasks → Local models or LM Studio models

## Basic Usage

### Direct LLM Selection

```python
from llm_dev_assistant.llm.llm_selector import choose_llm, TaskPurpose, TaskComplexity

# Select an LLM for code generation (normal complexity)
llm = choose_llm(
    task_purpose=TaskPurpose.CODE_GENERATION,
    task_complexity=TaskComplexity.NORMAL,
    online_model="gpt-4",
    local_model="/path/to/local/model",
    lm_studio_model="llama2-7b"
)

# Use the LLM
response = llm.query("Write a Python function to calculate factorial")
```

### WorkflowEngine Integration

For more advanced usage, integrate with the WorkflowEngine:

```python
from llm_dev_assistant.workflow.workflow_engine import WorkflowEngine
from llm_dev_assistant.workflow.workflow_engine_llm_extension import WorkflowEngineLLMExtension

# Create workflow engine
workflow_engine = WorkflowEngine(...)

# Create extension
llm_extension = WorkflowEngineLLMExtension(workflow_engine)

# Generate code with appropriate LLM
code_result = llm_extension.generate_code(
    prompt="Implement a function to parse JSON files",
    complexity=TaskComplexity.HARD
)
```

## Configuration

The LLM system is configured through the `LLMConfig` class:

```python
from llm_dev_assistant.config.llm_config import LLMConfig

# Load configuration
config = LLMConfig("path/to/config.json")

# Set API key
config.set_api_key("openai", "your-api-key")
```

### Configuration File Format

```json
{
  "vision_model": "gpt-4-vision-preview",
  "online_model": "gpt-4",
  "local_model": "/path/to/local/model",
  "lm_studio_model": "llama2-7b",
  "default_complexity": "NORMAL",
  "task_complexity_mapping": {
    "CODE_GENERATION": "NORMAL",
    "CODE_DEBUGGING": "HARD",
    "PLANNING": "HARD",
    "QA": "NORMAL"
  },
  "api_keys": {
    "openai": "your-api-key"
  }
}
```

### Environment Variables

You can also configure the system through environment variables:

```sh
export OPENAI_API_KEY=your-api-key
export LLM_VISION_MODEL=gpt-4-vision-preview
export LLM_ONLINE_MODEL=gpt-4
export LLM_LOCAL_MODEL=/path/to/local/model
export LLM_LMSTUDIO_MODEL=llama2-7b
export LMSTUDIO_HOST=localhost
export LMSTUDIO_PORT=1234
```

## Task Purposes

The system supports a wide range of task purposes:

| Task Purpose | Description |
|--------------|-------------|
| VISION | Visual content analysis, OCR, image understanding |
| CODE_GENERATION | Generate new code from scratch |
| CODE_MODIFICATION | Modify existing code |
| CODE_REVIEW | Review and analyze code quality, security, etc. |
| CODE_DEBUGGING | Debug and fix issues in code |
| TEXT_GENERATION | Generate natural language text |
| TEXT_SUMMARIZATION | Summarize longer text into concise format |
| TRANSLATION | Translate between languages |
| SENTIMENT_ANALYSIS | Analyze sentiment in text |
| DATA_ANALYSIS | Analyze and interpret data |
| PLANNING | Plan operations, create roadmaps, strategy |
| PROMPT_ENGINEERING | Create or optimize prompts for other LLMs |
| REASONING | Complex reasoning, problem-solving |
| QA | Question answering |
| CREATIVE | Creative tasks like storytelling, poetry, etc. |
| RAG | Retrieval Augmented Generation |
| TECHNICAL_WRITING | Technical documentation, API docs, etc. |
| CONVERSATION | Conversational purposes or chat |
| DOMAIN_EXPERT | Domain-specific expertise (legal, medical, etc.) |
| MULTI_MODAL | Tasks involving multiple modalities (text, images, audio) |

## Task Complexities

| Complexity | Description |
|------------|-------------|
| LOW | Simple, straightforward tasks with minimal context |
| NORMAL | Moderate complexity, average context requirements |
| HARD | Complex tasks requiring more context and reasoning |
| NIGHTMARE | Extremely complex tasks requiring maximum context and capabilities |

## Model Types

| Model Type | Description |
|------------|-------------|
| VISION | Vision-capable models like GPT-4 Vision |
| ONLINE | Cloud-based powerful models like GPT-4 |
| LOCAL | Locally deployed models |
| LMSTUDIO | Models deployed via LM Studio |

## LM Studio Integration

The system includes special support for LM Studio models:

```python
from llm_dev_assistant.llm.lmstudio_adapter import LMStudioAdapter

# Mount a model manually
lm_studio = LMStudioAdapter(
    model_name="llama2-7b",
    context_length=8192,
    host="localhost",
    port=1234
)

# Unmount when done
lm_studio.unmount_model()
```

## Advanced Examples

### Using Different LLMs for Different Tasks

```python
code_llm = llm_extension.get_llm_for_task(TaskPurpose.CODE_GENERATION)
review_llm = llm_extension.get_llm_for_task(TaskPurpose.CODE_REVIEW)
planning_llm = llm_extension.get_llm_for_task(TaskPurpose.PLANNING, TaskComplexity.HARD)

# Generate code
code_suggestions = code_llm.get_code_suggestions(prompt)

# Review code
review_results = review_llm.verify_code_changes(original_code, new_code, requirements)

# Create development plan
plan = planning_llm.plan_next_steps(current_state, project_goals)
```

### Analyzing Images with Vision LLMs

```python
vision_llm = llm_extension.get_llm_for_task(TaskPurpose.VISION)
result = vision_llm.analyze_image(
    image_path="path/to/image.jpg",
    prompt="Describe what you see in this image and identify any code or text"
)
```

## Factory Pattern

The system uses a factory pattern to create LLM instances:

```python
from llm_dev_assistant.llm.llm_selector import LLMFactory, ModelType

# Create different types of LLMs
vision_llm = LLMFactory.create_vision_llm("gpt-4-vision-preview")
online_llm = LLMFactory.create_online_llm("gpt-4")
local_llm = LLMFactory.create_local_llm("/path/to/local/model")
lm_studio_llm = LLMFactory.create_lmstudio_llm("llama2-7b")
```

## Command Line Example

```sh
# Run the workflow integration example
python -m examples.workflow_llm_integration_example --config config.json --code-prompt "Implement a binary search tree" --output results.json
```

## Best Practices

1. **Use the WorkflowEngine extension** for most cases - it handles caching and selection automatically
2. **Configure through a file** for reproducible settings
3. **Match task complexity correctly** - use HARD for complex reasoning, LOW for simple tasks
4. **Clear the LLM cache** when done to free resources: `llm_extension.clear_llm_cache()`
5. **Use specialized methods** like `generate_code()`, `review_code()` rather than raw LLM calls
