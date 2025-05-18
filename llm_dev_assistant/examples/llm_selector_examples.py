# examples/llm_selector_example.py
import os
import argparse
from typing import Optional

from llm_dev_assistant.llm.llm_selector import (
    TaskPurpose,
    TaskComplexity,
    choose_llm
)


def main():
    """CLI example for the LLM selector."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM Selector Example")

    parser.add_argument(
        "task",
        choices=[purpose.name for purpose in TaskPurpose],
        help="The purpose of the task"
    )

    parser.add_argument(
        "complexity",
        choices=[complexity.name for complexity in TaskComplexity],
        help="The complexity of the task"
    )

    parser.add_argument(
        "--vision-model",
        default="gpt-4-vision-preview",
        help="Name of the vision model to use (default: gpt-4-vision-preview)"
    )

    parser.add_argument(
        "--online-model",
        default="gpt-4",
        help="Name of the online model to use (default: gpt-4)"
    )

    parser.add_argument(
        "--local-model",
        help="Path to the local model"
    )

    parser.add_argument(
        "--lm-studio-model",
        help="Name of the LM Studio model"
    )

    parser.add_argument(
        "--prompt",
        default="Hello, how can you help me with this task?",
        help="Prompt to send to the LLM"
    )

    args = parser.parse_args()

    # Choose the appropriate LLM
    llm = choose_llm(
        task_purpose=args.task,
        task_complexity=args.complexity,
        vision_model=args.vision_model,
        online_model=args.online_model,
        local_model=args.local_model,
        lm_studio_model=args.lm_studio_model
    )

    # Send a query to the LLM
    print(f"\nTask: {args.task}, Complexity: {args.complexity}")
    print(f"Using model: {type(llm).__name__}")
    print("\nSending prompt to LLM...")

    response = llm.query(args.prompt)

    print("\nResponse from LLM:")
    print("-" * 50)
    print(response)
    print("-" * 50)


def image_analysis_example(image_path: str, prompt: Optional[str] = None):
    """Example of using the LLM selector for image analysis."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    if prompt is None:
        prompt = "What can you see in this image? Describe it in detail."

    # Choose a vision LLM
    llm = choose_llm(
        task_purpose=TaskPurpose.VISION,
        task_complexity=TaskComplexity.NORMAL
    )

    # Make sure we got a vision LLM
    if not hasattr(llm, "analyze_image"):
        print("Error: Selected LLM does not support image analysis")
        return

    print(f"\nAnalyzing image: {image_path}")
    print("Prompt:", prompt)
    print("\nSending request to Vision LLM...")

    # Analyze the image
    response = llm.analyze_image(image_path, prompt)

    print("\nResponse from Vision LLM:")
    print("-" * 50)
    print(response)
    print("-" * 50)


def code_generation_example(complexity: str = "NORMAL",
                            prompt: Optional[str] = None):
    """Example of using the LLM selector for code generation."""
    if prompt is None:
        prompt = """
        Write a Python function to calculate the Fibonacci sequence up to n terms.
        The function should return a list of the sequence.
        """

    # Choose an appropriate LLM based on complexity
    llm = choose_llm(
        task_purpose=TaskPurpose.CODE_GENERATION,
        task_complexity=complexity
    )

    print(f"\nCode Generation Example (Complexity: {complexity})")
    print("Using model:", type(llm).__name__)
    print("\nPrompt:", prompt)
    print("\nGenerating code...")

    # Get code suggestions
    suggestions = llm.get_code_suggestions(prompt)

    print("\nGenerated Code:")
    print("-" * 50)
    print(suggestions.get("code", "No code generated"))
    print("-" * 50)

    print("\nExplanation:")
    print(suggestions.get("explanation", "No explanation provided"))


def demonstrate_model_selection():
    """Demonstrate how different tasks and complexities select different models."""
    examples = [
        {"task": TaskPurpose.VISION, "complexity": TaskComplexity.NORMAL, "description": "Image analysis (Vision)"},
        {"task": TaskPurpose.CODE_DEBUGGING, "complexity": TaskComplexity.HARD,
         "description": "Hard code debugging (Online)"},
        {"task": TaskPurpose.TEXT_GENERATION, "complexity": TaskComplexity.NORMAL,
         "description": "Normal text generation (Local/LM Studio)"},
        {"task": TaskPurpose.MULTI_MODAL, "complexity": TaskComplexity.LOW, "description": "Multi-modal task (Vision)"},
        {"task": TaskPurpose.REASONING, "complexity": TaskComplexity.NIGHTMARE,
         "description": "Nightmare complexity reasoning (Online)"}
    ]

    print("\nDemonstrating LLM Model Selection:")
    print("-" * 70)
    print(f"{'Task Type':<25} {'Complexity':<15} {'Selected Model Type':<20} {'Description'}")
    print("-" * 70)

    for example in examples:
        # Create custom factory functions that return model type instead of creating actual instances
        factories = {model_type: lambda *args, model_type=model_type: f"[{model_type.name}_MODEL]"
                     for model_type in [type(enum) for enum in []]}

        # We're just checking what kind of model would be chosen, not creating real instances
        from llm_dev_assistant.llm.llm_selector import ModelType, LLMSelector

        selector = LLMSelector()
        _, model_type = selector.select_llm(
            task_purpose=example["task"],
            task_complexity=example["complexity"]
        )

        print(f"{example['task'].name:<25} {example['complexity'].name:<15} {model_type:<20} {example['description']}")


if __name__ == "__main__":
    # Uncomment this to see the model selection logic
    # demonstrate_model_selection()

    main()

    # Uncomment these examples to test specific use cases
    # image_analysis_example("path/to/your/image.jpg")
    # code_generation_example("HARD")