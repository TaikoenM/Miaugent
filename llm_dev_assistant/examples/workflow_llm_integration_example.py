# examples/workflow_llm_integration_example.py
import os
import argparse
import json
from typing import Optional, Dict, Any

from llm_dev_assistant.workflow.workflow_engine import WorkflowEngine
from llm_dev_assistant.workflow.workflow_engine_llm_extension import WorkflowEngineLLMExtension
from llm_dev_assistant.llm.llm_selector import TaskPurpose, TaskComplexity
from llm_dev_assistant.config.llm_config import LLMConfig
from llm_dev_assistant.llm.openai_adapter import OpenAIAdapter
from llm_dev_assistant.code_manager.code_analyzer import CodeAnalyzer
from llm_dev_assistant.code_manager.change_implementer import ChangeImplementer
from llm_dev_assistant.code_manager.test_manager import TestManager
from llm_dev_assistant.parsers.gms2_parser import GMS2Parser
from llm_dev_assistant.logging.logger import logger


def setup_workflow() -> WorkflowEngine:
    """Set up and return a workflow engine instance."""
    # Initialize logging
    log_paths = logger.setup()
    log = logger.get_logger("example")
    log.info("Starting LLM integration example")

    # Load configuration with defaults
    config = LLMConfig()

    # Create a basic LLM for the workflow engine
    basic_llm = OpenAIAdapter()

    # Create workflow engine components
    parser = GMS2Parser()
    code_analyzer = CodeAnalyzer()
    change_implementer = ChangeImplementer()
    test_manager = TestManager()

    # Create workflow engine
    workflow_engine = WorkflowEngine(
        llm=basic_llm,
        parser=parser,
        code_analyzer=code_analyzer,
        change_implementer=change_implementer,
        test_manager=test_manager
    )

    return workflow_engine


def extend_workflow_with_llm(workflow_engine: WorkflowEngine) -> WorkflowEngineLLMExtension:
    """Extend workflow engine with LLM capabilities."""
    # Create extension
    llm_extension = WorkflowEngineLLMExtension(workflow_engine)
    return llm_extension


def demonstrate_task_specific_llms(llm_extension: WorkflowEngineLLMExtension) -> Dict[str, Any]:
    """Demonstrate using different LLMs for different tasks."""
    results = {}

    # Get LLMs for different tasks
    code_gen_llm = llm_extension.get_llm_for_task(TaskPurpose.CODE_GENERATION)
    code_review_llm = llm_extension.get_llm_for_task(TaskPurpose.CODE_REVIEW)
    planning_llm = llm_extension.get_llm_for_task(TaskPurpose.PLANNING, TaskComplexity.HARD)

    # Log the LLM types
    log = logger.get_logger("example")
    log.info(f"Code generation LLM: {type(code_gen_llm).__name__}")
    log.info(f"Code review LLM: {type(code_review_llm).__name__}")
    log.info(f"Planning LLM: {type(planning_llm).__name__}")

    # Store results
    results["code_gen_llm"] = type(code_gen_llm).__name__
    results["code_review_llm"] = type(code_review_llm).__name__
    results["planning_llm"] = type(planning_llm).__name__

    return results


def generate_code_example(llm_extension: WorkflowEngineLLMExtension, prompt: str) -> Dict[str, Any]:
    """Generate code using the LLM extension."""
    log = logger.get_logger("example")
    log.info(f"Generating code for prompt: {prompt}")

    # Generate code with different complexities
    normal_code = llm_extension.generate_code(prompt, complexity=TaskComplexity.NORMAL)
    hard_code = llm_extension.generate_code(prompt, complexity=TaskComplexity.HARD)

    return {
        "normal_complexity": normal_code,
        "hard_complexity": hard_code
    }


def review_code_example(llm_extension: WorkflowEngineLLMExtension,
                        original_code: str, new_code: str, requirements: str) -> Dict[str, Any]:
    """Review code changes using the LLM extension."""
    log = logger.get_logger("example")
    log.info("Reviewing code changes")

    # Review code
    review_result = llm_extension.review_code(original_code, new_code, requirements)

    return review_result


def create_plan_example(llm_extension: WorkflowEngineLLMExtension,
                        current_state: Dict[str, Any], goals: list) -> list:
    """Create a development plan using the LLM extension."""
    log = logger.get_logger("example")
    log.info(f"Creating development plan with {len(goals)} goals")

    # Create plan
    plan = llm_extension.create_development_plan(current_state, goals)

    return plan


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="LLM Workflow Integration Example")

    parser.add_argument(
        "--config",
        help="Path to LLM configuration file"
    )

    parser.add_argument(
        "--code-prompt",
        default="Write a Python function to calculate the factorial of a number",
        help="Prompt for code generation example"
    )

    parser.add_argument(
        "--output",
        help="Path to save example results"
    )

    args = parser.parse_args()

    # Load custom configuration if provided
    if args.config:
        config = LLMConfig(args.config)

    # Setup workflow engine and extension
    workflow_engine = setup_workflow()
    llm_extension = extend_workflow_with_llm(workflow_engine)

    # Demonstrate features
    results = {}

    # Task-specific LLMs
    results["task_specific_llms"] = demonstrate_task_specific_llms(llm_extension)

    # Code generation
    results["code_generation"] = generate_code_example(llm_extension, args.code_prompt)

    # Code review
    original_code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

    new_code = """
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

    requirements = "Add error handling for negative numbers and improve documentation"

    results["code_review"] = review_code_example(
        llm_extension, original_code, new_code, requirements
    )

    # Planning
    current_state = {
        "project_path": "/path/to/project",
        "completed_tasks": ["Setup project", "Implement parser"],
        "pending_tasks": []
    }

    goals = [
        "Complete LLM integration",
        "Add support for GitHub",
        "Improve test coverage"
    ]

    results["development_plan"] = create_plan_example(llm_extension, current_state, goals)

    # Print results
    print("\nExample Results:")
    print("===============")

    print("\nTask-Specific LLMs:")
    for task, llm_type in results["task_specific_llms"].items():
        print(f"- {task}: {llm_type}")

    print("\nCode Generation (Normal Complexity):")
    print(results["code_generation"]["normal_complexity"]["code"])

    print("\nCode Review:")
    print(f"Meets requirements: {results['code_review'].get('meets_requirements', 'Unknown')}")
    print(f"Analysis: {results['code_review'].get('analysis', '')[:100]}...")

    print("\nDevelopment Plan:")
    for i, step in enumerate(results["development_plan"]):
        print(f"{i + 1}. {step.get('task', 'Unknown task')} (Priority: {step.get('priority', 'medium')})")

    # Save results if output path provided
    if args.output:
        try:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"\nError saving results: {str(e)}")

    # Clean up
    llm_extension.clear_llm_cache()


if __name__ == "__main__":
    main()