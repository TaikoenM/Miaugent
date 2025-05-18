# llm_dev_assistant/__main__.py
import os
import argparse
import json
import logging
from typing import Dict, List, Any, Optional

from .logging.logger import logger
from .llm.openai_adapter import OpenAIAdapter
from .llm.local_llm_adapter import LocalLLMAdapter
from .parsers.gms2_parser import GMS2Parser
from .code_manager.code_analyzer import CodeAnalyzer
from .code_manager.change_implementer import ChangeImplementer
from .code_manager.test_manager import TestManager
from .workflow.workflow_engine import WorkflowEngine

def create_workflow_engine(llm_type: str, local_model_path: Optional[str] = None) -> WorkflowEngine:
    """Create a workflow engine with the specified components."""
    log = logger.get_logger("workflow_setup")
    log.info(f"Creating workflow engine with LLM type: {llm_type}")

    # Initialize LLM
    if llm_type.lower() == "openai":
        log.debug("Initializing OpenAI adapter")
        llm = OpenAIAdapter()
    elif llm_type.lower() == "local":
        log.debug(f"Initializing local LLM adapter with model: {local_model_path}")
        if not local_model_path:
            log.error("Local model path not provided")
            raise ValueError("local_model_path must be provided for local LLM")
        llm = LocalLLMAdapter(model_path=local_model_path)
    else:
        log.error(f"Unsupported LLM type: {llm_type}")
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    # Initialize components
    log.debug("Initializing parser, code analyzer, change implementer, and test manager")
    parser = GMS2Parser()
    code_analyzer = CodeAnalyzer()
    change_implementer = ChangeImplementer()
    test_manager = TestManager()

    # Create workflow engine
    log.info("Creating workflow engine with initialized components")
    workflow_engine = WorkflowEngine(
        llm=llm,
        parser=parser,
        code_analyzer=code_analyzer,
        change_implementer=change_implementer,
        test_manager=test_manager
    )

    return workflow_engine


def main():
    """Main CLI entry point."""
    # Initialize logging system
    log_paths = logger.setup()
    log = logger.get_logger("main")
    log.info("LLM Development Assistant starting")
    log.info(f"Debug log: {log_paths['debug_log']}")
    log.info(f"Info log: {log_paths['info_log']}")

    # ... rest of the main function ...
    parser = argparse.ArgumentParser(description="LLM-assisted development automation tool")

    # Top-level arguments
    parser.add_argument(
        "--llm",
        choices=["openai", "local"],
        default="openai",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--local-model-path",
        help="Path to local model (required if --llm=local)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Initialize project
    init_parser = subparsers.add_parser("init", help="Initialize project")
    init_parser.add_argument("project_path", help="Path to the project directory")
    init_parser.add_argument("--output", help="Path to save project context")

    # Request code implementation
    code_parser = subparsers.add_parser("code", help="Request code implementation")
    code_parser.add_argument("task", help="Task description")
    code_parser.add_argument("--file", help="Path to the file to modify")
    code_parser.add_argument("--output", help="Path to save suggestions")

    # Verify implementation
    verify_parser = subparsers.add_parser("verify", help="Verify code implementation")
    verify_parser.add_argument("original_file", help="Path to the original file")
    verify_parser.add_argument("new_file", help="Path to the new implementation")
    verify_parser.add_argument("requirements", help="Requirements for verification")
    verify_parser.add_argument("--output", help="Path to save verification results")

    # Implement changes
    implement_parser = subparsers.add_parser("implement", help="Implement code changes")
    implement_parser.add_argument("file", help="Path to the file to modify")
    implement_parser.add_argument("code_file", help="Path to the file with new code")
    implement_parser.add_argument("--output", help="Path to save implementation results")

    # Verify tests
    test_verify_parser = subparsers.add_parser("verify-tests", help="Verify if tests cover the modified file")
    test_verify_parser.add_argument("file", help="Path to the modified file")
    test_verify_parser.add_argument("--output", help="Path to save test verification results")

    # Run tests
    test_run_parser = subparsers.add_parser("run-tests", help="Run integration tests")
    test_run_parser.add_argument("--tests", nargs="+", help="Paths to test files to run")
    test_run_parser.add_argument("--output", help="Path to save test results")

    # Plan next steps
    plan_parser = subparsers.add_parser("plan", help="Plan next development steps")
    plan_parser.add_argument("goals", nargs="+", help="Project goals")
    plan_parser.add_argument("--output", help="Path to save planning results")

    # Save/load workflow state
    save_parser = subparsers.add_parser("save", help="Save workflow state")
    save_parser.add_argument("output", help="Path to save workflow state")

    load_parser = subparsers.add_parser("load", help="Load workflow state")
    load_parser.add_argument("input", help="Path to load workflow state from")

    # Parse arguments
    args = parser.parse_args()

    # Create workflow engine
    workflow_engine = create_workflow_engine(args.llm, args.local_model_path)

    # Execute command
    result = {}

    if args.command == "init":
        log.info(f"Initializing project: {args.project_path}")
        result = workflow_engine.initialize_project(args.project_path)
        if args.output:
            log.debug(f"Saving project context to: {args.output}")
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    elif args.command == "code":
        result = workflow_engine.request_code_implementation(args.task, args.file)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    elif args.command == "verify":
        with open(args.original_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        with open(args.new_file, 'r', encoding='utf-8') as f:
            new_code = f.read()
        with open(args.requirements, 'r', encoding='utf-8') as f:
            requirements = f.read()

        result = workflow_engine.verify_implementation(
            original_code=original_code,
            new_code=new_code,
            requirements=requirements
        )
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    elif args.command == "implement":
        with open(args.code_file, 'r', encoding='utf-8') as f:
            new_code = f.read()

        result = workflow_engine.implement_changes(
            file_path=args.file,
            new_code=new_code
        )
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    elif args.command == "verify-tests":
        result = workflow_engine.verify_tests(args.file)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    elif args.command == "run-tests":
        result = workflow_engine.run_integration_tests(args.tests)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    elif args.command == "plan":
        result = workflow_engine.plan_next_steps(args.goals)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    elif args.command == "save":
        result = {"path": workflow_engine.save_workflow_state(args.output)}

    elif args.command == "load":
        result = workflow_engine.load_workflow_state(args.input)

    # Print result
    log.debug(f"Command result: {json.dumps(result, indent=2)[:1000]}..." if len(
        json.dumps(result)) > 1000 else f"Command result: {json.dumps(result, indent=2)}")
    print(json.dumps(result, indent=2))

    log.info("LLM Development Assistant finished")


if __name__ == "__main__":
    main()