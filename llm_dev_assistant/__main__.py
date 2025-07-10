# llm_dev_assistant/__main__.py
import os
import argparse
import json
import logging
from typing import Dict, List, Any, Optional

from .log_system.logger import logger
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
    # Check if GUI was requested
    if len(os.sys.argv) > 1 and os.sys.argv[1] == 'gui':
        # Launch GUI
        from .gui.main_window import main as gui_main
        gui_main()
        return

    # Initialize log_system system
    log_paths = logger.setup()
    log = logger.get_logger("main")
    log.info("LLM Development Assistant starting")
    log.info(f"Debug log: {log_paths['debug_log']}")
    log.info(f"Info log: {log_paths['info_log']}")

    # ... rest of the main function ...
    parser = argparse.ArgumentParser(description="LLM-assisted development automation tool")

    # Add GUI option
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI interface"
    )

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

    # Launch GUI if requested
    if args.gui:
        from .gui.main_window import main as gui_main
        gui_main()
        return

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

"""***
# Launch GUI interface
python -m llm_dev_assistant gui
# or
python -m llm_dev_assistant --gui

# Set GitHub token
export GITHUB_TOKEN=your_github_token

# Clone a repository
python -m llm_dev_assistant github clone https://github.com/username/repo.git

# Create a feature branch
python -m llm_dev_assistant github branch feature/new-feature

# Generate code with LLM
python -m llm_dev_assistant code "Implement a function to parse GML files" --file path/to/file.py

# Implement the changes
python -m llm_dev_assistant implement path/to/file.py suggested_code.json

# Commit the changes
python -m llm_dev_assistant github commit "Add GML file parser function"

# Create a pull request
python -m llm_dev_assistant github pr username/repo feature/new-feature --title "Add GML Parser"

# Using OpenAI (default)
python -m llm_dev_assistant [command] [arguments]

# Using a local LLM
python -m llm_dev_assistant --llm=local --local-model-path=/path/to/llm/model [command] [arguments]

# Initialize a Game Maker Studio 2 project
python -m llm_dev_assistant init /path/to/gms2/project

# Initialize and save context to a file
python -m llm_dev_assistant init /path/to/gms2/project --output=project_context.json

# Request code for a new feature with no existing file
python -m llm_dev_assistant code "Implement a function to parse GMS2 sprite files" 

# Request code for modifying an existing file
python -m llm_dev_assistant code "Add error handling to the parse_script method" --file=llm_dev_assistant/parsers/gms2_parser.py

# Save the LLM suggestions to a file
python -m llm_dev_assistant code "Create a unit test for the GMS2Parser" --output=parser_test_suggestion.json

# Verify if implementation meets requirements
python -m llm_dev_assistant verify original_gms2_parser.py new_gms2_parser.py requirements.txt

# Save verification results
python -m llm_dev_assistant verify original_gms2_parser.py new_gms2_parser.py requirements.txt --output=verification_result.json

# Implement code changes from a suggestion file
python -m llm_dev_assistant implement llm_dev_assistant/parsers/gms2_parser.py new_parser_code.txt

# Implement and save the implementation report
python -m llm_dev_assistant implement llm_dev_assistant/parsers/gms2_parser.py new_parser_code.txt --output=implementation_report.json

# Check if existing tests cover a modified file
python -m llm_dev_assistant verify-tests llm_dev_assistant/parsers/gms2_parser.py

# Save test verification results
python -m llm_dev_assistant verify-tests llm_dev_assistant/parsers/gms2_parser.py --output=test_coverage.json

# Run all available tests
python -m llm_dev_assistant run-tests

# Run specific test files
python -m llm_dev_assistant run-tests --tests tests/test_gms2_parser.py tests/test_workflow.py

# Save test results
python -m llm_dev_assistant run-tests --tests tests/test_gms2_parser.py --output=test_results.json

# Plan next steps based on a single goal
python -m llm_dev_assistant plan "Complete GMS2 parser implementation"

# Plan next steps with multiple goals
python -m llm_dev_assistant plan "Fix parser bugs" "Add support for sprite sheets" "Improve test coverage"

# Save planning results
python -m llm_dev_assistant plan "Implement GitHub integration" --output=development_plan.json

# Save current workflow state
python -m llm_dev_assistant save workflow_state.json

# Load saved workflow state
python -m llm_dev_assistant load workflow_state.json

# 1. Initialize the project
python -m llm_dev_assistant init /path/to/gms2/project

# 2. Plan next development steps
python -m llm_dev_assistant plan "Implement GMS2 asset parser" "Create unit tests"

# 3. Request code for implementation
python -m llm_dev_assistant code "Create a function to parse GMS2 room files" --file=llm_dev_assistant/parsers/gms2_parser.py --output=room_parser.json

# 4. Implement the suggested changes
python -m llm_dev_assistant implement llm_dev_assistant/parsers/gms2_parser.py room_parser.json

# 5. Verify if tests cover the implementation
python -m llm_dev_assistant verify-tests llm_dev_assistant/parsers/gms2_parser.py

# 6. Request test code if needed
python -m llm_dev_assistant code "Create unit tests for GMS2 room parser" --file=tests/test_gms2_parser.py --output=room_parser_tests.json

# 7. Implement the test code
python -m llm_dev_assistant implement tests/test_gms2_parser.py room_parser_tests.json

# 8. Run the tests
python -m llm_dev_assistant run-tests --tests tests/test_gms2_parser.py

# 9. Save workflow state
python -m llm_dev_assistant save workflows/gms2_room_parser.json

***"""