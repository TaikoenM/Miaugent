# llm_dev_assistant/code_manager/test_manager.py
import os
import subprocess
import sys
from typing import Dict, List, Any, Optional


class TestManager:
    """Manages test verification and execution."""

    def find_relevant_tests(self, file_path: str) -> List[str]:
        """
        Find tests relevant to a file.

        Args:
            file_path: Path to the file

        Returns:
            List of relevant test file paths
        """
        relevant_tests = []

        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Get the directory containing the file
        parent_dir = os.path.dirname(file_path)
        project_root = self._find_project_root(parent_dir)

        # Common test directories
        test_dirs = [
            os.path.join(project_root, "tests"),
            os.path.join(project_root, "test"),
            parent_dir
        ]

        # Look for test files in common test directories
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for root, _, files in os.walk(test_dir):
                    for file in files:
                        # Look for files like test_*.py or *_test.py that might be related
                        if (file.startswith(f"test_{base_name}") or
                                file.startswith(f"{base_name}_test") or
                                file == f"test_{base_name}.py" or
                                file == f"{base_name}_test.py"):
                            relevant_tests.append(os.path.join(root, file))

        return relevant_tests

    def _find_project_root(self, start_dir: str) -> str:
        """Find the project root directory."""
        current_dir = start_dir

        # Look for common project root indicators
        while current_dir and current_dir != os.path.dirname(current_dir):
            if (os.path.exists(os.path.join(current_dir, "setup.py")) or
                    os.path.exists(os.path.join(current_dir, "pyproject.toml")) or
                    os.path.exists(os.path.join(current_dir, ".git"))):
                return current_dir
            current_dir = os.path.dirname(current_dir)

        # If no root indicators are found, return the original directory
        return start_dir

    def analyze_test_coverage(self, file_path: str, test_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze if tests provide good coverage for a file.

        Args:
            file_path: Path to the file
            test_paths: List of test file paths

        Returns:
            Dictionary with test coverage analysis
        """
        result = {
            "file_path": file_path,
            "test_paths": test_paths,
            "has_tests": len(test_paths) > 0,
            "coverage_estimate": "unknown"
        }

        # Try to use coverage module if available
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()

            # Run the tests
            self.run_tests(test_paths)

            # Stop coverage and analyze
            cov.stop()
            cov.save()

            # Get coverage data
            file_coverage = cov.analysis(file_path)
            if file_coverage:
                _, _, excluded, missing, _ = file_coverage
                total_lines = len(self._get_file_lines(file_path))
                excluded_lines = len(excluded)
                missing_lines = len(missing)
                covered_lines = total_lines - excluded_lines - missing_lines

                if total_lines - excluded_lines > 0:
                    coverage_percent = (covered_lines / (total_lines - excluded_lines)) * 100
                    result["coverage_percent"] = coverage_percent

                    if coverage_percent >= 80:
                        result["coverage_estimate"] = "good"
                    elif coverage_percent >= 50:
                        result["coverage_estimate"] = "moderate"
                    else:
                        result["coverage_estimate"] = "poor"

        except (ImportError, Exception) as e:
            # Fallback to simple estimate based on test existence
            if result["has_tests"]:
                if len(test_paths) >= 2:
                    result["coverage_estimate"] = "potentially_good"
                else:
                    result["coverage_estimate"] = "potentially_moderate"
            else:
                result["coverage_estimate"] = "missing"

            result["error"] = str(e)

        return result

    def _get_file_lines(self, file_path: str) -> List[str]:
        """Get lines of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.readlines()
        except Exception:
            return []

    def run_tests(self, test_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run tests.

        Args:
            test_paths: Optional list of test paths to run

        Returns:
            Dictionary with test results
        """
        result = {
            "test_paths": test_paths,
            "success": True,
            "results": {}
        }

        # If no test paths provided, find all tests
        if not test_paths:
            # Find project root
            current_dir = os.getcwd()
            project_root = self._find_project_root(current_dir)

            # Try common test directories
            test_dirs = [
                os.path.join(project_root, "tests"),
                os.path.join(project_root, "test")
            ]

            test_paths = []
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    for root, _, files in os.walk(test_dir):
                        for file in files:
                            if file.startswith("test_") and file.endswith(".py"):
                                test_paths.append(os.path.join(root, file))

        # Run the tests
        if test_paths:
            for test_path in test_paths:
                if os.path.exists(test_path):
                    try:
                        # Try to use pytest for running tests
                        try:
                            import pytest
                            pytest_result = pytest.main(["-v", test_path])
                            result["results"][test_path] = {
                                "success": pytest_result == 0,
                                "exit_code": pytest_result,
                                "output": "Ran with pytest"
                            }
                            if pytest_result != 0:
                                result["success"] = False
                        except ImportError:
                            # Fallback to running with subprocess
                            process = subprocess.run(
                                [sys.executable, test_path],
                                capture_output=True,
                                text=True
                            )
                            output = process.stdout + process.stderr
                            result["results"][test_path] = {
                                "success": process.returncode == 0,
                                "exit_code": process.returncode,
                                "output": output
                            }
                            if process.returncode != 0:
                                result["success"] = False
                    except Exception as e:
                        result["results"][test_path] = {
                            "success": False,
                            "error": str(e)
                        }
                        result["success"] = False
                else:
                    result["results"][test_path] = {
                        "success": False,
                        "error": f"Test file does not exist: {test_path}"
                    }
                    result["success"] = False
        else:
            result["success"] = False
            result["error"] = "No test paths provided or found"

        return result