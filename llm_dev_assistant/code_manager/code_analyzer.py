# llm_dev_assistant/code_manager/code_analyzer.py
import os
import ast
from typing import Dict, List, Any, Optional


class CodeAnalyzer:
    """Analyzes code structure and dependencies."""

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze a project's code structure.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with project structure information
        """
        result = {
            "files": [],
            "dependencies": {},
            "modules": {},
            "functions": {},
            "classes": {}
        }

        # Walk through the project directory
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_analysis = self.analyze_file(file_path)

                    # Add to results
                    result["files"].append(file_path)

                    # Add functions and classes
                    for func in file_analysis.get("functions", []):
                        result["functions"][func["name"]] = {
                            "file": file_path,
                            **func
                        }

                    for cls in file_analysis.get("classes", []):
                        result["classes"][cls["name"]] = {
                            "file": file_path,
                            **cls
                        }

                    # Add module info
                    module_name = self._get_module_name(file_path, project_path)
                    result["modules"][module_name] = {
                        "file": file_path,
                        "imports": file_analysis.get("imports", []),
                        "functions": [f["name"] for f in file_analysis.get("functions", [])],
                        "classes": [c["name"] for c in file_analysis.get("classes", [])]
                    }

                    # Add dependencies
                    for imp in file_analysis.get("imports", []):
                        if imp not in result["dependencies"]:
                            result["dependencies"][imp] = []
                        result["dependencies"][imp].append(module_name)

        return result

    def _get_module_name(self, file_path: str, project_path: str) -> str:
        """Get the module name from a file path."""
        rel_path = os.path.relpath(file_path, project_path)
        module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
        return module_name

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with file analysis results
        """
        result = {
            "imports": [],
            "functions": [],
            "classes": []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        result["imports"].append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for name in node.names:
                            result["imports"].append(f"{node.module}.{name.name}")

                # Extract functions
                elif isinstance(node, ast.FunctionDef):
                    function_info = {
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    }
                    result["functions"].append(function_info)

                # Extract classes
                elif isinstance(node, ast.ClassDef):
                    class_methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                "name": item.name,
                                "lineno": item.lineno,
                                "args": [arg.arg for arg in item.args.args],
                                "docstring": ast.get_docstring(item)
                            }
                            class_methods.append(method_info)

                    class_info = {
                        "name": node.name,
                        "lineno": node.lineno,
                        "bases": [base.id if isinstance(base, ast.Name) else "complex_base" for base in node.bases],
                        "methods": class_methods,
                        "docstring": ast.get_docstring(node)
                    }
                    result["classes"].append(class_info)

        except Exception as e:
            result["error"] = str(e)

        return result

    def find_dependencies(self, file_path: str, project_path: str) -> Dict[str, Any]:
        """
        Find dependencies for a specific file.

        Args:
            file_path: Path to the file
            project_path: Path to the project

        Returns:
            Dictionary with dependency information
        """
        result = {
            "direct_dependencies": [],
            "reverse_dependencies": []
        }

        # Get module name
        module_name = self._get_module_name(file_path, project_path)

        # Analyze project
        project_analysis = self.analyze_project(project_path)

        # Get direct dependencies
        if module_name in project_analysis["modules"]:
            result["direct_dependencies"] = project_analysis["modules"][module_name]["imports"]

        # Get reverse dependencies (modules that import this one)
        for mod, data in project_analysis["modules"].items():
            if module_name in data["imports"]:
                result["reverse_dependencies"].append(mod)

        return result