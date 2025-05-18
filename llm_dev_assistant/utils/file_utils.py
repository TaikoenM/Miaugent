# llm_dev_assistant/utils/file_utils.py
import os
from typing import Optional


def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read a file and return its contents.

    Args:
        file_path: Path to the file
        encoding: File encoding

    Returns:
        File contents as a string
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Write content to a file.

    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directories if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception:
        return False


def ensure_directory(directory_path: str) -> bool:
    """
    Ensure a directory exists.

    Args:
        directory_path: Path to the directory

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception:
        return False


def list_files(directory_path: str, extension: Optional[str] = None) -> list:
    """
    List files in a directory, optionally filtered by extension.

    Args:
        directory_path: Path to the directory
        extension: Optional file extension filter

    Returns:
        List of file paths
    """
    result = []

    if not os.path.exists(directory_path):
        return result

    for root, _, files in os.walk(directory_path):
        for file in files:
            if extension is None or file.endswith(extension):
                result.append(os.path.join(root, file))

    return result


def get_relative_path(file_path: str, base_path: str) -> str:
    """
    Get the path of a file relative to a base path.

    Args:
        file_path: Absolute path to the file
        base_path: Base path

    Returns:
        Relative path
    """
    return os.path.relpath(file_path, base_path)