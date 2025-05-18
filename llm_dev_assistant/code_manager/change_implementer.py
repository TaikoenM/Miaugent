# llm_dev_assistant/code_manager/change_implementer.py
import os
import shutil
from typing import Dict, Any, Optional


class ChangeImplementer:
    """Implements code changes."""

    def implement_changes(self, file_path: str, new_code: str) -> Dict[str, Any]:
        """
        Implement changes to a file.

        Args:
            file_path: Path to the file to modify
            new_code: New code to implement

        Returns:
            Dictionary with implementation results
        """
        result = {
            "file_path": file_path,
            "status": "success",
            "message": "Changes implemented successfully"
        }

        try:
            # Create a backup
            backup_path = f"{file_path}.bak"
            self._create_backup(file_path, backup_path)
            result["backup_path"] = backup_path

            # Write the new code
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_code)

            # Verify the changes were written
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content != new_code:
                    result["status"] = "warning"
                    result["message"] = "Changes may not have been fully implemented"

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error implementing changes: {str(e)}"

            # Try to restore from backup
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, file_path)
                    result["message"] += " (restored from backup)"
                except Exception as restore_error:
                    result["message"] += f" (failed to restore from backup: {str(restore_error)})"

        return result

    def _create_backup(self, file_path: str, backup_path: str) -> None:
        """Create a backup of a file."""
        if os.path.exists(file_path):
            shutil.copy2(file_path, backup_path)

    def create_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Create a new file.

        Args:
            file_path: Path to the file to create
            content: File content

        Returns:
            Dictionary with creation results
        """
        result = {
            "file_path": file_path,
            "status": "success",
            "message": "File created successfully"
        }

        try:
            # Create directories if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Check if file exists
            if os.path.exists(file_path):
                result["status"] = "warning"
                result["message"] = "File already exists, creating backup and overwriting"

                # Create a backup
                backup_path = f"{file_path}.bak"
                self._create_backup(file_path, backup_path)
                result["backup_path"] = backup_path

            # Write the content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error creating file: {str(e)}"

        return result

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file.

        Args:
            file_path: Path to the file to delete

        Returns:
            Dictionary with deletion results
        """
        result = {
            "file_path": file_path,
            "status": "success",
            "message": "File deleted successfully"
        }

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result["status"] = "warning"
                result["message"] = "File does not exist"
                return result

            # Create a backup
            backup_path = f"{file_path}.bak"
            self._create_backup(file_path, backup_path)
            result["backup_path"] = backup_path

            # Delete the file
            os.remove(file_path)

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error deleting file: {str(e)}"

        return result