# llm_dev_assistant/github/repo_manager.py
import os
import shutil
from typing import Dict, List, Any, Optional
from git import Repo, GitCommandError


class RepoManager:
    """Manages Git repositories."""

    def clone_repo(self, repo_url: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Clone a Git repository.

        Args:
            repo_url: Repository URL
            target_dir: Target directory (if None, uses repo name)

        Returns:
            Dictionary with clone results
        """
        result = {
            "repo_url": repo_url,
            "status": "success",
            "message": "Repository cloned successfully"
        }

        try:
            # Determine target directory
            if target_dir is None:
                repo_name = repo_url.split('/')[-1]
                if repo_name.endswith('.git'):
                    repo_name = repo_name[:-4]
                target_dir = os.path.join(os.getcwd(), repo_name)

            # Clone repository
            repo = Repo.clone_from(repo_url, target_dir)
            result["target_dir"] = target_dir
            result["default_branch"] = repo.active_branch.name

        except GitCommandError as e:
            result["status"] = "error"
            result["message"] = f"Error cloning repository: {str(e)}"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Unexpected error: {str(e)}"

        return result

    def create_branch(self, repo_path: str, branch_name: str,
                      from_branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new branch.

        Args:
            repo_path: Path to repository
            branch_name: New branch name
            from_branch: Source branch (if None, uses current branch)

        Returns:
            Dictionary with branch creation results
        """
        result = {
            "repo_path": repo_path,
            "branch_name": branch_name,
            "status": "success",
            "message": f"Branch '{branch_name}' created successfully"
        }

        try:
            repo = Repo(repo_path)

            # Check if branch already exists
            if branch_name in [b.name for b in repo.branches]:
                result["status"] = "warning"
                result["message"] = f"Branch '{branch_name}' already exists, checking it out"
                repo.git.checkout(branch_name)
                return result

            # Checkout source branch if specified
            if from_branch:
                repo.git.checkout(from_branch)

            # Create and checkout new branch
            repo.git.checkout('-b', branch_name)

        except GitCommandError as e:
            result["status"] = "error"
            result["message"] = f"Git error: {str(e)}"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Unexpected error: {str(e)}"

        return result

    def commit_changes(self, repo_path: str, message: str,
                       files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Commit changes to repository.

        Args:
            repo_path: Path to repository
            message: Commit message
            files: List of files to commit (if None, commits all changes)

        Returns:
            Dictionary with commit results
        """
        result = {
            "repo_path": repo_path,
            "status": "success",
            "message": "Changes committed successfully"
        }

        try:
            repo = Repo(repo_path)

            # Add files
            if files:
                for file in files:
                    repo.git.add(file)
            else:
                repo.git.add('.')

            # Check if there are changes to commit
            if not repo.is_dirty():
                result["status"] = "warning"
                result["message"] = "No changes to commit"
                return result

            # Commit changes
            commit = repo.git.commit('-m', message)
            result["commit"] = commit

        except GitCommandError as e:
            result["status"] = "error"
            result["message"] = f"Git error: {str(e)}"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Unexpected error: {str(e)}"

        return result

    def push_changes(self, repo_path: str, branch_name: Optional[str] = None,
                     remote: str = "origin") -> Dict[str, Any]:
        """
        Push changes to remote repository.

        Args:
            repo_path: Path to repository
            branch_name: Branch to push (if None, uses current branch)
            remote: Remote name

        Returns:
            Dictionary with push results
        """
        result = {
            "repo_path": repo_path,
            "status": "success",
            "message": "Changes pushed successfully"
        }

        try:
            repo = Repo(repo_path)

            # Determine branch name
            if branch_name is None:
                branch_name = repo.active_branch.name

            # Push changes
            push_info = repo.git.push(remote, branch_name)
            result["branch_name"] = branch_name
            result["remote"] = remote
            result["push_info"] = push_info

        except GitCommandError as e:
            result["status"] = "error"
            result["message"] = f"Git error: {str(e)}"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Unexpected error: {str(e)}"

        return result