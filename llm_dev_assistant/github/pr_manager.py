# llm_dev_assistant/github/pr_manager.py
from typing import Dict, List, Any, Optional
from .github_client import GitHubClient
from .repo_manager import RepoManager


class PRManager:
    """Manages GitHub pull requests."""

    def __init__(self, github_client: GitHubClient, repo_manager: RepoManager):
        """
        Initialize PR manager.

        Args:
            github_client: GitHub client
            repo_manager: Repository manager
        """
        self.github_client = github_client
        self.repo_manager = repo_manager

    def create_pr_for_changes(self, repo_path: str, repo_name: str,
                              title: str, description: str,
                              branch_name: str, base_branch: str = "main") -> Dict[str, Any]:
        """
        Create PR for changes.

        Args:
            repo_path: Local repository path
            repo_name: GitHub repository name (format: 'owner/repo')
            title: PR title
            description: PR description
            branch_name: Branch with changes
            base_branch: Target branch

        Returns:
            Dictionary with PR creation results
        """
        result = {
            "repo_path": repo_path,
            "repo_name": repo_name,
            "branch_name": branch_name,
            "status": "processing"
        }

        # Push changes
        push_result = self.repo_manager.push_changes(repo_path, branch_name)
        result["push_result"] = push_result

        if push_result["status"] != "success":
            result["status"] = "error"
            result["message"] = f"Failed to push changes: {push_result['message']}"
            return result

        # Create PR
        try:
            pr_result = self.github_client.create_pull_request(
                repo_name=repo_name,
                title=title,
                body=description,
                head=branch_name,
                base=base_branch
            )

            result["status"] = "success"
            result["message"] = f"Pull request created successfully: #{pr_result['number']}"
            result["pr"] = pr_result

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Failed to create pull request: {str(e)}"

        return result

    def get_pr_status(self, repo_name: str, pr_number: int) -> Dict[str, Any]:
        """
        Get PR status.

        Args:
            repo_name: GitHub repository name (format: 'owner/repo')
            pr_number: PR number

        Returns:
            Dictionary with PR status
        """
        repo = self.github_client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        # Get review status
        reviews = list(pr.get_reviews())
        approved = any(review.state == "APPROVED" for review in reviews)
        changes_requested = any(review.state == "CHANGES_REQUESTED" for review in reviews)

        # Get checks status
        statuses = list(pr.get_statuses())
        checks_passed = all(status.state == "success" for status in statuses) if statuses else None

        return {
            "pr_number": pr_number,
            "title": pr.title,
            "state": pr.state,
            "mergeable": pr.mergeable,
            "approved": approved,
            "changes_requested": changes_requested,
            "checks_passed": checks_passed,
            "html_url": pr.html_url
        }