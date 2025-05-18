# llm_dev_assistant/github/github_client.py
import os
from typing import Dict, List, Any, Optional
from github import Github, GithubException
from git import Repo


class GitHubClient:
    """Client for GitHub API operations."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.

        Args:
            token: GitHub API token (if None, uses GITHUB_TOKEN env var)
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token must be provided or set as GITHUB_TOKEN environment variable")

        self.github = Github(self.token)

    def get_repo(self, repo_name: str):
        """
        Get a GitHub repository.

        Args:
            repo_name: Repository name (format: 'owner/repo')

        Returns:
            GitHub repository object
        """
        try:
            return self.github.get_repo(repo_name)
        except GithubException as e:
            raise ValueError(f"Error accessing repository '{repo_name}': {str(e)}")

    def get_repo_contents(self, repo_name: str, path: str = "", ref: str = "main") -> List[Dict[str, Any]]:
        """
        Get contents of a repository directory.

        Args:
            repo_name: Repository name (format: 'owner/repo')
            path: Path within repository
            ref: Branch or commit reference

        Returns:
            List of content items with metadata
        """
        repo = self.get_repo(repo_name)
        contents = repo.get_contents(path, ref=ref)

        result = []
        for content in contents:
            result.append({
                "name": content.name,
                "path": content.path,
                "type": "file" if content.type == "file" else "directory",
                "size": content.size if content.type == "file" else None,
                "download_url": content.download_url if content.type == "file" else None
            })

        return result

    def get_file_content(self, repo_name: str, file_path: str, ref: str = "main") -> str:
        """
        Get content of a specific file.

        Args:
            repo_name: Repository name (format: 'owner/repo')
            file_path: Path to file within repository
            ref: Branch or commit reference

        Returns:
            File content as string
        """
        repo = self.get_repo(repo_name)
        content = repo.get_contents(file_path, ref=ref)

        if isinstance(content, list):
            raise ValueError(f"Path '{file_path}' refers to a directory, not a file")

        return content.decoded_content.decode('utf-8')

    def get_issues(self, repo_name: str, state: str = "open") -> List[Dict[str, Any]]:
        """
        Get repository issues.

        Args:
            repo_name: Repository name (format: 'owner/repo')
            state: Issue state ('open', 'closed', or 'all')

        Returns:
            List of issues with metadata
        """
        repo = self.get_repo(repo_name)
        issues = repo.get_issues(state=state)

        result = []
        for issue in issues:
            result.append({
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "state": issue.state,
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
                "labels": [label.name for label in issue.labels]
            })

        return result

    def create_pull_request(self, repo_name: str, title: str, body: str,
                            head: str, base: str = "main") -> Dict[str, Any]:
        """
        Create a pull request.

        Args:
            repo_name: Repository name (format: 'owner/repo')
            title: PR title
            body: PR description
            head: Head branch name
            base: Base branch name

        Returns:
            Pull request metadata
        """
        repo = self.get_repo(repo_name)
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head,
            base=base
        )

        return {
            "number": pr.number,
            "html_url": pr.html_url,
            "state": pr.state,
            "title": pr.title,
            "body": pr.body
        }