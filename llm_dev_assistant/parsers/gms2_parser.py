# llm_dev_assistant/parsers/gms2_parser.py
import os
import json
from typing import Dict, List, Any, Optional
from .parser_interface import ParserInterface


class GMS2Parser(ParserInterface):
    """Parser for Game Maker Studio 2 projects."""

    def __init__(self):
        self.supported_extensions = {
            'code': ['.gml'],
            'metadata': ['.yy'],
            'description': ['.txt']
        }

    def parse_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Parse a Game Maker Studio 2 project directory.

        Args:
            directory_path: Path to the GMS2 project

        Returns:
            Dictionary containing structured project data
        """
        result = {
            'scripts': [],
            'other_assets': []
        }

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # Handle script files (.gml)
                if file_ext in self.supported_extensions['code']:
                    script_data = self._parse_script(file_path)
                    if script_data:
                        result['scripts'].append(script_data)

                # Note other asset files
                elif file_ext in self.supported_extensions['metadata']:
                    asset_data = self._parse_asset_metadata(file_path)
                    if asset_data and asset_data not in result['scripts']:
                        result['other_assets'].append(asset_data)

        return result

    def _parse_script(self, script_path: str) -> Dict[str, Any]:
        """Parse a GMS2 script file (.gml)."""
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        description = self._get_asset_description(script_path)

        # Get corresponding metadata file (.yy)
        metadata_path = os.path.splitext(script_path)[0] + '.yy'
        metadata = self._parse_asset_metadata(metadata_path) if os.path.exists(metadata_path) else {}

        return {
            'name': script_name,
            'type': 'script',
            'path': script_path,
            'description': description,
            'metadata': metadata,
            'content': self.get_file_content(script_path)
        }

    def _parse_asset_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Parse a GMS2 metadata file (.yy)."""
        if not os.path.exists(metadata_path):
            return {}

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            asset_name = os.path.splitext(os.path.basename(metadata_path))[0]
            asset_type = metadata.get('resourceType', 'unknown')

            return {
                'name': asset_name,
                'type': asset_type,
                'path': metadata_path,
                'description': self._get_asset_description(metadata_path),
                'metadata': metadata
            }
        except (json.JSONDecodeError, UnicodeDecodeError, IOError):
            return {
                'name': os.path.splitext(os.path.basename(metadata_path))[0],
                'type': 'unknown',
                'path': metadata_path,
                'description': 'Error parsing metadata file',
                'metadata': {}
            }

    def _get_asset_description(self, asset_path: str) -> str:
        """Get description from an accompanying .txt file."""
        description_path = os.path.splitext(asset_path)[0] + '.txt'

        if os.path.exists(description_path):
            try:
                with open(description_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except (IOError, UnicodeDecodeError):
                return "Error reading description file"

        return "Missing description"

    def get_file_content(self, file_path: str) -> str:
        """Get the content of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, UnicodeDecodeError):
            return "Error reading file"

    def generate_context(self, parsed_data: Dict[str, Any]) -> str:
        """
        Generate a contextual description for LLM prompting.

        Args:
            parsed_data: Dictionary with parsed project data

        Returns:
            Formatted context string for LLM
        """
        context = ["# Game Maker Studio 2 Project Context\n"]

        # Add scripts section
        context.append("## Available Scripts\n")
        for script in parsed_data['scripts']:
            context.append(f"### {script['name']}\n")
            context.append(f"**Description**: {script['description']}\n")
            context.append(f"**Path**: {script['path']}\n")

            # Include first few lines of code as a preview
            code_preview = "\n".join(script['content'].split('\n')[:10])
            if len(script['content'].split('\n')) > 10:
                code_preview += "\n// ... (more code continues)"

            context.append("```gml\n" + code_preview + "\n```\n")

        # Add other assets section
        context.append("## Other Assets\n")
        for asset in parsed_data['other_assets']:
            context.append(f"- **{asset['name']}** ({asset['type']}): {asset['description']}\n")

        return "\n".join(context)