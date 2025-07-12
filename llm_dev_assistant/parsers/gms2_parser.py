# llm_dev_assistant/parsers/gms2_parser.py
import os
import json
import re
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
            'all_files': [],
            'scripts': [],
            'objects': [],
            'other_assets': []
        }

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                relative_path = os.path.relpath(file_path, directory_path)

                # Add to all_files list
                result['all_files'].append({
                    'name': file,
                    'path': file_path,
                    'relative_path': relative_path,
                    'extension': file_ext,
                    'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                })

                # Handle script files (.gml)
                if file_ext in self.supported_extensions['code']:
                    script_data = self._parse_script(file_path)
                    if script_data:
                        result['scripts'].append(script_data)

                # Handle object metadata files (.yy) - specifically for objects
                elif file_ext in self.supported_extensions['metadata']:
                    asset_data = self._parse_asset_metadata(file_path)
                    if asset_data:
                        if asset_data.get('type') == 'GMObject':
                            # This is an object, parse its events
                            object_data = self._parse_object(file_path, asset_data)
                            if object_data:
                                result['objects'].append(object_data)
                        else:
                            # Other asset
                            if asset_data not in result['scripts']:
                                result['other_assets'].append(asset_data)

        return result

    def _parse_script(self, script_path: str) -> Dict[str, Any]:
        """Parse a GMS2 script file (.gml)."""
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        description = self._get_asset_description(script_path)
        content = self.get_file_content(script_path)

        # Parse functions from the script content
        functions = self._parse_functions_from_gml(content)

        # Get corresponding metadata file (.yy)
        metadata_path = os.path.splitext(script_path)[0] + '.yy'
        metadata = self._parse_asset_metadata(metadata_path) if os.path.exists(metadata_path) else {}

        return {
            'name': script_name,
            'type': 'script',
            'path': script_path,
            'description': description,
            'metadata': metadata,
            'content': content,
            'functions': functions
        }

    def _parse_object(self, metadata_path: str, asset_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a GMS2 object and its events."""
        object_name = asset_data.get('name', os.path.splitext(os.path.basename(metadata_path))[0])
        object_dir = os.path.dirname(metadata_path)

        # Look for event files in the object directory
        events = []
        if os.path.exists(object_dir):
            for file in os.listdir(object_dir):
                if file.endswith('.gml'):
                    event_path = os.path.join(object_dir, file)
                    event_data = self._parse_event(event_path)
                    if event_data:
                        events.append(event_data)

        return {
            'name': object_name,
            'type': 'object',
            'path': metadata_path,
            'description': self._get_asset_description(metadata_path),
            'metadata': asset_data,
            'events': events
        }

    def _parse_event(self, event_path: str) -> Optional[Dict[str, Any]]:
        """Parse a GMS2 event file (.gml)."""
        event_name = os.path.splitext(os.path.basename(event_path))[0]
        content = self.get_file_content(event_path)

        # Try to determine event type from filename
        event_type = self._determine_event_type(event_name)

        # Parse any functions within the event
        functions = self._parse_functions_from_gml(content)

        return {
            'name': event_name,
            'type': event_type,
            'path': event_path,
            'content': content,
            'functions': functions,
            'description': self._extract_description_from_code(content)
        }

    def _determine_event_type(self, event_name: str) -> str:
        """Determine the type of event from its filename."""
        event_name_lower = event_name.lower()

        # Common GMS2 event mappings
        event_mappings = {
            'create': 'Create Event',
            'destroy': 'Destroy Event',
            'step': 'Step Event',
            'draw': 'Draw Event',
            'collision': 'Collision Event',
            'alarm': 'Alarm Event',
            'keyboard': 'Keyboard Event',
            'mouse': 'Mouse Event',
            'other': 'Other Event'
        }

        for key, value in event_mappings.items():
            if key in event_name_lower:
                return value

        return 'Unknown Event'

    def _parse_functions_from_gml(self, content: str) -> List[Dict[str, Any]]:
        """Parse function definitions from GML content."""
        functions = []

        # GML function pattern: function name(args) or function name(args) -> return_type
        function_pattern = r'function\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(\w+))?[\s\S]*?(?=function\s+\w+\s*\(|$)'

        matches = re.finditer(function_pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            function_name = match.group(1)
            args_str = match.group(2).strip()
            return_type = match.group(3) if match.group(3) else None

            # Parse arguments
            args = []
            if args_str:
                for arg in args_str.split(','):
                    arg = arg.strip()
                    if arg:
                        args.append(arg)

            # Extract function body to look for description
            function_body = match.group(0)
            description = self._extract_function_description(function_body)

            functions.append({
                'name': function_name,
                'arguments': args,
                'return_type': return_type,
                'description': description
            })

        return functions

    def _extract_function_description(self, function_content: str) -> str:
        """Extract description from function content (comments)."""
        lines = function_content.split('\n')
        description_lines = []

        for line in lines:
            line = line.strip()
            # Look for comments at the beginning of the function
            if line.startswith('//') or line.startswith('/*') or line.startswith('*'):
                # Clean up comment markers
                clean_line = re.sub(r'^[/\*\s]*', '', line)
                clean_line = re.sub(r'\*/$', '', clean_line)
                if clean_line:
                    description_lines.append(clean_line)
            elif line and not line.startswith('function'):
                # Stop looking for comments once we hit actual code
                break

        return ' '.join(description_lines) if description_lines else "No description available"

    def _extract_description_from_code(self, content: str) -> str:
        """Extract description from code comments."""
        lines = content.split('\n')
        description_lines = []

        for line in lines[:10]:  # Only check first 10 lines
            line = line.strip()
            if line.startswith('//') or line.startswith('/*') or line.startswith('*'):
                # Clean up comment markers
                clean_line = re.sub(r'^[/\*\s]*', '', line)
                clean_line = re.sub(r'\*/$', '', clean_line)
                if clean_line:
                    description_lines.append(clean_line)

        return ' '.join(description_lines) if description_lines else "No description available"

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

        # Add objects section
        context.append("## Available Objects\n")
        for obj in parsed_data.get('objects', []):
            context.append(f"### {obj['name']}\n")
            context.append(f"**Description**: {obj['description']}\n")
            context.append(f"**Path**: {obj['path']}\n")

            if obj.get('events'):
                context.append("**Events**:\n")
                for event in obj['events']:
                    context.append(f"- {event['name']} ({event['type']}): {event['description']}\n")
            context.append("\n")

        # Add other assets section
        context.append("## Other Assets\n")
        for asset in parsed_data.get('other_assets', []):
            context.append(f"- **{asset['name']}** ({asset['type']}): {asset['description']}\n")

        return "\n".join(context)