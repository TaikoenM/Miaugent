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
            # Skip .git directories and files
            if '.git' in root:
                continue

            for file in files:
                # Skip .git files
                if '.git' in file:
                    continue

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

        # Parse objects by scanning the objects directory
        objects_dir = os.path.join(directory_path, 'objects')
        if os.path.exists(objects_dir):
            result['objects'] = self._parse_objects_directory(objects_dir)

        # Parse scripts by scanning the scripts directory
        scripts_dir = os.path.join(directory_path, 'scripts')
        if os.path.exists(scripts_dir):
            result['scripts'].extend(self._parse_scripts_directory(scripts_dir))

        # Also check for individual .gml files that might be scripts
        for file_info in result['all_files']:
            if file_info['extension'] == '.gml':
                file_path = file_info['path']
                # Only process if it's not already handled by objects or scripts directories
                if not (file_path.startswith(objects_dir) or file_path.startswith(scripts_dir)):
                    script_data = self._parse_script(file_path)
                    if script_data:
                        result['scripts'].append(script_data)

        return result

    def _parse_objects_directory(self, objects_dir: str) -> List[Dict[str, Any]]:
        """Parse all objects from the objects directory."""
        objects = []

        for obj_name in os.listdir(objects_dir):
            obj_path = os.path.join(objects_dir, obj_name)
            if os.path.isdir(obj_path):
                object_data = self._parse_object_from_directory(obj_path, obj_name)
                if object_data:
                    objects.append(object_data)

        return objects

    def _parse_scripts_directory(self, scripts_dir: str) -> List[Dict[str, Any]]:
        """Parse all scripts from the scripts directory."""
        scripts = []

        for script_name in os.listdir(scripts_dir):
            script_path = os.path.join(scripts_dir, script_name)
            if os.path.isdir(script_path):
                # Look for .gml file in the script directory
                gml_file = os.path.join(script_path, script_name + '.gml')
                if os.path.exists(gml_file):
                    script_data = self._parse_script(gml_file)
                    if script_data:
                        scripts.append(script_data)

        return scripts

    def _parse_object_from_directory(self, obj_path: str, obj_name: str) -> Optional[Dict[str, Any]]:
        """Parse a GMS2 object from its directory structure."""
        events = []

        # Look for all .gml files in the object directory
        for file in os.listdir(obj_path):
            if file.endswith('.gml'):
                event_path = os.path.join(obj_path, file)
                event_data = self._parse_event(event_path)
                if event_data:
                    events.append(event_data)

        # Get description from the first event file or use default
        description = "No description available"
        if events:
            # Look for description in any of the events
            for event in events:
                if event.get('description') and event['description'] != "No description available":
                    description = event['description']
                    break

        return {
            'name': obj_name,
            'type': 'object',
            'path': obj_path,
            'description': description,
            'events': events
        }

    def _parse_script(self, script_path: str) -> Optional[Dict[str, Any]]:
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

    def _get_asset_description(self, asset_path: str) -> str:
        """Get description for an asset from comments or description files."""
        # First try to get description from the code itself
        if asset_path.endswith('.gml'):
            content = self.get_file_content(asset_path)
            description = self._extract_description_from_gml(content)
            if description != "No description available":
                return description

        # Try to find a description file
        base_path = os.path.splitext(asset_path)[0]
        desc_file = base_path + '.txt'
        if os.path.exists(desc_file):
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except (IOError, UnicodeDecodeError):
                pass

        return "No description available"

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
            'description': self._extract_description_from_gml(content)
        }

    def _determine_event_type(self, event_name: str) -> str:
        """Determine the type of event from its filename."""
        # GameMaker Studio 2 uses specific naming conventions for events
        # Examples: Create_0, Destroy_0, Step_0, Draw_0, Draw_64, etc.

        if event_name.startswith('Create'):
            return 'Create Event'
        elif event_name.startswith('Destroy'):
            return 'Destroy Event'
        elif event_name.startswith('Step'):
            return 'Step Event'
        elif event_name.startswith('Draw_64'):
            return 'Draw GUI Event'
        elif event_name.startswith('Draw'):
            return 'Draw Event'
        elif event_name.startswith('Collision'):
            return 'Collision Event'
        elif event_name.startswith('Alarm'):
            return 'Alarm Event'
        elif event_name.startswith('Keyboard'):
            return 'Keyboard Event'
        elif event_name.startswith('Mouse'):
            return 'Mouse Event'
        elif event_name.startswith('Other'):
            return 'Other Event'
        elif event_name.startswith('CleanUp'):
            return 'Clean Up Event'
        elif event_name.startswith('User'):
            return 'User Event'
        else:
            return f'Unknown Event ({event_name})'

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
            # Look for /// @description comments or regular /// comments
            if line.startswith('/// @description'):
                desc_text = line.replace('/// @description', '').strip()
                if desc_text:
                    description_lines.append(desc_text)
            elif line.startswith('///'):
                # Regular /// comment
                desc_text = line.replace('///', '').strip()
                if desc_text and not desc_text.startswith('@'):
                    description_lines.append(desc_text)
            elif line.startswith('//'):
                # Regular // comment
                desc_text = line.replace('//', '').strip()
                if desc_text:
                    description_lines.append(desc_text)
            elif line and not line.startswith('function'):
                # Stop looking for comments once we hit actual code (unless it's the function declaration)
                break

        return ' '.join(description_lines) if description_lines else "No description available"

    def _extract_description_from_gml(self, content: str) -> str:
        """Extract description from /// @description comments in GML code."""
        lines = content.split('\n')
        description_lines = []

        for line in lines:
            line = line.strip()
            # Look for /// @description comments
            if line.startswith('/// @description'):
                # Extract the description text after the @description tag
                desc_text = line.replace('/// @description', '').strip()
                if desc_text:
                    description_lines.append(desc_text)
            elif line.startswith('///') and description_lines:
                # Continue multi-line description
                desc_text = line.replace('///', '').strip()
                if desc_text:
                    description_lines.append(desc_text)
            elif line and not line.startswith('///') and not line.startswith('//'):
                # Stop looking for descriptions once we hit actual code
                break

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
                'description': "Metadata file (description in code)",
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

        # Add objects section
        context.append("## Available Objects\n")
        for obj in parsed_data.get('objects', []):
            context.append(f"### {obj['name']}\n")
            if obj['description'] != "No description available":
                context.append(f"**Description**: {obj['description']}\n")
            context.append(f"**Path**: {obj['path']}\n")

            if obj.get('events'):
                context.append("**Events**:\n")
                for event in obj['events']:
                    context.append(f"- {event['name']} ({event['type']})")
                    if event['description'] != "No description available":
                        context.append(f": {event['description']}")
                    context.append("\n")
            context.append("\n")

        # Add scripts section
        context.append("## Available Scripts\n")
        for script in parsed_data.get('scripts', []):
            context.append(f"### {script['name']}\n")
            if script['description'] != "No description available":
                context.append(f"**Description**: {script['description']}\n")
            context.append(f"**Path**: {script['path']}\n")

            # Include first few lines of code as a preview
            code_preview = "\n".join(script['content'].split('\n')[:10])
            if len(script['content'].split('\n')) > 10:
                code_preview += "\n// ... (more code continues)"

            context.append("```gml\n" + code_preview + "\n```\n")

        return "\n".join(context)