# llm_dev_assistant/gui_main.py
"""
GUI entry point for LLM Development Assistant.

Run with: python -m llm_dev_assistant.gui_main
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_dev_assistant.gui.main_window import main

if __name__ == "__main__":
    main()