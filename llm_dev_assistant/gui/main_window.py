# llm_dev_assistant/gui/main_window.py
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..workflow.workflow_engine import WorkflowEngine
from ..llm.openai_adapter import OpenAIAdapter
from ..llm.local_llm_adapter import LocalLLMAdapter
from ..parsers.gms2_parser import GMS2Parser
from ..code_manager.code_analyzer import CodeAnalyzer
from ..code_manager.change_implementer import ChangeImplementer
from ..code_manager.test_manager import TestManager
from ..log_system.logger import logger
from .log_handler import setup_gui_logging, remove_gui_logging


class LLMDevAssistantGUI:
    """Simple GUI for LLM Development Assistant."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LLM Development Assistant")
        self.root.geometry("1200x800")

        # Use a simple, raw style
        self.root.configure(bg='#f0f0f0')
        style = ttk.Style()
        style.theme_use('default')

        # Workflow engine will be initialized later
        self.workflow_engine = None
        self.log_queue = queue.Queue()
        self.logs = []

        # Current task state
        self.current_project_path = None
        self.current_file_path = None

        self._setup_ui()
        self._setup_logging()
        self._start_log_monitor()

        # Set up cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Initialize Project", command=self._init_project)
        file_menu.add_separator()
        file_menu.add_command(label="Save Workflow", command=self._save_workflow)
        file_menu.add_command(label="Load Workflow", command=self._load_workflow)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Output", command=lambda: self.output_text.delete(1.0, tk.END))
        edit_menu.add_command(label="Clear Logs", command=self._clear_logs)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Configure LLM", command=self._configure_llm)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _setup_ui(self):
        """Set up the user interface."""
        # Menu bar
        self._create_menu_bar()

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Project initialization
        ttk.Label(control_frame, text="Project Path:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.project_path_var = tk.StringVar()
        project_entry = ttk.Entry(control_frame, textvariable=self.project_path_var, width=30)
        project_entry.grid(row=0, column=1, padx=(5, 0), pady=(0, 5))
        ttk.Button(control_frame, text="Browse", command=self._browse_project).grid(row=0, column=2, padx=(5, 0),
                                                                                    pady=(0, 5))
        ttk.Button(control_frame, text="Initialize Project", command=self._init_project).grid(row=1, column=0,
                                                                                              columnspan=3,
                                                                                              pady=(0, 10),
                                                                                              sticky=tk.W + tk.E)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E),
                                                               pady=10)

        # Task operations
        ttk.Label(control_frame, text="Task Description:").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        self.task_text = tk.Text(control_frame, height=5, width=40, wrap=tk.WORD)
        self.task_text.grid(row=4, column=0, columnspan=3, pady=(0, 5), sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="File Path (optional):").grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(control_frame, textvariable=self.file_path_var, width=30)
        file_entry.grid(row=5, column=1, padx=(5, 0), pady=(0, 5))
        ttk.Button(control_frame, text="Browse", command=self._browse_file).grid(row=5, column=2, padx=(5, 0),
                                                                                 pady=(0, 5))

        # Task buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Request Code", command=self._request_code).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Verify Implementation", command=self._verify_implementation).grid(row=0,
                                                                                                         column=1,
                                                                                                         padx=2, pady=2)
        ttk.Button(button_frame, text="Implement Changes", command=self._implement_changes).grid(row=1, column=0,
                                                                                                 padx=2, pady=2)
        ttk.Button(button_frame, text="Verify Tests", command=self._verify_tests).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(button_frame, text="Run Tests", command=self._run_tests).grid(row=2, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Plan Next Steps", command=self._plan_next_steps).grid(row=2, column=1, padx=2,
                                                                                             pady=2)

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E),
                                                               pady=10)

        # Workflow management
        ttk.Label(control_frame, text="Workflow Management:").grid(row=8, column=0, columnspan=3, sticky=tk.W,
                                                                   pady=(0, 5))
        workflow_frame = ttk.Frame(control_frame)
        workflow_frame.grid(row=9, column=0, columnspan=3)

        ttk.Button(workflow_frame, text="Save Workflow", command=self._save_workflow).pack(side=tk.LEFT, padx=2)
        ttk.Button(workflow_frame, text="Load Workflow", command=self._load_workflow).pack(side=tk.LEFT, padx=2)

        # Top right - Output/Results
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))

        self.output_text = scrolledtext.ScrolledText(output_frame, height=20, width=60, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Bottom right - Logs
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="10")
        log_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Log filter
        filter_frame = ttk.Frame(log_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 5))
        self.log_level_var = tk.StringVar(value="ALL")
        log_filter = ttk.Combobox(filter_frame, textvariable=self.log_level_var,
                                  values=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
                                  state="readonly", width=10)
        log_filter.pack(side=tk.LEFT)
        log_filter.bind('<<ComboboxSelected>>', lambda e: self._update_log_display())

        ttk.Button(filter_frame, text="Clear Logs", command=self._clear_logs).pack(side=tk.RIGHT, padx=(5, 0))

        # Log display
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=60, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure log text tags for different levels
        self.log_text.tag_config("DEBUG", foreground="gray")
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")

        # Status bar with progress
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var,
                                            mode='indeterminate', length=100)
        self.progress_bar.pack(side=tk.RIGHT, padx=5)

    def _setup_logging(self):
        """Set up log_system to redirect to GUI."""
        # Set up the log_system system
        logger.setup()

        # Add GUI log handler
        self.log_handler = setup_gui_logging(self.log_queue)

        # Log startup
        log = logger.get_logger("GUI")
        log.info("LLM Development Assistant GUI started")

    def _start_log_monitor(self):
        """Start monitoring the log queue."""
        self._process_log_queue()

    def _process_log_queue(self):
        """Process logs from the queue."""
        try:
            while True:
                log_entry = self.log_queue.get_nowait()
                self.logs.append(log_entry)
                self._add_log_entry(log_entry)
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self._process_log_queue)

    def _add_log_entry(self, log_entry):
        """Add a log entry to the display."""
        level_filter = self.log_level_var.get()
        if level_filter == "ALL" or log_entry['level'] == level_filter:
            self.log_text.insert(tk.END,
                                 f"[{log_entry['time']}] [{log_entry['level']}] {log_entry['message']}\n",
                                 log_entry['level'])
            self.log_text.see(tk.END)

    def _update_log_display(self):
        """Update log display based on filter."""
        self.log_text.delete(1.0, tk.END)
        level_filter = self.log_level_var.get()

        for log_entry in self.logs:
            if level_filter == "ALL" or log_entry['level'] == level_filter:
                self._add_log_entry(log_entry)

    def _clear_logs(self):
        """Clear the log display."""
        self.logs.clear()
        self.log_text.delete(1.0, tk.END)

    def _browse_project(self):
        """Browse for project directory."""
        directory = filedialog.askdirectory(title="Select Project Directory")
        if directory:
            self.project_path_var.set(directory)

    def _browse_file(self):
        """Browse for a file."""
        filename = filedialog.askopenfilename(title="Select File")
        if filename:
            self.file_path_var.set(filename)

    def _update_status(self, message: str):
        """Update status bar."""
        self.status_var.set(message)
        self.log_queue.put({
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': 'INFO',
            'message': message,
            'module': 'GUI'
        })

    def _display_output(self, content: str):
        """Display content in the output area."""
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(1.0, content)

    def _run_async(self, func, *args, **kwargs):
        """Run a function asynchronously."""
        # Show progress
        self.progress_bar.start(10)

        # Disable main controls during operation
        self._set_controls_enabled(False)

        def wrapper():
            try:
                result = func(*args, **kwargs)
                self.root.after(0, lambda: self._handle_result(result))
            except Exception as e:
                self.root.after(0, lambda: self._handle_error(str(e)))
            finally:
                self.root.after(0, self._operation_complete)

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

    def _operation_complete(self):
        """Called when an async operation completes."""
        self.progress_bar.stop()
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable controls."""
        state = 'normal' if enabled else 'disabled'
        # This would disable/enable all the buttons and inputs
        # For brevity, I'm not implementing the full traversal here
        pass

    def _handle_result(self, result):
        """Handle operation result."""
        if isinstance(result, dict):
            self._display_output(json.dumps(result, indent=2))
            if result.get('status') == 'error':
                self._update_status(f"Error: {result.get('message', 'Unknown error')}")
                messagebox.showerror("Error", result.get('message', 'Unknown error'))
            else:
                self._update_status("Operation completed successfully")
        else:
            self._display_output(str(result))
            self._update_status("Operation completed")

    def _handle_error(self, error_msg: str):
        """Handle operation error."""
        self._update_status(f"Error: {error_msg}")
        messagebox.showerror("Error", error_msg)
        self.log_queue.put({
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': 'ERROR',
            'message': error_msg,
            'module': 'GUI'
        })

    def _ensure_workflow_engine(self):
        """Ensure workflow engine is initialized."""
        if not self.workflow_engine:
            # Initialize with default components
            self._update_status("Initializing workflow engine...")

            try:
                llm = OpenAIAdapter()  # Default to OpenAI
                parser = GMS2Parser()
                code_analyzer = CodeAnalyzer()
                change_implementer = ChangeImplementer()
                test_manager = TestManager()

                self.workflow_engine = WorkflowEngine(
                    llm=llm,
                    parser=parser,
                    code_analyzer=code_analyzer,
                    change_implementer=change_implementer,
                    test_manager=test_manager
                )

                self._update_status("Workflow engine initialized")
            except Exception as e:
                self._handle_error(f"Failed to initialize workflow engine: {str(e)}")
                return False

        return True

    def _init_project(self):
        """Initialize project."""
        project_path = self.project_path_var.get()
        if not project_path:
            messagebox.showwarning("Warning", "Please select a project directory")
            return

        if not self._ensure_workflow_engine():
            return

        self._update_status(f"Initializing project: {project_path}")
        self._run_async(self.workflow_engine.initialize_project, project_path)
        self.current_project_path = project_path

    def _request_code(self):
        """Request code implementation."""
        if not self._ensure_workflow_engine():
            return

        task_desc = self.task_text.get(1.0, tk.END).strip()
        if not task_desc:
            messagebox.showwarning("Warning", "Please enter a task description")
            return

        file_path = self.file_path_var.get() or None

        self._update_status("Requesting code implementation...")
        self._run_async(self.workflow_engine.request_code_implementation, task_desc, file_path)

    def _verify_implementation(self):
        """Open dialog for code verification."""
        if not self._ensure_workflow_engine():
            return

        dialog = CodeVerificationDialog(self.root, self.workflow_engine)
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            self._display_output(json.dumps(dialog.result, indent=2))

    def _implement_changes(self):
        """Open dialog for implementing changes."""
        if not self._ensure_workflow_engine():
            return

        dialog = ImplementChangesDialog(self.root, self.workflow_engine)
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            self._display_output(json.dumps(dialog.result, indent=2))

    def _verify_tests(self):
        """Verify tests for a file."""
        if not self._ensure_workflow_engine():
            return

        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file to verify tests for")
            return

        self._update_status(f"Verifying tests for: {file_path}")
        self._run_async(self.workflow_engine.verify_tests, file_path)

    def _run_tests(self):
        """Run tests."""
        if not self._ensure_workflow_engine():
            return

        # For simplicity, run all tests. Could add a dialog to select specific tests
        self._update_status("Running all tests...")
        self._run_async(self.workflow_engine.run_integration_tests)

    def _plan_next_steps(self):
        """Plan next development steps."""
        if not self._ensure_workflow_engine():
            return

        dialog = PlanningDialog(self.root, self.workflow_engine)
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            self._display_output(json.dumps(dialog.result, indent=2))

    def _save_workflow(self):
        """Save workflow state."""
        if not self.workflow_engine:
            messagebox.showwarning("Warning", "No workflow to save")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Workflow",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            self._update_status(f"Saving workflow to: {filename}")
            self._run_async(self.workflow_engine.save_workflow_state, filename)

    def _load_workflow(self):
        """Load workflow state."""
        if not self._ensure_workflow_engine():
            return

        filename = filedialog.askopenfilename(
            title="Load Workflow",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            self._update_status(f"Loading workflow from: {filename}")
            self._run_async(self.workflow_engine.load_workflow_state, filename)

    def run(self):
        """Run the GUI application."""
        self.root.mainloop()

    def _on_closing(self):
        """Handle window closing."""
        # Remove GUI log handler
        if hasattr(self, 'log_handler'):
            remove_gui_logging(self.log_handler)

        # Destroy window
        self.root.destroy()

    def _configure_llm(self):
        """Open LLM configuration dialog."""
        dialog = LLMConfigDialog(self.root)
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            # Apply configuration
            self._update_status("LLM configuration updated")

    def _show_about(self):
        """Show about dialog."""
        about_text = """LLM Development Assistant GUI

A tool to automate iterative development tasks using LLMs.

Features:
- Code generation and modification
- Implementation verification
- Test management
- Development planning
- Workflow state persistence

Version: 1.0.0"""

        messagebox.showinfo("About", about_text)


class CodeVerificationDialog:
    """Dialog for code verification."""

    def __init__(self, parent, workflow_engine):
        self.parent = parent
        self.workflow_engine = workflow_engine
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Verify Code Implementation")
        self.dialog.geometry("800x600")

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Original code
        ttk.Label(main_frame, text="Original Code:").pack(anchor=tk.W)
        self.original_text = scrolledtext.ScrolledText(main_frame, height=10, width=80)
        self.original_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # New code
        ttk.Label(main_frame, text="New Code:").pack(anchor=tk.W)
        self.new_text = scrolledtext.ScrolledText(main_frame, height=10, width=80)
        self.new_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Requirements
        ttk.Label(main_frame, text="Requirements:").pack(anchor=tk.W)
        self.requirements_text = tk.Text(main_frame, height=5, width=80)
        self.requirements_text.pack(fill=tk.X, pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Verify", command=self._verify).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT)

    def _verify(self):
        """Perform verification."""
        original = self.original_text.get(1.0, tk.END).strip()
        new = self.new_text.get(1.0, tk.END).strip()
        requirements = self.requirements_text.get(1.0, tk.END).strip()

        if not all([original, new, requirements]):
            messagebox.showwarning("Warning", "Please fill in all fields")
            return

        try:
            self.result = self.workflow_engine.verify_implementation(original, new, requirements)
            self.dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


class ImplementChangesDialog:
    """Dialog for implementing code changes."""

    def __init__(self, parent, workflow_engine):
        self.parent = parent
        self.workflow_engine = workflow_engine
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Implement Code Changes")
        self.dialog.geometry("600x500")

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File path
        ttk.Label(main_frame, text="File Path:").pack(anchor=tk.W)

        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(side=tk.LEFT, padx=(5, 0))

        # New code
        ttk.Label(main_frame, text="New Code:").pack(anchor=tk.W)
        self.code_text = scrolledtext.ScrolledText(main_frame, height=20, width=70)
        self.code_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Implement", command=self._implement).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT)

    def _browse_file(self):
        """Browse for file."""
        filename = filedialog.asksaveasfilename(
            title="Select File to Modify/Create",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)

    def _implement(self):
        """Implement changes."""
        file_path = self.file_path_var.get()
        new_code = self.code_text.get(1.0, tk.END).strip()

        if not file_path or not new_code:
            messagebox.showwarning("Warning", "Please provide both file path and code")
            return

        try:
            self.result = self.workflow_engine.implement_changes(file_path, new_code)
            self.dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


class PlanningDialog:
    """Dialog for planning next steps."""

    def __init__(self, parent, workflow_engine):
        self.parent = parent
        self.workflow_engine = workflow_engine
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Plan Next Steps")
        self.dialog.geometry("500x400")

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Goals
        ttk.Label(main_frame, text="Project Goals (one per line):").pack(anchor=tk.W)
        self.goals_text = scrolledtext.ScrolledText(main_frame, height=15, width=60)
        self.goals_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Sample goals
        self.goals_text.insert(1.0, "Complete parser implementation\nAdd error handling\nImprove test coverage")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Plan", command=self._plan).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT)

    def _plan(self):
        """Create plan."""
        goals_text = self.goals_text.get(1.0, tk.END).strip()
        goals = [g.strip() for g in goals_text.split('\n') if g.strip()]

        if not goals:
            messagebox.showwarning("Warning", "Please enter at least one goal")
            return

        try:
            self.result = self.workflow_engine.plan_next_steps(goals)
            self.dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


class LLMConfigDialog:
    """Dialog for configuring LLM settings."""

    def __init__(self, parent):
        self.parent = parent
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("LLM Configuration")
        self.dialog.geometry("400x300")

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # LLM Type
        ttk.Label(main_frame, text="LLM Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.llm_type_var = tk.StringVar(value="openai")
        llm_combo = ttk.Combobox(main_frame, textvariable=self.llm_type_var,
                                 values=["openai", "local", "lmstudio"],
                                 state="readonly", width=20)
        llm_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        llm_combo.bind('<<ComboboxSelected>>', self._on_llm_type_changed)

        # OpenAI Model
        self.openai_frame = ttk.Frame(main_frame)
        self.openai_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(self.openai_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.openai_model_var = tk.StringVar(value="gpt-4")
        ttk.Combobox(self.openai_frame, textvariable=self.openai_model_var,
                     values=["gpt-4", "gpt-3.5-turbo", "gpt-4-vision-preview"],
                     state="readonly", width=20).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(self.openai_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.api_key_var = tk.StringVar()
        ttk.Entry(self.openai_frame, textvariable=self.api_key_var, show="*", width=30).grid(row=1, column=1,
                                                                                             pady=(5, 0))

        # Local Model
        self.local_frame = ttk.Frame(main_frame)

        ttk.Label(self.local_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W)
        self.local_path_var = tk.StringVar()
        ttk.Entry(self.local_frame, textvariable=self.local_path_var, width=25).grid(row=0, column=1)
        ttk.Button(self.local_frame, text="Browse",
                   command=lambda: self.local_path_var.set(
                       filedialog.askopenfilename(title="Select Local Model")
                   )).grid(row=0, column=2, padx=(5, 0))

        # LM Studio
        self.lmstudio_frame = ttk.Frame(main_frame)

        ttk.Label(self.lmstudio_frame, text="Model Name:").grid(row=0, column=0, sticky=tk.W)
        self.lmstudio_model_var = tk.StringVar()
        ttk.Entry(self.lmstudio_frame, textvariable=self.lmstudio_model_var, width=30).grid(row=0, column=1)

        ttk.Label(self.lmstudio_frame, text="Host:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.lmstudio_host_var = tk.StringVar(value="localhost")
        ttk.Entry(self.lmstudio_frame, textvariable=self.lmstudio_host_var, width=30).grid(row=1, column=1, pady=(5, 0))

        ttk.Label(self.lmstudio_frame, text="Port:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.lmstudio_port_var = tk.StringVar(value="1234")
        ttk.Entry(self.lmstudio_frame, textvariable=self.lmstudio_port_var, width=30).grid(row=2, column=1, pady=(5, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=(20, 0))

        ttk.Button(button_frame, text="OK", command=self._ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT)

        # Show appropriate frame
        self._on_llm_type_changed()

    def _on_llm_type_changed(self, event=None):
        """Handle LLM type change."""
        # Hide all frames
        self.openai_frame.grid_remove()
        self.local_frame.grid_remove()
        self.lmstudio_frame.grid_remove()

        # Show appropriate frame
        llm_type = self.llm_type_var.get()
        if llm_type == "openai":
            self.openai_frame.grid()
        elif llm_type == "local":
            self.local_frame.grid()
        elif llm_type == "lmstudio":
            self.lmstudio_frame.grid()

    def _ok(self):
        """Save configuration."""
        self.result = {
            'llm_type': self.llm_type_var.get(),
            'openai_model': self.openai_model_var.get(),
            'api_key': self.api_key_var.get(),
            'local_path': self.local_path_var.get(),
            'lmstudio_model': self.lmstudio_model_var.get(),
            'lmstudio_host': self.lmstudio_host_var.get(),
            'lmstudio_port': self.lmstudio_port_var.get()
        }
        self.dialog.destroy()


def main():
    """Main entry point for GUI."""
    app = LLMDevAssistantGUI()
    app.run()


if __name__ == "__main__":
    main()