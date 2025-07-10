from llm_dev_assistant.__main__ import create_workflow_engine
from llm_dev_assistant.workflow.workflow_engine import WorkflowEngine
from llm_dev_assistant.workflow.collaborative_workflow import CollaborativeLLMWorkflow

# Initialize your existing components
workflow_engine = create_workflow_engine("openai")  # Your existing function

# Create the collaborative workflow
collaborative_workflow = CollaborativeLLMWorkflow(workflow_engine)

# Start a feature development
result = collaborative_workflow.start_feature_development(
    feature_name="Game Map Editor",
    feature_description="Create a game map editor interface that allows for placing, editing, and deleting map elements. The editor should support saving/loading maps, zooming, panning, and layer management."
)

# Execute the full workflow automatically
final_result = collaborative_workflow.execute_full_workflow()

# Or execute steps manually
while True:
    next_step = collaborative_workflow.continue_workflow()
    if next_step.get("status") == "completed":
        break

    # You could pause here for human input if needed
    print(f"Completed phase: {next_step.get('phase')}")