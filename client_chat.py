# Standard Library Imports
import numpy as np 
import pandas as pd 
import json 
import uuid 
import os 
from datetime import datetime
import pytz 
from dotenv import load_dotenv

# Scikit-learn Imports
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# LangChain and LangGraph Imports
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Pydantic Imports for AgentState definition
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Any, Dict

# --- CHANGED: Local Tool Imports from all model scripts ---
from regression_models.linear_reg_tools import run_linear_regression_workflow
from regression_models.decision_tree_reg_tools import run_decision_tree_workflow
from regression_models.random_forest_reg_tools import run_random_forest_workflow
from regression_models.xg_boost_reg_tools import run_xgboost_workflow
from regression_models.svr_tools import run_svr_workflow
from regression_models.svr_tools import print_df_head # Can be imported from any, it's a shared utility

# Import global stores to allow clearing them
from regression_models.linear_reg_tools import _data_store, _pipeline_store, _X_train_store, _X_test_store, _y_train_store, _y_test_store


# --- LangGraph Checkpointer ---
memory = MemorySaver()

# --- AgentState Definition (Unified with parameter memory) ---
class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[BaseMessage] = Field(default_factory=list)

    # --- NEW: Field to track the algorithm used ---
    algorithm_used: Optional[str] = None
    
    # Data/Pipeline Keys
    data_key: Optional[str] = None
    X_train_key: Optional[str] = None
    X_test_key: Optional[str] = None
    y_train_key: Optional[str] = None
    y_test_key: Optional[str] = None
    pipeline_key: Optional[str] = None

    # Core Workflow Parameters
    user_query : Optional[str] = None
    path: Optional[str] = None
    target_column: Optional[str] = None
    ignore_columns: Optional[List[str]] = None
    feature_engineering_prompt: Optional[str] = None
    
    # --- SVR Hyperparameters (example, add others as needed) ---
    kernel: Optional[str] = 'rbf'
    C: Optional[float] = 1.0
    gamma: Optional[str] = 'scale'
    
    # Other model params
    numeric_imputer_type: Optional[str] = "simple"
    numeric_imputer_strategy: Optional[str] = "mean"
    categorical_imputer_type: Optional[str] = "simple"
    categorical_imputer_strategy: Optional[str] = "constant"
    numeric_scaler_type: Optional[str] = "standard"
    categorical_encoder_type: Optional[str] = "onehot"

    # Other computed/result fields
    best_hyperparameters: Optional[Dict[str, Any]] = None
    numeric_cols: Optional[List[str]] = Field(default_factory=list)
    categorical_cols: Optional[List[str]] = Field(default_factory=list)
    newly_added_columns: Optional[List[str]] = Field(default_factory=list)
    evaluation_metrics: Optional[Dict[str, float]] = Field(default_factory=dict)
    workflow_summary: Optional[Dict[str, Any]] = Field(default_factory=dict)


# --- Chat Model and LLM Setup ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- CHANGED: Unified list of all workflow tools ---
tools_for_llm = [
    run_linear_regression_workflow,
    run_decision_tree_workflow,
    run_random_forest_workflow,
    run_xgboost_workflow,
    run_svr_workflow,
    print_df_head
]
workflow_tool_names = [tool.__name__ for tool in tools_for_llm if "workflow" in tool.__name__]


def get_current_time_and_location():
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    current_time_str = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    current_location_str = "Hyderabad, Telangana, India"
    return current_time_str, current_location_str


# --- `chat_model` Node Definition ---
def chat_model(state: AgentState) -> dict:
    print("--- In chat_model node ---")
    
    state_updates_to_return = state.model_dump(exclude={"messages"})
    message_to_add_to_history = None

    if state.messages and isinstance(state.messages[-1], HumanMessage):
        state_updates_to_return['user_query'] = state.messages[-1].content

    if state.messages and isinstance(state.messages[-1], ToolMessage):
        last_tool_message = state.messages[-1]
        tool_raw_content = last_tool_message.content
        print(f"Processing ToolMessage from: {last_tool_message.name}")
        
        tool_output_dict = {}
        try:
            tool_output_dict = json.loads(tool_raw_content)
        except json.JSONDecodeError:
            message_to_add_to_history = AIMessage(
                content=f"Error: Tool '{last_tool_message.name}' returned unparseable JSON output: {tool_raw_content}"
            )
        
        if tool_output_dict and message_to_add_to_history is None:
            print("Applying tool output to state_updates_to_return.")
            for key, value in tool_output_dict.items():
                if key in state_updates_to_return:
                    state_updates_to_return[key] = value

            # --- CHANGED: Dynamically handle any workflow tool ---
            if last_tool_message.name in workflow_tool_names:
                # Extract algorithm name from the tool name for state update
                algo_name = last_tool_message.name.replace("run_", "").replace("_workflow", "").replace("_", " ").title()
                state_updates_to_return['algorithm_used'] = algo_name

                workflow_summary = tool_output_dict.get("workflow_summary", {})
                workflow_status = workflow_summary.get("status")
                metrics = tool_output_dict.get("evaluation_metrics", {})
                
                if workflow_status == "completed successfully":
                    # Build response dynamically
                    response_parts = [
                        "Great news! The ML workflow completed successfully. Here's a summary:\n",
                        f"  - **Algorithm Used**: `{algo_name}`",
                        f"  - **Dataset**: `{tool_output_dict.get('path', 'N/A')}`",
                        f"  - **Target Column**: `{tool_output_dict.get('target_column', 'N/A')}`"
                    ]
                    
                    # --- ADD THIS BLOCK TO DISPLAY THE PARAMETERS ---
                    best_params = tool_output_dict.get("best_hyperparameters")
                    if best_params:
                        # This part cleans up names like 'model__alpha' to just 'alpha' for display
                        params_str = ", ".join([f"'{k.split('__')[-1]}': {v}" for k, v in best_params.items()])
                        response_parts.append(f"  - **Best Hyperparameters**: {{{params_str}}}")

                    
                    if tool_output_dict.get("feature_engineering_prompt"):
                        response_parts.append(f"  - **Feature Engineering**: Performed with prompt: \"_{tool_output_dict.get('feature_engineering_prompt')}_\"")
                        new_cols = tool_output_dict.get('newly_added_columns', [])
                        if new_cols:
                            response_parts.append(f"    - New columns created: `{', '.join(new_cols)}`")

                    response_parts.extend([
                        f"  - **Numeric Scaler**: `{tool_output_dict.get('numeric_scaler_type', 'N/A')}`",
                        "\n**Evaluation Metrics:**",
                        f"  - **R-squared**: `{metrics.get('r2', 0):.3f}`",
                        f"  - **MAE**: `{metrics.get('mae', 0):.3f}`",
                        f"  - **RMSE**: `{metrics.get('rmse', 0):.3f}`",
                        "\nHow else can I assist you? You can ask to change a parameter or try a different algorithm like 'now use random forest'."
                    ])
                    response_text = "\n".join(response_parts)

                elif workflow_status == "failed":
                    error_message = workflow_summary.get("error", "An unknown error occurred.")
                    response_text = f"Apologies, the ML workflow failed. Error: {error_message}\nPlease check your input or try again."
                
                message_to_add_to_history = AIMessage(content=response_text)

            elif last_tool_message.name == "print_df_head":
                if tool_output_dict.get("df_head_printed"):
                    df_name = tool_output_dict.get("df_name", "DataFrame")
                    sample_head_str = json.dumps(tool_output_dict.get("df_sample_head", []), indent=2)
                    response_text = f"Here is the head of the {df_name}:\n```json\n{sample_head_str}\n```\nWhat else would you like to know?"
                else:
                    response_text = "I couldn't find the specified DataFrame to print its head."
                message_to_add_to_history = AIMessage(content=response_text)
                
    if message_to_add_to_history is None:
        agent_executor = llm.bind_tools(tools_for_llm)
        current_time, current_location = get_current_time_and_location()
        
        # --- CHANGED: Updated system prompt for multiple models ---
        dynamic_system_message_content = (
            "You are an AI assistant specialized in machine learning workflows. "
            "Your goal is to help users run regression models. You have access to several workflow tools:\n"
            "- `run_linear_regression_workflow`\n"
            "- `run_decision_tree_workflow`\n"
            "- `run_random_forest_workflow`\n"
            "- `run_xgboost_workflow`\n"
            "- `run_svr_workflow`\n\n"
            "**Instructions:**\n"
            "1. **Select the Right Tool:** Based on the user's prompt (e.g., 'use random forest', 'try svr'), select the corresponding workflow tool. If no model is specified, default to `run_linear_regression_workflow`.\n"
            "2. **Reuse Current State:** For any parameter not in the user's current prompt, you MUST reuse the value from the `Current Agent State` provided below.\n"
            "3. **Feature Engineering:** If the user asks to create new features, use the `feature_engineering_prompt` argument in your chosen workflow tool.\n"
            "4. **Handle Errors:** If a tool fails, inform the user clearly and ask for corrected information.\n"
            "5. **Summarize Results:** After a workflow tool completes successfully, your next response MUST be a clear summary. DO NOT call another tool in that turn.\n"
            f"\nCurrent date and time: {current_time} in {current_location}.\n\n"
            "**Current Agent State (For your use, do not show to user unless asked):**\n"
            f"Algorithm Used: {state_updates_to_return.get('algorithm_used')}\n"
            f"Path: {state_updates_to_return.get('path')}\n"
            f"Target Column: {state_updates_to_return.get('target_column')}\n"
            f"Numeric Scaler: {state_updates_to_return.get('numeric_scaler_type')}\n"
            f"Evaluation Metrics: {state_updates_to_return.get('evaluation_metrics', 'N/A')}\n"
            f"Last Workflow Status: {state_updates_to_return.get('workflow_summary', {}).get('status', 'N/A')}"
            f"User Query: {state_updates_to_return.get('user_query', 'Run the regression model')}\n"
        )
        dynamic_system_message = SystemMessage(content=dynamic_system_message_content)
        llm_input_messages = [dynamic_system_message] + state.messages
        message_to_add_to_history = agent_executor.invoke(llm_input_messages)
        
    print(f"Message to append to history: {message_to_add_to_history}")
    state_updates_to_return["messages"] = add_messages(state.messages, message_to_add_to_history)
    print("--- Returning from chat_model node ---")
    return state_updates_to_return

# --- LangGraph Graph Definition ---
graph = StateGraph(AgentState)
graph.add_node("chat_model", chat_model)
graph.add_node("tools", ToolNode(tools_for_llm))
graph.add_edge(START, "chat_model")
graph.add_conditional_edges("chat_model", tools_condition)
graph.add_edge("tools", "chat_model")
graph.add_edge("chat_model", END) # Allows graph to finish if no tool is called
builder = graph.compile(checkpointer=memory)

# --- Main Execution Loop ---
config = {"configurable": {"thread_id": "1"}}

def clear_global_stores():
    _data_store.clear()
    _pipeline_store.clear()
    _X_train_store.clear()
    _X_test_store.clear()
    _y_train_store.clear()
    _y_test_store.clear()
    print("\n--- Global in-memory data stores cleared ---")

print("ML Workflow Agent: Type 'exit' to quit. Use 'reset' to clear conversation memory.")
while True:
    prompt = input("Enter your message: ")

    if prompt.lower() == "exit":
        break
    elif prompt.lower() == "reset":
        clear_global_stores()
        config["configurable"]["thread_id"] = str(uuid.uuid4())
        print("Conversation memory cleared. Starting a fresh session.")
        continue
    else:
        response = builder.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        final_message_from_agent = response.get("messages")[-1]
        print("\nFinal Output:\n\n", final_message_from_agent.content)
        
        print("\n--- Current AgentState (for debugging) ---")
        current_state = builder.get_state(config)
        # Filter out large data/message objects for cleaner debugging display
        debug_display_state = {k: v for k, v in current_state.values.items() if k not in ["messages", "data_key", "X_train_key", "X_test_key", "y_train_key", "y_test_key", "pipeline_key"]}
        print(json.dumps(debug_display_state, indent=2, default=str))
        print("-------------------------------------------\n")