# Standard Library Imports
import numpy as np
import pandas as pd
import json
import uuid
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Scikit-learn Imports (minimal here, most are in shared_tools or specific tool files)
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Now in shared_tools
# from sklearn.model_selection import train_test_split # Now in shared_tools

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

# --- Import MLWorkflowState and common LLM instance from shared_tools ---
# shared_tools.py is at the same level as chat.py
from shared_tools import MLWorkflowState, llm, FeatureEngineeringPrompt, print_df_head

# --- Import all specific workflow orchestrators ---
# CORRECTED IMPORT PATHS based on the provided directory structure (all models are directly under 'models/')
from models.linear_reg_tools import run_linear_regression_workflow
from models.decision_tree_reg_tools import run_decision_tree_workflow
from models.random_forest_reg_tools import run_random_forest_workflow
from models.xgboost_reg_tools import run_xgboost_workflow
from models.svr_tools import run_svr_workflow

from models.logistic_regression_tools import run_logistic_regression_workflow
from models.knn_classification_tools import run_knn_workflow
from models.svm_classification_tools import run_svc_workflow
from models.decision_tree_classification_tools import run_decision_tree_classification_workflow
from models.random_forest_classification_tools import run_random_forest_classification_workflow as run_random_forest_classification_workflow # Alias for clarity
from models.naive_bayes_classification_tools import run_naive_bayes_workflow


# --- LangGraph Checkpointer ---
memory = MemorySaver()

# --- AgentState Definition (Unified with MLWorkflowState) ---
class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[BaseMessage] = Field(default_factory=list)

    # --- CHANGED: MLWorkflowState instance directly in AgentState ---
    ml_state: MLWorkflowState = Field(default_factory=MLWorkflowState)

    # --- Fields for Agent's direct understanding/parsing (often mirroring MLWorkflowState for convenience) ---
    algorithm_used: Optional[str] = None # e.g., "Linear Regression", "SVC"
    user_query : Optional[str] = None # The original user query for context/FE parsing
    path: Optional[str] = None # Path to dataset
    target_column: Optional[str] = None # Name of the target column
    ignore_columns: Optional[List[str]] = None # Columns to ignore during preprocessing

    # Preprocessing Parameters (agent can set these, and they'll be passed to workflow tools)
    numeric_imputer_type: Optional[str] = "simple"
    numeric_imputer_strategy: Optional[str] = "mean"
    categorical_imputer_type: Optional[str] = "simple"
    categorical_imputer_strategy: Optional[str] = "constant"
    numeric_scaler_type: Optional[str] = "standard" # Default to standard, let tool adjust for tree models
    categorical_encoder_type: Optional[str] = "onehot"
    perform_correlation_drop: Optional[bool] = False # Only for Linear Regression

    # --- Model-Specific Hyperparameters (agent can set these) ---
    # Linear Regression
    lr_alpha: Optional[float] = 1.0
    lr_fit_intercept: Optional[bool] = True
    lr_solver: Optional[str] = 'auto'
    lr_tuning_method: Optional[str] = "GridSearchCV" # Specific to LR

    # Decision Tree (Regression & Classification)
    dt_criterion: Optional[str] = "squared_error" # For Regressor, "gini" for Classifier
    dt_max_depth: Optional[int] = None
    dt_min_samples_split: Optional[int] = 2
    dt_min_samples_leaf: Optional[int] = 1

    # Random Forest (Regression & Classification)
    rf_n_estimators: Optional[int] = 100
    rf_criterion: Optional[str] = "squared_error" # For Regressor, "gini" for Classifier
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: Optional[int] = 2
    rf_min_samples_leaf: Optional[int] = 1
    rf_max_features: Optional[str] = "sqrt"
    rf_n_jobs: Optional[int] = -1

    # XGBoost (Regression)
    xgb_n_estimators: Optional[int] = 100
    xgb_max_depth: Optional[int] = 6
    xgb_learning_rate: Optional[float] = 0.1
    xgb_subsample: Optional[float] = 0.8
    xgb_colsample_bytree: Optional[float] = 0.8
    xgb_objective: Optional[str] = 'reg:squarederror'
    xgb_n_jobs: Optional[int] = -1

    # SVR (Regression)
    svr_kernel: Optional[str] = 'rbf'
    svr_C: Optional[float] = 1.0
    svr_gamma: Optional[str] = 'scale'
    svr_epsilon: Optional[float] = 0.1

    # SVC (Classification)
    svc_kernel: Optional[str] = 'rbf'
    svc_C: Optional[float] = 1.0
    svc_gamma: Optional[str] = 'scale'
    svc_probability: Optional[bool] = False
    svc_class_weight: Optional[Dict[Any, float]] = None

    # Logistic Regression (Classification)
    logreg_penalty: Optional[str] = 'l2'
    logreg_C: Optional[float] = 1.0
    logreg_solver: Optional[str] = 'lbfgs'
    logreg_max_iter: Optional[int] = 100
    logreg_multi_class: Optional[str] = 'auto'
    logreg_class_weight: Optional[Dict[Any, float]] = None

    # KNN (Classification)
    knn_n_neighbors: Optional[int] = 5
    knn_weights: Optional[str] = 'uniform'
    knn_algorithm: Optional[str] = 'auto'
    knn_p: Optional[int] = 2

    # Naive Bayes (Classification - GaussianNB)
    nb_var_smoothing: Optional[float] = 1e-9

    # --- Computed/Result Fields (mirrored from ml_state.workflow_summary for easy access in chat_model) ---
    best_hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    numeric_cols: Optional[List[str]] = Field(default_factory=list)
    categorical_cols: Optional[List[str]] = Field(default_factory=list)
    newly_added_columns: Optional[List[str]] = Field(default_factory=list)
    evaluation_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict) # Can be float or str (e.g., "NA")
    workflow_summary: Optional[Dict[str, Any]] = Field(default_factory=dict) # Full summary from the last workflow run


# --- Unified list of all workflow tools ---
tools_for_llm = [
    run_linear_regression_workflow,
    run_decision_tree_workflow, # Regression DT
    run_random_forest_workflow, # Regression RF
    run_xgboost_workflow,
    run_svr_workflow,
    run_logistic_regression_workflow, # Classification LR
    run_knn_workflow, # Classification KNN
    run_svc_workflow, # Classification SVC
    run_decision_tree_classification_workflow, # Classification DT
    run_random_forest_classification_workflow, # Classification RF (aliased name)
    run_naive_bayes_workflow, # Classification Naive Bayes
    print_df_head # Shared utility to print dataframe head
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
    
    # Prepare state updates to return
    # Copy relevant fields from AgentState to pass to LLM for context
    # DO NOT include `ml_state` in direct dump to LLM if it contains large DataFrames
    state_for_llm_context = state.model_dump(exclude={"messages", "ml_state"})
    
    message_to_add_to_history = None

    if state.messages and isinstance(state.messages[-1], HumanMessage):
        state_for_llm_context['user_query'] = state.messages[-1].content
        # Update the agent's user_query field for persistence
        state.user_query = state.messages[-1].content

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
            # If the tool returned an MLWorkflowState, update the main AgentState's ml_state
            if "ml_state" in tool_output_dict:
                # IMPORTANT: This copies the *contents* of the returned ml_state (which is a dict from .model_dump())
                # into a new MLWorkflowState instance, ensuring LangGraph's checkpointer tracks changes correctly.
                state.ml_state = MLWorkflowState.model_validate(tool_output_dict["ml_state"])
                
                # Also directly update agent state's summary fields for cleaner access/display in prompt
                # These fields are mirrors of ml_state for convenience in AgentState
                state.workflow_summary = state.ml_state.workflow_summary
                state.evaluation_metrics = state.ml_state.evaluation_metrics
                state.best_hyperparameters = state.ml_state.best_hyperparameters
                state.numeric_cols = state.ml_state.numeric_cols
                state.categorical_cols = state.ml_state.categorical_cols
                state.newly_added_columns = state.ml_state.newly_added_columns
                state.target_column = state.ml_state.target_column
                # Update preprocessing params from ml_state for persistence
                state.numeric_imputer_type = state.ml_state.numeric_imputer_type
                state.numeric_imputer_strategy = state.ml_state.numeric_imputer_strategy
                state.categorical_imputer_type = state.ml_state.categorical_imputer_type
                state.categorical_imputer_strategy = state.ml_state.categorical_imputer_strategy
                state.numeric_scaler_type = state.ml_state.numeric_scaler_type
                state.categorical_encoder_type = state.ml_state.categorical_encoder_type
                state.perform_correlation_drop = state.ml_state.perform_correlation_drop
                state.path = state.ml_state.path # Update path in AgentState too

            # --- Dynamically handle any workflow tool ---
            if last_tool_message.name in workflow_tool_names:
                # Extract algorithm name from the tool name for state update
                algo_name = last_tool_message.name.replace("run_", "").replace("_workflow", "").replace("_", " ").title()
                state.algorithm_used = algo_name # Update algorithm_used in AgentState

                workflow_summary = state.workflow_summary # Get summary directly from AgentState (updated above)
                workflow_status = workflow_summary.get("status")
                metrics = state.evaluation_metrics # Get metrics directly from AgentState

                if workflow_status == "completed successfully":
                    # Build response dynamically
                    response_parts = [
                        "Great news! The ML workflow completed successfully. Here's a summary:\n",
                        f"  - **Algorithm Used**: `{algo_name}`",
                        f"  - **Dataset**: `{state.path or 'N/A'}`", # Use state.path
                        f"  - **Target Column**: `{state.target_column or 'N/A'}`" # Use state.target_column
                    ]
                    
                    best_params = state.best_hyperparameters
                    if best_params:
                        # This part cleans up names like 'model__alpha' to just 'alpha' for display
                        params_str = ", ".join([f"'{k.split('__')[-1]}': {v}" for k, v in best_params.items()])
                        response_parts.append(f"  - **Best Hyperparameters**: {{{params_str}}}")

                    if state.ml_state.feature_engineering_prompt: # Check from ml_state
                        response_parts.append(f"  - **Feature Engineering**: Performed with prompt: \"_{state.ml_state.feature_engineering_prompt}_\"")
                        new_cols = state.ml_state.newly_added_columns # Check from ml_state
                        if new_cols:
                            response_parts.append(f"    - New columns created: `{', '.join(new_cols)}`")

                    # Use values directly from AgentState for response
                    response_parts.extend([
                        f"  - **Numeric Imputer**: `{state.numeric_imputer_type}` (`{state.numeric_imputer_strategy}`)",
                        f"  - **Categorical Imputer**: `{state.categorical_imputer_type}` (`{state.categorical_imputer_strategy}`)",
                        f"  - **Numeric Scaler**: `{state.numeric_scaler_type}`",
                        f"  - **Categorical Encoder**: `{state.categorical_encoder_type}`",
                        "\n**Evaluation Metrics:**"
                    ])

                    # Dynamically add metrics based on whether it's regression or classification
                    if "r2" in metrics: # Likely regression metrics
                        response_parts.extend([
                            f"  - **R-squared**: `{metrics.get('r2', 0):.3f}`",
                            f"  - **MAE**: `{metrics.get('mae', 0):.3f}`",
                            f"  - **RMSE**: `{metrics.get('rmse', 0):.3f}`"
                        ])
                    elif "accuracy" in metrics: # Likely classification metrics
                        response_parts.extend([
                            f"  - **Accuracy**: `{metrics.get('accuracy', 0):.3f}`",
                            f"  - **Precision**: `{metrics.get('precision', 0):.3f}`",
                            f"  - **Recall**: `{metrics.get('recall', 0):.3f}`",
                            f"  - **F1-Score**: `{metrics.get('f1_score', 0):.3f}`"
                        ])
                        if metrics.get('roc_auc') not in ["N/A", "Error"]:
                            response_parts.append(f"  - **ROC AUC**: `{metrics.get('roc_auc', 0):.3f}`")
                        elif metrics.get('roc_auc') == "Error":
                             response_parts.append(f"  - **ROC AUC**: Error during calculation")
                        else:
                             response_parts.append(f"  - **ROC AUC**: N/A")

                    response_parts.append("\nHow else can I assist you? You can ask to change a parameter or try a different algorithm.")
                    response_text = "\n".join(response_parts)

                elif workflow_status == "failed":
                    error_message = workflow_summary.get("error", "An unknown error occurred.")
                    response_text = f"Apologies, the ML workflow failed. Error: {error_message}\nPlease check your input or try again."
                
                message_to_add_to_history = AIMessage(content=response_text)

            elif last_tool_message.name == "print_df_head": # Corrected tool name
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
        
        # --- CHANGED: Updated system prompt for multiple models and state access ---
        dynamic_system_message_content = (
            "You are an AI assistant specialized in machine learning workflows. "
            "Your goal is to help users run regression and classification models. "
            "You have access to several workflow tools, each corresponding to a specific ML algorithm. "
            "The tools are named `run_ALGORITHM_workflow` (e.g., `run_linear_regression_workflow`).\n\n"
            "**Available Workflow Tools & Their Key Parameters:**\n"
            "- `run_linear_regression_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, perform_correlation_drop, tuning_method, lr_alpha, lr_fit_intercept, lr_solver)`\n"
            "- `run_decision_tree_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, dt_criterion, dt_max_depth, dt_min_samples_split, dt_min_samples_leaf)`\n"
            "- `run_random_forest_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, rf_n_estimators, rf_criterion, rf_max_depth, rf_min_samples_split, rf_min_samples_leaf, rf_max_features, rf_n_jobs)`\n"
            "- `run_xgboost_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, xgb_n_estimators, xgb_max_depth, xgb_learning_rate, xgb_subsample, xgb_colsample_bytree, xgb_objective, xgb_n_jobs)`\n"
            "- `run_svr_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, svr_kernel, svr_C, svr_gamma, svr_epsilon)`\n"
            "- `run_logistic_regression_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, logreg_penalty, logreg_C, logreg_solver, logreg_max_iter, logreg_multi_class, logreg_class_weight)`\n"
            "- `run_knn_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, knn_n_neighbors, knn_weights, knn_algorithm, knn_p)`\n"
            "- `run_svc_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, svc_kernel, svc_C, svc_gamma, svc_probability, svc_class_weight)`\n"
            "- `run_decision_tree_classification_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, dt_criterion, dt_max_depth, dt_min_samples_split, dt_min_samples_leaf)`\n"
            "- `run_random_forest_classification_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, rf_n_estimators, rf_criterion, rf_max_depth, rf_min_samples_split, rf_min_samples_leaf, rf_max_features, rf_n_jobs)`\n"
            "- `run_naive_bayes_workflow(path, target_column, user_query, ignore_columns, numeric_imputer_type, numeric_imputer_strategy, categorical_imputer_type, categorical_imputer_strategy, numeric_scaler_type, categorical_encoder_type, nb_var_smoothing)`\n"
            "- `print_df_head(state, df_type)`: Use this to inspect dataframes (df_type can be 'data', 'X_train', 'X_test', 'y_train', 'y_test').\n\n"
            "**Instructions:**\n"
            "1. **Select the Right Tool:** Based on the user's prompt (e.g., 'use random forest for classification', 'try svr for regression'), select the corresponding workflow tool. If no model is specified, default to `run_linear_regression_workflow`.\n"
            "2. **Reuse Current State:** For any parameter not explicitly mentioned or overridden in the user's current prompt, you MUST reuse the value from the `Current Agent State` provided below. This includes all common preprocessing parameters and model-specific hyperparameters.\n"
            "3. **Feature Engineering:** If the user asks to create new features, use the `user_query` argument in your chosen workflow tool, which will be parsed for a `feature_engineering_prompt` internally.\n"
            "4. **Handle Errors:** If a tool fails, inform the user clearly and ask for corrected information.\n"
            "5. **Summarize Results:** After a workflow tool completes successfully, your next response MUST be a clear summary. DO NOT call another tool in that turn.\n"
            "6. **Parameter Prioritization:** User-provided parameters in the current turn take precedence over current state values. Current state values take precedence over tool default values.\n"
            f"\nCurrent date and time: {current_time} in {current_location}.\n\n"
            "**Current Agent State (For your use, do not show to user unless asked):**\n"
            f"Algorithm Used: {state.algorithm_used}\n"
            f"Path: {state.path}\n"
            f"Target Column: {state.target_column}\n"
            f"Numeric Imputer Type: {state.numeric_imputer_type}, Strategy: {state.numeric_imputer_strategy}\n"
            f"Categorical Imputer Type: {state.categorical_imputer_type}, Strategy: {state.categorical_imputer_strategy}\n"
            f"Numeric Scaler Type: {state.numeric_scaler_type}\n"
            f"Categorical Encoder Type: {state.categorical_encoder_type}\n"
            f"Perform Correlation Drop (LR only): {state.perform_correlation_drop}\n"
            f"LR Tuning Method: {state.lr_tuning_method}\n"
            f"LR Alpha: {state.lr_alpha}, Fit Intercept: {state.lr_fit_intercept}, Solver: {state.lr_solver}\n"
            f"DT Criterion: {state.dt_criterion}, Max Depth: {state.dt_max_depth}, Min Samples Split: {state.dt_min_samples_split}, Min Samples Leaf: {state.dt_min_samples_leaf}\n"
            f"RF Estimators: {state.rf_n_estimators}, RF Criterion: {state.rf_criterion}, RF Max Depth: {state.rf_max_depth}, RF Min Samples Split: {state.rf_min_samples_split}, RF Min Samples Leaf: {state.rf_min_samples_leaf}, RF Max Features: {state.rf_max_features}, RF N Jobs: {state.rf_n_jobs}\n"
            f"XGB Estimators: {state.xgb_n_estimators}, Max Depth: {state.xgb_max_depth}, Learning Rate: {state.xgb_learning_rate}, Subsample: {state.xgb_subsample}, Colsample By Tree: {state.xgb_colsample_bytree}, Objective: {state.xgb_objective}, N Jobs: {state.xgb_n_jobs}\n"
            f"SVR Kernel: {state.svr_kernel}, C: {state.svr_C}, Gamma: {state.svr_gamma}, Epsilon: {state.svr_epsilon}\n"
            f"SVC Kernel: {state.svc_kernel}, C: {state.svc_C}, Gamma: {state.svc_gamma}, Probability: {state.svc_probability}, Class Weight: {state.svc_class_weight}\n"
            f"Logistic Reg Penalty: {state.logreg_penalty}, C: {state.logreg_C}, Solver: {state.logreg_solver}, Max Iter: {state.logreg_max_iter}, Multi Class: {state.logreg_multi_class}, Class Weight: {state.logreg_class_weight}\n"
            f"KNN N Neighbors: {state.knn_n_neighbors}, Weights: {state.knn_weights}, Algorithm: {state.knn_algorithm}, P: {state.knn_p}\n"
            f"Naive Bayes Var Smoothing: {state.nb_var_smoothing}\n"
            f"Evaluation Metrics (Last Run): {state.evaluation_metrics}\n"
            f"Last Workflow Status: {state.workflow_summary.get('status', 'N/A')}\n"
            f"User Query (this turn, for FE parsing): {state.messages[-1].content if isinstance(state.messages[-1], HumanMessage) else 'N/A'}\n"
            "Available DataFrames for `print_df_head`: 'data', 'X_train', 'X_test', 'y_train', 'y_test'\n"
        )
        dynamic_system_message = SystemMessage(content=dynamic_system_message_content)
        llm_input_messages = [dynamic_system_message] + state.messages
        message_to_add_to_history = agent_executor.invoke(llm_input_messages)
        
    print(f"Message to append to history: {message_to_add_to_history}")
    return {"messages": add_messages(state.messages, message_to_add_to_history)}

# --- LangGraph Graph Definition ---
graph = StateGraph(AgentState)
graph.add_node("chat_model", chat_model)
# The ToolNode will automatically inspect the arguments of the tools_for_llm and pass the AgentState's
# attributes that match the tool's arguments.
graph.add_node("tools", ToolNode(tools_for_llm))
graph.add_edge(START, "chat_model")
graph.add_conditional_edges("chat_model", tools_condition)
graph.add_edge("tools", "chat_model")
graph.add_edge("chat_model", END) # Allows graph to finish if no tool is called
builder = graph.compile(checkpointer=memory)

# --- Main Execution Loop ---
# Initializing config with a unique thread_id for each session
config = {"configurable": {"thread_id": "1"}} # Start with a default, can be reset

print("ML Workflow Agent: Type 'exit' to quit. Use 'reset' to clear conversation memory and start fresh.")
while True:
    prompt = input("Enter your message: ")

    if prompt.lower() == "exit":
        break
    elif prompt.lower() == "reset":
        # Simply change the thread_id to reset the conversation state
        config["configurable"]["thread_id"] = str(uuid.uuid4())
        print("Conversation memory cleared. Starting a fresh session.")
        continue
    else:
        # Pass the full AgentState to the builder.invoke
        # LangGraph handles merging the output of `chat_model` and `tools` back into the state
        response = builder.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        
        # Access messages directly from the response
        final_message_from_agent = response.get("messages")[-1]
        print("\nFinal Output:\n\n", final_message_from_agent.content)
        
        print("\n--- Current AgentState (for debugging) ---")
        current_state = builder.get_state(config)
        # Filter out messages and the ml_state's large internal stores for cleaner debugging display
        debug_display_state = {k: v for k, v in current_state.values.items() if k not in ["messages"]}
        if 'ml_state' in debug_display_state:
            # Further filter internal stores of ml_state for clean display
            debug_display_state['ml_state'] = {
                k_ml: v_ml for k_ml, v_ml in debug_display_state['ml_state'].__dict__.items()
                if not k_ml.startswith('_') # Exclude _data_store, etc.
            }
        print(json.dumps(debug_display_state, indent=2, default=str)) # Use default=str for any non-serializable types
        print("-------------------------------------------\n")
