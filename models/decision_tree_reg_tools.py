import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor # Specific model for this file
from typing import List, Optional, Any, Dict

# Import common tools and the state class from shared_tools
from shared_tools import (
    MLWorkflowState, load_data, preprocess_data, select_imputer,
    select_scaler, evaluate_model, _engineer_features,
    llm, FeatureEngineeringPrompt, train_model # No correlation drop for tree models
)

# --- Specific Tool Functions for Decision Tree Regression ---

def build_decision_tree_pipeline(ml_state: MLWorkflowState,
                                 numeric_imputer_params: Dict[str, Any],
                                 categorical_imputer_params: Dict[str, Any],
                                 numeric_scaler_params: Dict[str, Any],
                                 categorical_encoder_params: Dict[str, Any],
                                 model_params: Dict[str, Any]) -> None:
    """
    Tool to build a scikit-learn preprocessing and Decision Tree Regressor model pipeline.
    Constructs objects from parameters and stores the pipeline in MLWorkflowState.
    """
    # Instantiate imputers
    if numeric_imputer_params["type"] == "simple":
        numeric_imputer = SimpleImputer(strategy=numeric_imputer_params["strategy"])
    elif numeric_imputer_params["type"] == "knn":
        numeric_imputer = KNNImputer()
    else:
        raise ValueError(f"Unknown numeric imputer type: {numeric_imputer_params['type']}")

    if categorical_imputer_params["type"] == "simple":
        if categorical_imputer_params["strategy"] == "constant":
            categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        else: # 'most_frequent'
            categorical_imputer = SimpleImputer(strategy=categorical_imputer_params["strategy"])
    elif categorical_imputer_params["type"] == "constant":
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        raise ValueError(f"Unknown categorical imputer type: {categorical_imputer_params['type']}")

    # Instantiate scalers/encoders
    if numeric_scaler_params["type"] == "standard":
        numeric_scaler_obj = StandardScaler()
    elif numeric_scaler_params["type"] == "minmax":
        numeric_scaler_obj = MinMaxScaler()
    elif numeric_scaler_params["type"] == "none": # Default for tree-based models
        numeric_scaler_obj = 'passthrough'
    else:
        raise ValueError(f"Unknown numeric scaler type: {numeric_scaler_params['type']}")

    if categorical_encoder_params["type"] == "onehot":
        categorical_encoder_obj = OneHotEncoder(handle_unknown="ignore")
    elif categorical_encoder_params["type"] == "ordinal":
        categorical_encoder_obj = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        raise ValueError(f"Unknown categorical encoder type: {categorical_encoder_params['type']}")

    # Create numeric pipeline steps, handling the 'none' scaler case
    numeric_pipeline_steps = [("imputer", numeric_imputer)]
    if numeric_scaler_obj != 'passthrough':
        numeric_pipeline_steps.append(("scaler", numeric_scaler_obj))
    numeric_pipeline = Pipeline(numeric_pipeline_steps)

    categorical_pipeline = Pipeline([
        ("imputer", categorical_imputer),
        ("encoder", categorical_encoder_obj)
    ])

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, ml_state.numeric_cols),
            ("cat", categorical_pipeline, ml_state.categorical_cols)
        ],
        remainder='passthrough'
    )

    # Instantiate the Decision Tree Regressor with specified parameters
    decision_tree_model = DecisionTreeRegressor(**model_params)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", decision_tree_model)
    ])
    
    ml_state.pipeline = pipeline # Store the pipeline in the state

    print(f"Pipeline with DecisionTreeRegressor built and stored with key: {ml_state.pipeline_key}")
    print(f"Model Parameters: {model_params}")


# --- Orchestrator Tool: run_decision_tree_workflow ---
def run_decision_tree_workflow(
    path: str,
    target_column: str,
    user_query : Optional[str] = None,
    ignore_columns: Optional[List[str]] = None, # Keep this name for agent input
    numeric_imputer_type: str = "simple",
    numeric_imputer_strategy: str = "mean",
    categorical_imputer_type: str = "simple",
    categorical_imputer_strategy: str = "constant",
    numeric_scaler_type: str = "none", # Default to 'none' for tree-based models
    categorical_encoder_type: str = "onehot",
    # --- Decision Tree Regressor Hyperparameters ---
    dt_criterion: str = "squared_error",
    dt_max_depth: Optional[int] = None,
    dt_min_samples_split: int = 2,
    dt_min_samples_leaf: int = 1,
) -> Dict[str, Any]: # Return a dictionary for the agent's observable output
    """
    A comprehensive tool to orchestrate an end-to-end Decision Tree Regressor workflow.
    """
    # Initialize the MLWorkflowState object inside the workflow tool
    ml_state = MLWorkflowState() # Create a fresh MLWorkflowState for this run

    # Store initial parameters in ml_state for comprehensive summary at the end
    ml_state.workflow_summary.update({
        "path": path, "target_column": target_column, "ignore_columns": ignore_columns,
        "numeric_imputer_type": numeric_imputer_type, "numeric_imputer_strategy": numeric_imputer_strategy,
        "categorical_imputer_type": categorical_imputer_type, "categorical_imputer_strategy": categorical_imputer_strategy,
        "numeric_scaler_type": numeric_scaler_type, "categorical_encoder_type": categorical_encoder_type,
        "model_hyperparameters": {
            "criterion": dt_criterion,
            "max_depth": dt_max_depth,
            "min_samples_split": dt_min_samples_split,
            "min_samples_leaf": dt_min_samples_leaf,
            "random_state": 42 # for reproducibility
        }
    })
    ml_state.workflow_summary["status"] = "started"

    print(f"Starting Decision Tree workflow for {path} with target '{target_column}'...")

    try:
        # Step 1: Load Data
        print("\n--- Step 1: Loading Data ---")
        load_data(ml_state, path=path)
        ml_state.workflow_summary["steps_completed"].append("Data Loaded")
        
        # Step 2: Preprocess Data
        print("\n--- Step 2: Preprocessing Data ---")
        preprocess_data(
            ml_state, target_column,
            ignore_cols=ignore_columns # CHANGED: Pass ignore_cols here
        )
        ml_state.workflow_summary["steps_completed"].append("Data Preprocessed")
        ml_state.workflow_summary["numeric_columns"] = ml_state.numeric_cols
        ml_state.workflow_summary["categorical_columns"] = ml_state.categorical_cols
        ml_state.workflow_summary["target_column"] = ml_state.target_column

        print("\n--- Step 2.5: Checking for Feature Engineering Request ---")
        extractor = llm.with_structured_output(FeatureEngineeringPrompt)
        print("User Query:",user_query)
        extraction_prompt_text = (
            "From the following user query, extract the specific instruction for "
            "creating new features. If no such instruction exists, return null.\n\n"
            f"USER QUERY: '{user_query}'"
        )
        extracted_prompt = extractor.invoke(extraction_prompt_text).prompt
        ml_state.feature_engineering_prompt = extracted_prompt
        print("extracted prompt:",extracted_prompt)
        
        if extracted_prompt:
            print(f"Found feature engineering instruction: '{extracted_prompt}'")
            _engineer_features(
                ml_state, user_prompt=extracted_prompt
            )
            ml_state.workflow_summary["steps_completed"].append("Feature Engineering")
            ml_state.workflow_summary["newly_added_columns"] = ml_state.newly_added_columns
        else:
            print("No feature engineering request found in the query.")
            pass

        # NOTE: Correlation drop step is typically skipped for tree-based models.
        print("\n--- Skipping Correlation Drop (not typically necessary for Decision Trees) ---")

        # Step 4: Select Imputers
        print("\n--- Step 4: Selecting Imputers ---")
        select_imputer(
            ml_state,
            numeric_type=numeric_imputer_type, categorical_type=categorical_imputer_type,
            numeric_strategy=numeric_imputer_strategy, categorical_strategy=categorical_imputer_strategy
        )
        ml_state.workflow_summary["steps_completed"].append("Imputers Selected")

        # Step 5: Select Scaler and Encoder
        print("\n--- Step 5: Selecting Scaler and Encoder ---")
        select_scaler(
            ml_state,
            numeric_scaler=numeric_scaler_type, categorical_encoder=categorical_encoder_type
        )
        ml_state.workflow_summary["steps_completed"].append("Scaler and Encoder Selected")

        # Step 6: Build Pipeline
        print("\n--- Step 6: Building Pipeline ---")
        build_decision_tree_pipeline( # Call the specific pipeline builder
            ml_state,
            numeric_imputer_params={"type": ml_state.numeric_imputer_type, "strategy": ml_state.numeric_imputer_strategy},
            categorical_imputer_params={"type": ml_state.categorical_imputer_type, "strategy": ml_state.categorical_imputer_strategy},
            numeric_scaler_params={"type": ml_state.numeric_scaler_type},
            categorical_encoder_params={"type": ml_state.categorical_encoder_type},
            model_params=ml_state.workflow_summary["model_hyperparameters"]
        )
        ml_state.workflow_summary["steps_completed"].append("Pipeline Built")

        # Step 7: Train Model
        print("\n--- Step 7: Training Model ---")
        train_model(ml_state, apply_log_transform=False) # No log transform for DT by default
        ml_state.workflow_summary["steps_completed"].append("Model Trained")

        # Step 8: Evaluate Model
        print("\n--- Step 8: Evaluating Model ---")
        evaluate_model(ml_state, apply_exp_transform=False) # No exp transform for DT by default
        ml_state.workflow_summary["steps_completed"].append("Model Evaluated")
        ml_state.workflow_summary["evaluation_metrics"] = ml_state.evaluation_metrics
        if ml_state.workflow_summary["evaluation_status"] == "failed":
            raise Exception(f"Model evaluation failed: {ml_state.workflow_summary.get('error', 'Unknown error')}")

        ml_state.workflow_summary["status"] = "completed successfully"
        print(f"\nML Workflow {ml_state.workflow_summary['status']}!")

    except Exception as e:
        ml_state.workflow_summary["status"] = "failed"
        ml_state.workflow_summary["error"] = str(e)
        print(f"\nML Workflow {ml_state.workflow_summary['status']} with error: {e}")
            
    # Return the relevant parts of ml_state.workflow_summary for LangGraph to merge into AgentState
    return {
        "ml_state": ml_state.model_dump(),
        "workflow_summary": ml_state.workflow_summary
    }
