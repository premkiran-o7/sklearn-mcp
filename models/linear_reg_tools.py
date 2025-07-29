import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv # Required for Halving search
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from typing import List, Optional, Any, Dict

# Import common tools and the state class from shared_tools
from shared_tools import (
    MLWorkflowState, load_data, preprocess_data, select_imputer,
    select_scaler, evaluate_model, _engineer_features,
    calculate_correlation, drop_highly_correlated, llm,
    FeatureEngineeringPrompt, train_model
)

# --- Specific Tool Functions for Linear Regression ---

def build_linear_regression_pipeline(ml_state: MLWorkflowState,
                                     numeric_imputer_params: Dict[str, Any],
                                     categorical_imputer_params: Dict[str, Any],
                                     numeric_scaler_params: Dict[str, Any],
                                     categorical_encoder_params: Dict[str, Any],
                                     model_params: Dict[str, Any]) -> None:
    """
    Tool to build a scikit-learn preprocessing and Ridge model pipeline.
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
    elif numeric_scaler_params["type"] == "none": # Although not typical for LR, keep consistency with select_scaler
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
            ("num", numeric_pipeline, ml_state.numeric_cols), # CHANGED: Access from ml_state
            ("cat", categorical_pipeline, ml_state.categorical_cols) # CHANGED: Access from ml_state
        ],
        remainder='passthrough'
    )

    # Instantiate Ridge with parameters from model_params
    ridge_model = Ridge(**model_params)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", ridge_model)
    ])
    
    ml_state.pipeline = pipeline # CHANGED: Store the pipeline in the state

    print(f"Linear Regression Pipeline built and stored with key: {ml_state.pipeline_key}")


def tune_and_train_linear_model(ml_state: MLWorkflowState, tuning_method: str) -> None:
    """
    Tunes hyperparameters using the specified method and trains the Ridge model.
    Falls back to GridSearchCV if the chosen method fails.
    Updates the MLWorkflowState directly.
    """
    if ml_state.pipeline is None: raise ValueError("Pipeline not found in state.")
    if ml_state.X_train is None: raise ValueError("X_train not found in state.")
    if ml_state.y_train is None: raise ValueError("y_train not found in state.")

    pipeline = ml_state.pipeline
    X_train = ml_state.X_train
    y_train = ml_state.y_train # For Linear Regression, we often log transform the target

    param_grid = {
        'model__alpha': [0.1, 1.0, 10.0, 100.0],
        'model__fit_intercept': [True, False],
        'model__solver': ['auto', 'svd', 'sag']
    }
    search_cv = None
    method_used = tuning_method
    
    try:
        print(f"\n--- Attempting to tune with {tuning_method} ---")
        if tuning_method == "RandomizedSearchCV":
            search_cv = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=5, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        elif tuning_method == "HalvingGridSearchCV":
            search_cv = HalvingGridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        elif tuning_method == "HalvingRandomSearchCV":
            search_cv = HalvingRandomSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        elif tuning_method == "GridSearchCV":
            search_cv = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        else:
            raise ValueError(f"Unknown tuning_method: {tuning_method}")
            
        # Apply log transform to y_train for Linear Regression fitting
        search_cv.fit(X_train, np.log(y_train + 1e-6)) # Add epsilon to avoid log(0)

    except Exception as e:
        print(f"Warning: {tuning_method} failed with error: {e}.")
        print("--- Falling back to GridSearchCV ---")
        method_used = "GridSearchCV"
        search_cv = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        search_cv.fit(X_train, np.log(y_train + 1e-6)) # Apply log transform for fallback too

    print("Hyperparameter tuning and training complete.")
    print(f"Best parameters found: {search_cv.best_params_}")
    
    ml_state.pipeline = search_cv # Update the pipeline in state with the best estimator
    ml_state.best_hyperparameters = search_cv.best_params_
    ml_state.tuning_method_used = method_used


# --- Orchestrator Tool: run_linear_regression_workflow ---
def run_linear_regression_workflow(
    path: str,
    target_column: str,
    user_query : Optional[str] = None,
    ignore_columns: Optional[List[str]] = None, # Keep this name for agent input
    numeric_imputer_type: str = "simple",
    numeric_imputer_strategy: str = "mean",
    categorical_imputer_type: str = "simple",
    categorical_imputer_strategy: str = "constant",
    numeric_scaler_type: str = "standard",
    categorical_encoder_type: str = "onehot",
    perform_correlation_drop: bool = False,
    tuning_method: str = "GridSearchCV",
    # Linear Regression specific hyperparameters
    lr_alpha: float = 1.0,
    lr_fit_intercept: bool = True,
    lr_solver: str = 'auto'
) -> Dict[str, Any]: # Return a dictionary for the agent's observable output
    """
    Orchestrates an end-to-end Linear Regression workflow with mandatory tuning.
    """
    # Initialize the MLWorkflowState object inside the workflow tool
    # This MLWorkflowState instance will be used to pass data between common functions
    ml_state = MLWorkflowState() # Create a fresh MLWorkflowState for this run

    # Store initial parameters in ml_state for comprehensive summary at the end
    ml_state.workflow_summary.update({
        "path": path, "target_column": target_column, "ignore_columns": ignore_columns,
        "numeric_imputer_type": numeric_imputer_type, "numeric_imputer_strategy": numeric_imputer_strategy,
        "categorical_imputer_type": categorical_imputer_type, "categorical_imputer_strategy": categorical_imputer_strategy,
        "numeric_scaler_type": numeric_scaler_type, "categorical_encoder_type": categorical_encoder_type,
        "perform_correlation_drop": perform_correlation_drop,
        "tuning_method": tuning_method,
        "model_hyperparameters": {
            "alpha": lr_alpha,
            "fit_intercept": lr_fit_intercept,
            "solver": lr_solver
        }
    })
    ml_state.workflow_summary["status"] = "started"

    print(f"Starting ML workflow for {path} with target '{target_column}'...")

    try:
        # Step 1: Load Data
        print("\n--- Step 1: Loading Data ---")
        load_data(ml_state, path=path) # Pass ml_state
        ml_state.workflow_summary["steps_completed"].append("Data Loaded")
        
        # Step 2: Preprocess Data
        print("\n--- Step 2: Preprocessing Data ---")
        preprocess_data(
            ml_state, target_col=target_column, # Pass ml_state
            ignore_cols=ignore_columns # CHANGED: Pass ignore_cols here
        )
        ml_state.workflow_summary["steps_completed"].append("Data Preprocessed")
        ml_state.workflow_summary["numeric_columns"] = ml_state.numeric_cols # Update summary from ml_state
        ml_state.workflow_summary["categorical_columns"] = ml_state.categorical_cols # Update summary from ml_state
        ml_state.workflow_summary["target_column"] = ml_state.target_column # Ensure target_column is also in summary

        print("\n--- Step 2.5: Checking for Feature Engineering Request ---")
        extractor = llm.with_structured_output(FeatureEngineeringPrompt)
        print("User Query:",user_query)
        extraction_prompt_text = (
            "From the following user query, extract the specific instruction for "
            "creating new features. If no such instruction exists, return null.\n\n"
            f"USER QUERY: '{user_query}'"
        )
        extracted_prompt = extractor.invoke(extraction_prompt_text).prompt
        ml_state.feature_engineering_prompt = extracted_prompt # Update ml_state directly
        print("extracted prompt:",extracted_prompt)
        
        if extracted_prompt:
            print(f"Found feature engineering instruction: '{extracted_prompt}'")
            _engineer_features(
                ml_state, user_prompt=extracted_prompt # Pass ml_state
            )
            ml_state.workflow_summary["steps_completed"].append("Feature Engineering")
            ml_state.workflow_summary["newly_added_columns"] = ml_state.newly_added_columns # Update summary from ml_state
        else:
            print("No feature engineering request found in the query.")
            pass

        # Step 3: Handle Highly Correlated Features (if requested)
        if perform_correlation_drop:
            print("\n--- Step 3: Dropping Highly Correlated Features ---")
            drop_highly_correlated(ml_state) # Pass ml_state
            ml_state.workflow_summary["steps_completed"].append("Highly Correlated Features Handled")
            ml_state.workflow_summary["dropped_columns_by_correlation"] = ml_state.dropped_columns_by_correlation
        else:
            print("\n--- Skipping Step 3: Dropping Highly Correlated Features (not requested) ---")

        # Step 4: Select Imputers
        print("\n--- Step 4: Selecting Imputers ---")
        select_imputer(
            ml_state, # Pass ml_state
            numeric_type=numeric_imputer_type, categorical_type=categorical_imputer_type,
            numeric_strategy=numeric_imputer_strategy, categorical_strategy=categorical_imputer_strategy
        )
        ml_state.workflow_summary["steps_completed"].append("Imputers Selected")

        # Step 5: Select Scaler and Encoder
        print("\n--- Step 5: Selecting Scaler and Encoder ---")
        select_scaler(
            ml_state, # Pass ml_state
            numeric_scaler=numeric_scaler_type, categorical_encoder=categorical_encoder_type
        )
        ml_state.workflow_summary["steps_completed"].append("Scaler and Encoder Selected")

        # Step 6: Build Pipeline
        print("\n--- Step 6: Building Pipeline ---")
        build_linear_regression_pipeline( # Call the specific pipeline builder
            ml_state, # Pass ml_state
            numeric_imputer_params={"type": ml_state.numeric_imputer_type, "strategy": ml_state.numeric_imputer_strategy},
            categorical_imputer_params={"type": ml_state.categorical_imputer_type, "strategy": ml_state.categorical_imputer_strategy},
            numeric_scaler_params={"type": ml_state.numeric_scaler_type},
            categorical_encoder_params={"type": ml_state.categorical_encoder_type},
            model_params=ml_state.workflow_summary["model_hyperparameters"] # Get model params from summary
        )
        ml_state.workflow_summary["steps_completed"].append("Pipeline Built")

        # Step 7: Tune and Train Model (Now a mandatory step)
        print("\n--- Step 7: Tuning and Training Model ---")
        tune_and_train_linear_model(ml_state, tuning_method=tuning_method) # Pass ml_state
        ml_state.workflow_summary["steps_completed"].append("Model Tuned and Trained")
        ml_state.workflow_summary["best_hyperparameters"] = ml_state.best_hyperparameters
        ml_state.workflow_summary["tuning_method_used"] = ml_state.tuning_method_used


        # Step 8: Evaluate Model
        print("\n--- Step 8: Evaluating Model ---")
        evaluate_model(ml_state, apply_exp_transform=True) # Pass ml_state, apply inverse transform
        ml_state.workflow_summary["steps_completed"].append("Model Evaluated")
        ml_state.workflow_summary["evaluation_metrics"] = ml_state.evaluation_metrics # Update summary from ml_state
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
        "ml_state": ml_state.model_dump(), # Pass the entire ml_state object as a dictionary for Pydantic to handle
        "workflow_summary": ml_state.workflow_summary # Also return summary for direct access/logging
    }

