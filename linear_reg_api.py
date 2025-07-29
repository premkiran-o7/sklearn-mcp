from fastapi import FastAPI
from regression_models.linear_reg_tools import run_linear_regression_workflow
from typing import Optional
from typing import List # Import List

server = FastAPI()

@server.get("/") # Added a path for the GET endpoint
def hello():
    return {"Greeting": "Hello"}

@server.post("/ml_method") # Added a path for the POST endpoint
def ml_method(
    path: str,
    target_column: str,
    user_query: Optional[str] = None,
    feature_engineering_prompt: Optional[str] = None,
    ignore_columns: Optional[List[str]] = None,
    numeric_imputer_type: str = "simple",
    numeric_imputer_strategy: str = "mean",
    categorical_imputer_type: str = "simple",
    categorical_imputer_strategy: str = "constant",
    numeric_scaler_type: str = "standard",
    categorical_encoder_type: str = "onehot",
    perform_correlation_drop: bool = False,
    tuning_method: str = "GridSearchCV",
):
    # Pass the arguments by their names, not with type declarations
    return run_linear_regression_workflow(
        path=path,
        target_column=target_column,
        user_query=user_query,
        feature_engineering_prompt=feature_engineering_prompt,
        ignore_columns=ignore_columns,
        numeric_imputer_type=numeric_imputer_type,
        numeric_imputer_strategy=numeric_imputer_strategy,
        categorical_imputer_type=categorical_imputer_type,
        categorical_imputer_strategy=categorical_imputer_strategy,
        numeric_scaler_type=numeric_scaler_type,
        categorical_encoder_type=categorical_encoder_type,
        perform_correlation_drop=perform_correlation_drop,
        tuning_method=tuning_method,
    )