# Standard Library Imports
import numpy as np
import pandas as pd
import uuid
import os
from typing import List, Optional, Any, Dict

# Scikit-learn Imports
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report # Added classification metrics
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline # Needed for building pipelines in shared functions if generic steps are included

# Pydantic Imports for LLM structured output and state management
from pydantic import BaseModel, Field, ConfigDict # No PrivateAttr needed in MLWorkflowState itself
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the LLM instance (common across all tools)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- Global In-Memory Stores (Reintroduced for actual DataFrame/Pipeline objects) ---
# These global dictionaries will hold the actual dataframes and pipeline objects.
# MLWorkflowState will only store keys referencing these.
_data_store: Dict[str, pd.DataFrame] = {}
_pipeline_store: Dict[str, Any] = {}
_X_train_store: Dict[str, pd.DataFrame] = {}
_X_test_store: Dict[str, pd.DataFrame] = {}
_y_train_store: Dict[str, pd.Series] = {}
_y_test_store: Dict[str, pd.Series] = {}

# --- MLWorkflowState Class ---
class MLWorkflowState(BaseModel):
    """
    Manages the comprehensive state of the ML workflow for a single run.
    This class now primarily stores keys and metadata, with actual
    DataFrame/Pipeline objects stored in global dictionaries.
    """
    # Allow arbitrary types for properties that return complex objects, though the fields themselves are simple.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Keys to track the currently active data/pipeline in the global stores
    data_key: Optional[str] = None
    X_train_key: Optional[str] = None
    X_test_key: Optional[str] = None
    y_train_key: Optional[str] = None
    y_test_key: Optional[str] = None
    pipeline_key: Optional[str] = None

    # Core Workflow Parameters (can be set by user or derived)
    path: Optional[str] = None
    target_column: Optional[str] = None
    ignore_columns: Optional[List[str]] = None
    feature_engineering_prompt: Optional[str] = None # Prompt used for FE
    
    # Preprocessing Parameters (selected by agent or default)
    numeric_imputer_type: Optional[str] = "simple"
    numeric_imputer_strategy: Optional[str] = "mean"
    categorical_imputer_type: Optional[str] = "simple"
    categorical_imputer_strategy: Optional[str] = "constant"
    numeric_scaler_type: Optional[str] = "standard"
    categorical_encoder_type: Optional[str] = "onehot"
    perform_correlation_drop: Optional[bool] = False # Specific to Linear Regression primarily

    # Results and Metadata
    numeric_cols: List[str] = Field(default_factory=list) # Identified numeric columns
    categorical_cols: List[str] = Field(default_factory=list) # Identified categorical columns
    newly_added_columns: List[str] = Field(default_factory=list) # Columns added by FE
    dropped_columns_by_correlation: List[str] = Field(default_factory=list) # Columns dropped due to high correlation
    
    evaluation_metrics: Dict[str, Any] = Field(default_factory=dict) # Final evaluation metrics
    best_hyperparameters: Optional[Dict[str, Any]] = None # Best params from tuning
    tuning_method_used: Optional[str] = None # Method used for tuning (e.g., GridSearchCV)

    # General workflow summary for agent response
    workflow_summary: Dict[str, Any] = Field(default_factory=lambda: {"status": "initialized", "steps_completed": [], "error": None})

    # Properties to access the actual DataFrames/Pipeline objects from global stores
    @property
    def data(self) -> Optional[pd.DataFrame]:
        return _data_store.get(self.data_key)

    @data.setter
    def data(self, df: pd.DataFrame):
        new_key = f"df_{uuid.uuid4()}"
        _data_store[new_key] = df
        # Clean up old key if it exists and is different
        if self.data_key and self.data_key != new_key:
            del _data_store[self.data_key]
        self.data_key = new_key

    @property
    def X_train(self) -> Optional[pd.DataFrame]:
        return _X_train_store.get(self.X_train_key)

    @X_train.setter
    def X_train(self, df: pd.DataFrame):
        new_key = f"X_train_{uuid.uuid4()}"
        _X_train_store[new_key] = df
        # Clean up old key if it exists and is different
        if self.X_train_key and self.X_train_key != new_key:
            del _X_train_store[self.X_train_key]
        self.X_train_key = new_key

    @property
    def X_test(self) -> Optional[pd.DataFrame]:
        return _X_test_store.get(self.X_test_key)

    @X_test.setter
    def X_test(self, df: pd.DataFrame):
        new_key = f"X_test_{uuid.uuid4()}"
        _X_test_store[new_key] = df
        # Clean up old key if it exists and is different
        if self.X_test_key and self.X_test_key != new_key:
            del _X_test_store[self.X_test_key]
        self.X_test_key = new_key

    @property
    def y_train(self) -> Optional[pd.Series]:
        return _y_train_store.get(self.y_train_key)

    @y_train.setter
    def y_train(self, s: pd.Series):
        new_key = f"y_train_{uuid.uuid4()}"
        _y_train_store[new_key] = s
        # Clean up old key if it exists and is different
        if self.y_train_key and self.y_train_key != new_key:
            del _y_train_store[self.y_train_key]
        self.y_train_key = new_key

    @property
    def y_test(self) -> Optional[pd.Series]:
        return _y_test_store.get(self.y_test_key)

    @y_test.setter
    def y_test(self, s: pd.Series):
        new_key = f"y_test_{uuid.uuid4()}"
        _y_test_store[new_key] = s
        # Clean up old key if it exists and is different
        if self.y_test_key and self.y_test_key != new_key:
            del _y_test_store[self.y_test_key]
        self.y_test_key = new_key

    @property
    def pipeline(self) -> Any:
        return _pipeline_store.get(self.pipeline_key)

    @pipeline.setter
    def pipeline(self, p: Any):
        new_key = f"pipeline_{uuid.uuid4()}"
        _pipeline_store[new_key] = p
        # Clean up old key if it exists and is different
        if self.pipeline_key and self.pipeline_key != new_key:
            del _pipeline_store[self.pipeline_key]
        self.pipeline_key = new_key

# --- Helper Pydantic Models for LLM structured output (used by preprocess_data and _engineer_features) ---
class TargetColumnInput(BaseModel):
    target_column: str = Field(
        ...,
        description="Name of the target column to predict. Provide only the column name from the given list of columns.",
    )

class FeatureEngineeringCode(BaseModel):
    """Pydantic model for structuring the Python code generated by the LLM."""
    code: str = Field(
        ...,
        description=(
            "A string containing a single Python function named 'create_features'. "
            "This function must accept a pandas DataFrame as its only argument "
            "and return the modified DataFrame."
        )
    )

class FeatureEngineeringPrompt(BaseModel):
    """Pydantic model for structuring the extracted feature engineering prompt."""
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "The specific, concise instruction for creating features. "
            "If no feature engineering is requested, this should be null or an empty string."
        )
    )

# --- Common Helper Functions (operate on MLWorkflowState) ---

def _engineer_features(state: MLWorkflowState, user_prompt: str) -> None:
    """
    Internal helper to generate and apply feature engineering code.
    Updates the MLWorkflowState directly.
    """
    print("\n--- Step 2.5: Performing Feature Engineering ---")

    X_train_current = state.X_train
    X_test_current = state.X_test

    if X_train_current is None or X_test_current is None:
        raise ValueError("X_train and X_test must be set in state before feature engineering.")

    code_generation_prompt = f"""You are an expert Python data scientist. Your task is to write a single, self-contained Python function named `create_features`.

    **Function Definition:**
    - The function must be named `create_features`.
    - It must accept a pandas DataFrame `df` as its only argument.
    - It must return the modified pandas DataFrame.

    **User's Request for New Features:**
    "{user_prompt}"

    **Available DataFrame Columns:**
    {X_train_current.columns.tolist()}

    **First 5 Rows of Data (for context):**
    {X_train_current.head().to_string()}

    ---
    ### **CRITICAL INSTRUCTIONS for Robust Code Generation:**
    1.  **Avoid Side Effects**: The first line of your function **MUST** be `df = df.copy()` to ensure the original DataFrame is not modified.
    2.  **Self-Contained Function**: Import any necessary libraries (like `numpy` or `pandas`) **inside** the function to make it portable.
    3.  **Handle Missing Columns**: Before using a column, check if it exists in `df.columns`. If a required column is missing, print a warning and gracefully skip the creation of that specific feature.
    4.  **Prevent Division by Zero**: When creating ratio features (e.g., `a / b`), you **MUST** add a small epsilon (e.g., `1e-6`) to the denominator to avoid generating infinite values.
    5.  **Safe Log Transforms**: When using `np.log()`, add 1 to the column first (`np.log(df['column'] + 1)`) to handle zeros correctly.
    6.  **Correct Data Types**: Explicitly convert columns to the correct type (e.g., using `pd.to_numeric` with `errors='coerce'`) before performing operations.
    7.  **Code Clarity**: Add a brief, one-line comment explaining the purpose of each new feature you create.
    8.  **Return Value**: The function must end by returning the modified DataFrame.
    """
    code_generator = llm.with_structured_output(FeatureEngineeringCode)
    generated_code = code_generator.invoke(code_generation_prompt).code
    print(f"Generated code:\n{generated_code}")

    local_namespace = {}
    exec(generated_code, globals(), local_namespace)
    feature_creation_func = local_namespace['create_features']

    X_train_new = feature_creation_func(X_train_current.copy())
    X_test_new = feature_creation_func(X_test_current.copy())

    newly_added_columns = list(set(X_train_new.columns) - set(X_train_current.columns))
    print(f"New columns created: {newly_added_columns}")

    # Store new data using setter properties, which also handle old key cleanup
    state.X_train = X_train_new
    state.X_test = X_test_new

    state.numeric_cols = X_train_new.select_dtypes(include=np.number).columns.tolist()
    state.categorical_cols = X_train_new.select_dtypes(include=['object', 'category']).columns.tolist()
    state.newly_added_columns = newly_added_columns


def drop_missing_rows(df: pd.DataFrame, pct: float = 0.3) -> pd.DataFrame:
    """Helper function to drop rows with more than 'pct' missing values."""
    max_missing_cols = int(df.shape[1] * pct)
    missing_per_row = df.isnull().sum(axis=1)
    df_cleaned = df[missing_per_row <= max_missing_cols].copy()
    return df_cleaned

# --- Core Tool Functions (operate on MLWorkflowState) ---

def load_data(state: MLWorkflowState, path: str) -> None:
    """
    Tool to load a dataframe from a specified CSV file path.
    Stores the loaded DataFrame in MLWorkflowState.
    """
    if not path:
        raise ValueError("No path provided for dataset.")
    try:
        df = pd.read_csv(path)
        state.data = df # Use the setter property to store the DataFrame
        state.path = path # Store the path in the state for summary
        print(f"Successfully loaded DataFrame from {path} with shape: {df.shape}. Stored with key: {state.data_key}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except Exception as e:
        raise ValueError(f"Error loading data from {path}: {e}")

def preprocess_data(state: MLWorkflowState, target_col: str, ignore_cols: Optional[List[str]] = None) -> None:
    """
    Tool to preprocess a dataframe: splits into train/test, identifies numeric/categorical columns.
    Updates the MLWorkflowState directly.
    """
    if state.data is None:
        raise ValueError("No DataFrame found in state for preprocessing.")
    
    df = state.data.copy()

    if ignore_cols is None:
        ignore_cols = []

    train_df_copy = df.drop(columns=ignore_cols, errors='ignore').copy()

    # Drop rows with excessive missing values
    initial_rows = train_df_copy.shape[0]
    train_df_copy = drop_missing_rows(train_df_copy)
    rows_dropped = initial_rows - train_df_copy.shape[0]
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to excessive missing values during preprocessing.")

    # Use LLM for robust target column extraction
    target_extractor = llm.with_structured_output(TargetColumnInput)
    extracted_target = target_extractor.invoke(
                f"Extract the exact target column name similar to '{target_col}', if no column is provided, pick the target column as you deem fit"
                f"from the available columns: {list(train_df_copy.columns)}. Provide only the column name."
            ).target_column

    if extracted_target not in train_df_copy.columns:
        raise ValueError(f"Target column '{extracted_target}' not found in the DataFrame after dropping ignored columns/missing rows. Available columns: {list(train_df_copy.columns)}")
    else:
        state.target_column = extracted_target # Store the extracted target_col in state

    # Separate features (X) and target (y)
    X = train_df_copy.drop(columns=[state.target_column], axis=1)
    y = train_df_copy[state.target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Store the split data using setter properties
    state.X_train = X_train
    state.X_test = X_test
    state.y_train = y_train
    state.y_test = y_test

    # Identify numeric and categorical columns and store in state
    state.numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    state.categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Data preprocessed. X_train shape: {state.X_train.shape}, X_test shape: {state.X_test.shape}")
    print(f"Numeric columns: {state.numeric_cols[:5]}{'...' if len(state.numeric_cols) > 5 else ''}")
    print(f"Categorical columns: {state.categorical_cols[:5]}{'...' if len(state.categorical_cols) > 5 else ''}")


def select_imputer(state: MLWorkflowState, numeric_type: str = "simple", categorical_type: str = "constant",
                   numeric_strategy: str = "mean", categorical_strategy: str = "constant") -> None:
    """
    Tool to select and configure numeric and categorical imputers.
    Updates the MLWorkflowState directly with selected imputer types and strategies.
    """
    if numeric_type not in ["simple", "knn"]:
        raise ValueError(f"Unsupported numeric imputer type: {numeric_type}. Choose 'simple' or 'knn'.")
    if numeric_type == "simple" and numeric_strategy not in ["mean", "median", "most_frequent", "constant"]:
        raise ValueError(f"Unsupported numeric imputer strategy for 'simple': {numeric_strategy}. Choose 'mean', 'median', 'most_frequent', or 'constant'.")

    if categorical_type not in ["simple", "constant"]:
        raise ValueError(f"Unsupported categorical imputer type: {categorical_type}. Choose 'simple' or 'constant'.")
    if categorical_type == "simple" and categorical_strategy not in ["most_frequent", "constant"]:
            raise ValueError(f"Unsupported categorical imputer strategy for 'simple': {categorical_strategy}. Choose 'most_frequent' or 'constant'.")
    elif categorical_type == "constant" and categorical_strategy != "constant":
        print(f"Warning: When categorical_type is 'constant', categorical_strategy should also be 'constant'. Ignoring '{categorical_strategy}'.")
        categorical_strategy = "constant" # Ensure consistency

    state.numeric_imputer_type = numeric_type
    state.numeric_imputer_strategy = numeric_strategy
    state.categorical_imputer_type = categorical_type
    state.categorical_imputer_strategy = categorical_strategy

    print(f"Selected numeric imputer: {numeric_type} with strategy {numeric_strategy}")
    print(f"Selected categorical imputer: {categorical_type} with strategy {categorical_strategy}")


def select_scaler(state: MLWorkflowState, numeric_scaler: str = "standard", categorical_encoder: str = "onehot") -> None:
    """
    Tool to select and configure a numeric scaler and a categorical encoder.
    Updates the MLWorkflowState directly with selected scaler/encoder types.
    """
    if numeric_scaler not in ["standard", "minmax", "none"]:
        raise ValueError(f"Unsupported numeric scaler: {numeric_scaler}. Choose 'standard', 'minmax', or 'none'.")
    if categorical_encoder not in ["onehot", "ordinal"]:
        raise ValueError(f"Unsupported categorical encoder: {categorical_encoder}. Choose 'onehot' or 'ordinal'.")

    state.numeric_scaler_type = numeric_scaler
    state.categorical_encoder_type = categorical_encoder

    print(f"Selected numeric scaler: {numeric_scaler}")
    print(f"Selected categorical encoder: {categorical_encoder}")


def train_model(state: MLWorkflowState, apply_log_transform: bool = False) -> None:
    """
    Tool to train a scikit-learn pipeline.
    Retrieves pipeline and data from state, fits the pipeline, and updates the stored pipeline.
    Includes an option for log transformation of the target during training.
    """
    if state.pipeline is None:
        raise ValueError("No pipeline found in state for training.")
    if state.X_train is None or state.y_train is None:
        raise ValueError("X_train or y_train not found in state for training.")

    pipeline = state.pipeline
    X_train = state.X_train
    y_train = state.y_train

    print("Training model...")
    
    y_train_fit = y_train
    if apply_log_transform:
        # Apply log transform to target, handling non-positive values
        y_train_fit = np.log(y_train + 1e-6) # Add small epsilon to avoid log(0)

    pipeline.fit(X_train, y_train_fit)
    print("Model training complete.")

    state.pipeline = pipeline # Update the pipeline in state (might contain fitted model)


def evaluate_model(state: MLWorkflowState, apply_exp_transform: bool = False) -> None:
    """
    Tool to evaluate a trained REGRESSION model using common regression metrics.
    Retrieves pipeline and data from state, performs evaluation, and updates state with metrics.
    Includes an option for exponential inverse transformation of predictions.
    """
    try:
        if state.pipeline is None:
            raise ValueError("No pipeline found in state for evaluation.")
        if state.X_test is None or state.y_test is None:
            raise ValueError("X_test or y_test not found in state for evaluation.")

        pipeline = state.pipeline
        X_test = state.X_test
        y_test = state.y_test

        print("Evaluating REGRESSION model...")
        y_pred = pipeline.predict(X_test)
        
        if apply_exp_transform:
            # Inverse transform for evaluation metrics, assumes log transform during training
            y_pred = np.exp(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        state.evaluation_metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

        print("\n--- Model Evaluation Results (Regression Metrics) ---")
        print(f"Target Mean: {y_test.mean():.2f}")
        print(f"Target Std: {y_test.std():.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Relative RMSE (%): {(rmse / y_test.mean() * 100) if y_test.mean() != 0 else float('inf'):.2f}%")
        print(f"R-squared (R²): {r2:.2f}")
        print("--------------------------------")

        state.workflow_summary["evaluation_status"] = "completed successfully"

    except Exception as e:
        print(f"Error during evaluation: {e}")
        state.evaluation_metrics = {"mae": "NA", "mse": "NA", "rmse": "NA", "r2": "NA"}
        state.workflow_summary["evaluation_status"] = "failed"
        state.workflow_summary["error"] = str(e)


def evaluate_classification_model(state: MLWorkflowState) -> None:
    """
    Tool to evaluate a trained CLASSIFICATION model using common classification metrics.
    Retrieves pipeline and data from state, performs evaluation, and updates state with metrics.
    """
    try:
        if state.pipeline is None:
            raise ValueError("No pipeline found in state for evaluation.")
        if state.X_test is None or state.y_test is None:
            raise ValueError("X_test or y_test not found in state for evaluation.")

        pipeline = state.pipeline
        X_test = state.X_test
        y_test = state.y_test

        print("Evaluating CLASSIFICATION model...")
        y_pred = pipeline.predict(X_test)
        
        # Ensure y_test is appropriate for classification metrics (e.g., integer labels)
        # Convert to numpy arrays for metric calculation
        y_test_np = y_test.to_numpy()
        y_pred_np = y_pred # Assuming predict returns class labels

        # Calculate common classification metrics
        accuracy = accuracy_score(y_test_np, y_pred_np)
        
        # Handle cases for binary vs. multi-class for precision/recall/f1/roc_auc
        # If target has more than 2 unique values, it's multi-class
        is_binary = len(np.unique(y_test_np)) == 2

        precision = precision_score(y_test_np, y_pred_np, average='binary' if is_binary else 'weighted', zero_division=0)
        recall = recall_score(y_test_np, y_pred_np, average='binary' if is_binary else 'weighted', zero_division=0)
        f1 = f1_score(y_test_np, y_pred_np, average='binary' if is_binary else 'weighted', zero_division=0)
        
        roc_auc = "N/A"
        try:
            # ROC AUC requires probability estimates, not just class labels
            # Check if the model has predict_proba and if it's binary
            if hasattr(pipeline, 'predict_proba') and is_binary:
                y_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class
                roc_auc = roc_auc_score(y_test_np, y_proba)
            elif hasattr(pipeline, 'decision_function') and is_binary: # For SVC with decision_function
                y_score = pipeline.decision_function(X_test)
                roc_auc = roc_auc_score(y_test_np, y_score)
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC. Error: {e}")
            roc_auc = "Error"

        # Store classification metrics
        state.evaluation_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        print("\n--- Model Evaluation Results (Classification Metrics) ---")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        if roc_auc != "N/A" and roc_auc != "Error":
            print(f"ROC AUC: {roc_auc:.3f}")
        elif roc_auc == "Error":
            print("ROC AUC: Error during calculation (see warning above)")
        else:
            print("ROC AUC: N/A (Not applicable or not computable for this model/data type)")
        
        # Optional: Print classification report for more detail
        # print("\nClassification Report:\n", classification_report(y_test_np, y_pred_np, zero_division=0))
        print("--------------------------------")

        state.workflow_summary["evaluation_status"] = "completed successfully"

    except Exception as e:
        print(f"Error during evaluation: {e}")
        state.evaluation_metrics = {"accuracy": "NA", "precision": "NA", "recall": "NA", "f1_score": "NA", "roc_auc": "NA"}
        state.workflow_summary["evaluation_status"] = "failed"
        state.workflow_summary["error"] = str(e)


def calculate_correlation(state: MLWorkflowState) -> None:
    """
    Calculates and identifies highly correlated numeric features (correlation > 0.825).
    Updates state with columns identified for dropping.
    """
    if state.X_train is None:
        raise ValueError("X_train not found in state for correlation check.")
        
    numeric_df = state.X_train.select_dtypes(include=np.number)
    if numeric_df.empty:
        print("No numeric columns to calculate correlation for.")
        state.dropped_columns_by_correlation = []
        return

    corr_matrix = numeric_df.corr()

    threshold = 0.825 # Using 0.825 as per original template
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.loc[col1, col2]
            if abs(corr_val) > threshold:
                high_corr_pairs.append(tuple(sorted((col1, col2))))

    print("\nHighly correlated numeric feature pairs (correlation > 0.825):")
    drop_cols = []
    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"- {pair[0]} and {pair[1]} (Correlation: {corr_matrix.loc[pair[0], pair[1]]:.3f})")
            drop_cols.append(pair[1])
    else:
        print("No highly correlated numeric features found above threshold.")

    state.dropped_columns_by_correlation = list(set(drop_cols))


def drop_highly_correlated(state: MLWorkflowState) -> None:
    """
    Tool to identify and drop highly correlated numeric features from X_train and X_test.
    Updates the stored DataFrames and modifies numeric_cols and categorical_cols accordingly.
    """
    if state.X_train is None or state.X_test is None:
        raise ValueError("X_train or X_test not found in state for correlation check.")

    X_train_current = state.X_train
    X_test_current = state.X_test
    
    # Calculate and set the columns to drop in the state
    calculate_correlation(state) # This updates state.dropped_columns_by_correlation
    drop_cols_identified = state.dropped_columns_by_correlation

    if drop_cols_identified:
        print(f"Attempting to drop columns: {drop_cols_identified}")
        X_train_cleaned = X_train_current.drop(columns=drop_cols_identified, errors='ignore')
        X_test_cleaned = X_test_current.drop(columns=drop_cols_identified, errors='ignore')

        # Update state with new DataFrames. Properties handle old key cleanup.
        state.X_train = X_train_cleaned
        state.X_test = X_test_cleaned

        state.numeric_cols = [col for col in state.numeric_cols if col not in drop_cols_identified]
        state.categorical_cols = [col for col in state.categorical_cols if col not in drop_cols_identified]
        print(f"Successfully dropped columns: {drop_cols_identified}")
    else:
        print("No highly correlated columns to drop.")


def print_df_head(state: MLWorkflowState, df_type: str = "data") -> dict:
    """
    Tool to print the head of a specified DataFrame (full, X_train, X_test, y_train, or y_test)
    from the MLWorkflowState.
    Returns: A dictionary indicating if the head was printed and a sample.
    """
    df_to_print = None
    df_name = ""

    if df_type == "data" and state.data is not None:
        df_to_print = state.data
        df_name = "Full DataFrame"
    elif df_type == "X_train" and state.X_train is not None:
        df_to_print = state.X_train
        df_name = "X_train"
    elif df_type == "X_test" and state.X_test is not None:
        df_to_print = state.X_test
        df_name = "X_test"
    elif df_type == "y_train" and state.y_train is not None:
        df_to_print = state.y_train.to_frame(name=state.target_column or 'target')
        df_name = "y_train"
    elif df_type == "y_test" and state.y_test is not None:
        df_to_print = state.y_test.to_frame(name=state.target_column or 'target')
        df_name = "y_test"
    
    if df_to_print is not None:
        print(f"\n--- {df_name} Head ---")
        print(df_to_print.head().to_string())
        print("----------------------")
        return {"df_head_printed": True, "df_sample_head": df_to_print.head().to_dict('records'), "df_name": df_name}
    else:
        print(f"No specified DataFrame '{df_type}' found in the state for printing head.")
        return {"df_head_printed": False}

