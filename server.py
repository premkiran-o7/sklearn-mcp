from fastmcp import FastMCP
import pandas as pd
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# --------------------------------------
# Create MCP server
# --------------------------------------
mcp = FastMCP(name="ML Regression & Classification Server")


# --------------------------------------
# Simple hello tool
# --------------------------------------
@mcp.tool()
async def say_hello(name: str) -> dict:
    try:
        return {
            "description": "Hello response",
            "message": f"Hello, {name}!"
        }
    except Exception as e:
        print(f"[say_hello] Error: {e}")
        return {"description": f"Error in say_hello: {str(e)}"}


# --------------------------------------
# Helper: preprocess dataframe
# --------------------------------------
def preprocess_dataframe(df: pd.DataFrame, target_column: str):
    numerical_columns_scalers = {}
    categorical_columns_encoders = {}

    for col in df.columns:
        if col == target_column:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            # numerical_columns_scalers[col] = scaler
        elif pd.api.types.is_string_dtype(df[col]):
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            # categorical_columns_encoders[col] = encoder

    target_encoder = None
    if pd.api.types.is_string_dtype(df[target_column]):
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(df[target_column])

    return df#, numerical_columns_scalers, categorical_columns_encoders, target_encoder


# --------------------------------------
# Helper: regression metrics
# --------------------------------------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "r2_score": r2,
    }


# --------------------------------------
# Linear Regression Tool
# --------------------------------------
@mcp.tool()
async def linear_regression(training_dataset: str, target_column: str, task: str = "regression") -> dict:
    try:
        if task != "regression":
            return {"description": "Skipped: Linear Regression supports only regression tasks."}

        train_df = pd.DataFrame(json.loads(training_dataset))
        train_df = preprocess_dataframe(train_df, target_column)

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = regression_metrics(y_test, preds)

        values = [
            f"Row {i}: Predicted={preds[i]}, Actual={y_test.iloc[i]}"
            for i in range(min(5, len(preds)))
        ]

        return {
            "description": "Linear Regression Results",
            **metrics,
            "predictions": preds.tolist(),
            "values": values,
        }

    except Exception as e:
        print(f"[Linear Regression] Error: {e}")
        return {"description": f"Error in Linear Regression: {str(e)}"}


# --------------------------------------
# Logistic Regression Tool
# --------------------------------------
@mcp.tool()
async def logistic_regression(training_dataset: str, target_column: str, task: str = "classification") -> dict:
    try:
        if task != "classification":
            return {"description": "Skipped: Logistic Regression supports only classification tasks."}

        train_df = pd.DataFrame(json.loads(training_dataset))
        train_df = preprocess_dataframe(train_df, target_column)

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        values = [
            f"Row {i}: Predicted={preds[i]}, Actual={y_test.iloc[i]}"
            for i in range(min(5, len(preds)))
        ]

        return {
            "description": "Logistic Regression Results",
            "accuracy": acc,
            "predictions": preds.tolist(),
            "values": values,
        }

    except Exception as e:
        print(f"[Logistic Regression] Error: {e}")
        return {"description": f"Error in Logistic Regression: {str(e)}"}


# --------------------------------------
# Decision Tree Tool
# --------------------------------------
@mcp.tool()
async def decision_tree(training_dataset: str, target_column: str, task: str = "classification") -> dict:
    try:
        train_df = pd.DataFrame(json.loads(training_dataset))
        train_df = preprocess_dataframe(train_df, target_column)

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        if task == "classification":
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metric_value = accuracy_score(y_test, preds)
            metric_name = "accuracy"
        elif task == "regression":
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = regression_metrics(y_test, preds)
            metric_name = None
        else:
            return {"description": f"Skipped: Decision Tree does not support task '{task}'."}

        values = [
            f"Row {i}: Predicted={preds[i]}, Actual={y_test.iloc[i]}"
            for i in range(min(5, len(preds)))
        ]

        result = {
            "description": f"Decision Tree ({task}) Results",
            "predictions": preds.tolist(),
            "values": values,
        }

        if task == "classification":
            result[metric_name] = metric_value
        else:
            result.update(metrics)

        return result

    except Exception as e:
        print(f"[Decision Tree] Error: {e}")
        return {"description": f"Error in Decision Tree: {str(e)}"}


# --------------------------------------
# Random Forest Tool
# --------------------------------------
@mcp.tool()
async def random_forest(training_dataset: str, target_column: str, task: str = "classification") -> dict:
    try:
        train_df = pd.DataFrame(json.loads(training_dataset))
        train_df = preprocess_dataframe(train_df, target_column)

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        if task == "classification":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metric_value = accuracy_score(y_test, preds)
            metric_name = "accuracy"
        elif task == "regression":
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = regression_metrics(y_test, preds)
            metric_name = None
        else:
            return {"description": f"Skipped: Random Forest does not support task '{task}'."}

        values = [
            f"Row {i}: Predicted={preds[i]}, Actual={y_test.iloc[i]}"
            for i in range(min(5, len(preds)))
        ]

        result = {
            "description": f"Random Forest ({task}) Results",
            "predictions": preds.tolist(),
            "values": values,
        }

        if task == "classification":
            result[metric_name] = metric_value
        else:
            result.update(metrics)

        return result

    except Exception as e:
        print(f"[Random Forest] Error: {e}")
        return {"description": f"Error in Random Forest: {str(e)}"}


# --------------------------------------
# SVM Tool
# --------------------------------------
@mcp.tool()
async def svm(training_dataset: str, target_column: str, task: str = "classification") -> dict:
    try:
        if task != "classification":
            return {"description": "Skipped: SVM tool supports only classification."}

        train_df = pd.DataFrame(json.loads(training_dataset))
        train_df = preprocess_dataframe(train_df, target_column)

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = SVC()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        values = [
            f"Row {i}: Predicted={preds[i]}, Actual={y_test.iloc[i]}"
            for i in range(min(5, len(preds)))
        ]

        return {
            "description": "SVM Classification Results",
            "accuracy": acc,
            "predictions": preds.tolist(),
            "values": values,
        }

    except Exception as e:
        print(f"[SVM] Error: {e}")
        return {"description": f"Error in SVM: {str(e)}"}


# --------------------------------------
# Gradient Boosting Tool (Regression)
# --------------------------------------
@mcp.tool()
async def gradient_boosting(training_dataset: str, target_column: str, task: str = "regression") -> dict:
    try:
        if task != "regression":
            return {"description": "Skipped: Gradient Boosting supports only regression tasks."}

        train_df = pd.DataFrame(json.loads(training_dataset))
        train_df = preprocess_dataframe(train_df, target_column)

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = regression_metrics(y_test, preds)

        values = [
            f"Row {i}: Predicted={preds[i]}, Actual={y_test.iloc[i]}"
            for i in range(min(5, len(preds)))
        ]

        return {
            "description": "Gradient Boosting Regression Results",
            **metrics,
            "predictions": preds.tolist(),
            "values": values,
        }

    except Exception as e:
        print(f"[Gradient Boosting] Error: {e}")
        return {"description": f"Error in Gradient Boosting: {str(e)}"}


# --------------------------------------
# Run server
# --------------------------------------
if __name__ == "__main__":
    mcp.run()
