import os
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool

import pandas as pd
from pydantic import BaseModel, Field

# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -------------------------------------------------------
# Initialize the language model
# -------------------------------------------------------
model = ChatGroq(model="qwen-qwq-32b")

# -------------------------------------------------------
# Define MCP server launch
# -------------------------------------------------------
server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
)

# -------------------------------------------------------
# Shared schema for CSV tools
# -------------------------------------------------------
args_schema = {
    "path": {"type": "string", "description": "Path to the CSV file."},
    "target_column": {"type": "string", "description": "Target column to predict."},
    "task": {"type": "string", "description": "Type of task: 'regression' or 'classification'."}
}

# -------------------------------------------------------
# Structured output for target column extraction
# -------------------------------------------------------
class TargetColumnInput(BaseModel):
    target_column: str = Field(
        ...,
        description="Name of the target column to predict.",
    )

# -------------------------------------------------------
# ML tools you want to support
# -------------------------------------------------------
ML_TOOL_NAMES = [
    "linear_regression",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
    "gradient_boosting",
]

# -------------------------------------------------------
# Build wrapper for a single ML tool
# -------------------------------------------------------
def build_tool_wrapper(tool, tool_name):
    async def wrapped_ml_tool(path: str, target_column: str, task: str = None):
        path_clean = path.strip() if path else None
        target_column_clean = target_column.strip() if target_column else None

        if not path_clean:
            raise ValueError(f"No path provided for {tool_name} tool.")
        if not target_column_clean:
            raise ValueError(f"No target_column provided for {tool_name} tool.")

        try:
            df_train = pd.read_csv(path_clean)
        except FileNotFoundError:
            raise ValueError(f"File not found: {path_clean}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV: {e}")

        if df_train.empty:
            raise ValueError(f"Dataset at {path_clean} is empty.")

        df_train = df_train.dropna()

        target_extractor = model.with_structured_output(TargetColumnInput)
        extracted_target = target_extractor.invoke(
            f"Extract the exact target column name similar to '{target_column_clean}' "
            f"from columns: {list(df_train.columns)}."
        ).target_column

        if extracted_target not in df_train.columns:
            raise ValueError(
                f"Target column '{extracted_target}' not found in dataset columns: {list(df_train.columns)}"
            )

        if not task:
            if pd.api.types.is_numeric_dtype(df_train[extracted_target]):
                task = "regression"
            else:
                task = "classification"

        training_json = df_train.to_json(orient="records")

        new_args = {
            "training_dataset": training_json,
            "target_column": extracted_target,
            "task": task,
        }

        print(f"[{tool_name}] Calling MCP tool with task={task} and {len(df_train)} rows.")

        result = await tool.ainvoke(new_args)
        print(f"[{tool_name}] Tool returned result.")

        return result

    return wrapped_ml_tool

# -------------------------------------------------------
# Main async function
# -------------------------------------------------------
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            for tool_name in ML_TOOL_NAMES:
                matching_tool = next(
                    (t for t in tools if t.name == tool_name),
                    None
                )

                if matching_tool is not None:
                    wrapper_coro = build_tool_wrapper(matching_tool, tool_name)

                    wrapped_tool = StructuredTool(
                        name=f"{tool_name}_wrapped",
                        description=f"Trains a {tool_name.replace('_', ' ').title()} model on a CSV file.",
                        args_schema=args_schema,
                        coroutine=wrapper_coro,
                        func=wrapper_coro,
                    )

                    tools = [t for t in tools if t.name != tool_name]
                    tools.append(wrapped_tool)

            print("\n=== MCP Tools ===")
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")
            print("==================\n")

            agent = create_react_agent(model, tools)

            print("===== Interactive MCP Chat =====")
            print("Type 'exit' or 'quit' to end the conversation")
            print("Type 'clear' to clear conversation history")
            print("==================================")

            try:
                while True:
                    user_input = input("\nYou: ")

                    if user_input.lower() in ["exit", "quit"]:
                        print("Ending conversation...")
                        break
                    elif user_input.lower() == "clear":
                        print("Conversation history cleared.")
                        continue

                    print("Assistant: ", end="", flush=True)

                    try:
                        response = await agent.ainvoke({"messages": user_input})
                        print("\nKeys:", list(response.keys()))
                        print("Assistant Response:", response["messages"][-1].content)

                    except Exception as e:
                        print(f"\nError during agent response: {e}")

            except Exception as e:
                print(f"\nUnexpected error: {e}")

# -------------------------------------------------------
# Entry point
# -------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
