import os
import argparse
import pandas as pd
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

config_list = [
    {
        "model": "gpt-4.5-turbo",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    {
        "model": "gemini-pro", 
        "api_key":os.getenv("GEMINI_API_KEY"),
        "api_type": "google"
    }
]

config_list = [c for c in config_list if c["api_key"]]

if not config_list:
    print("Error: Please set OPENAI_API_KEY or GEMINI_API_KEY environment variable.")
    exit(1)

llm_config = {
    "config_list": config_list,
    "temperature": 0,
}

def build_agents(work_dir="coding"):

    user_proxy = UserProxyAgent(
        name="Admin",
        system_message="A human admin. Execute the code written by the Engineer and report errors.",
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": work_dir,
            "use_docker": False,
        },
        human_input_mode="ALWAYS", 
    )

    # 2. Planner (The Strategist)
    planner = AssistantAgent(
        name="Planner",
        system_message="""
        You are a Data Strategist.
        Given a user query and a dataframe schema, suggest a logical plan to answer the query.
        Do not write code. Just explain the steps in English.
        """,
        llm_config=llm_config,
    )

    # 3. Engineer (The Coder)
    engineer = AssistantAgent(
        name="Engineer",
        system_message="""
        You are a Python Data Engineer.
        You write python code to solve the Planner's strategy.
        - You MUST use pandas.
        - The data is already in a CSV file (path provided in context).
        - Use `print()` to output the result so the Admin can see it.
        - Wrap code in ```python code blocks.
        """,
        llm_config=llm_config,
    )

    # 4. Reporter (The Analyst)
    reporter = AssistantAgent(
        name="Reporter",
        system_message="""
        You are a Data Reporter.
        Review the output of the code execution.
        Answer the user's original question in a clear, professional sentence based on the numbers.
        If the code failed, ask the Engineer to fix it.
        """,
        llm_config=llm_config,
    )

    return user_proxy, planner, engineer, reporter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    # 1. Context Loading (Ingest)
    if not os.path.exists(args.csv):
        print(f"Error: File {args.csv} not found.")
        exit(1)
        
    df = pd.read_csv(args.csv)
    schema_info = f"Columns: {list(df.columns)}\nSample Data:\n{df.head(2).to_string()}"

    # 2. Initialize Agents
    user_proxy, planner, engineer, reporter = build_agents()

    # 3. Create Group Chat
    groupchat = GroupChat(
        agents=[user_proxy, planner, engineer, reporter], 
        messages=[], 
        max_round=12
    )
    
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # 4. Start the Conversation
    initial_message = f"""
    Context: We have a CSV file located at '{args.csv}'.
    Schema: {schema_info}
    
    User Query: {args.query}
    
    Process:
    1. Planner: Suggest a plan.
    2. Engineer: Write code to check the data.
    3. Admin: Execute code.
    4. Reporter: Summarize.
    """

    print(f"\nStarting AutoGen Group Chat for: '{args.query}'\n")
    user_proxy.initiate_chat(
        manager,
        message=initial_message
    )