import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from helpers import get_llm

# Ignore warnings 
warnings.filterwarnings('ignore')

# Turn off matplotlib interactive mode (Streamlit requirement)
plt.ioff()  

class State(TypedDict):
    """
    State class for the AI agent workflow.
    - message: AI-generated response containing code and insights
    - dataset_info: Information about uploaded dataset
    """
    message: str
    dataset_info: str

def generate_analysis(state: State):
    """
    Generates Python code for data visualization.
    """

    # Information about uploaded dataset from state
    dataset_info = state["dataset_info"]

    # Detailed system prompt 
    sys_prompt = f"""Generate Python code to analyze and visualize this dataset: {dataset_info}

    IMPORTANT: The dataset is already loaded as a pandas DataFrame called 'df'. Do NOT use pd.read_csv(). Just use 'df' directly.

    Create 3 different visualizations using matplotlib/seaborn. Structure your response as:
    - A brief dataset overview
    - Code block for each visualization  
    - Short explanation after each visualization

    Format code blocks like:
    ```python
    plt.figure(figsize=(10, 6))
    # Use 'df' directly - it's already loaded
    # Visualization code here using df
    plt.title("Title")
    plt.show()
    ```

    Remember: Use 'df' for the dataset, do not try to read any CSV files.
    """

    # Invoke LLM instance and generate response
    llm = get_llm(temperature=0.2, max_tokens=3000)
    response = llm.invoke([
        SystemMessage(content=sys_prompt), 
        HumanMessage(content="Generate analysis with visualizations.")
    ])

    # Return AI-generated analysis in the LLM response
    return {"message": response.content}

def create_workflow():
    """
    Creating agentic workflow using LangGraph.

    Execution flow:
    START -> generate_analysis() -> END
    """
    builder = StateGraph(State)
    builder.add_node("generate_analysis", generate_analysis)
    builder.add_edge(START, "generate_analysis")
    builder.add_edge("generate_analysis", END)
    return builder.compile()

def execute_code(code, df):
    """
    Executes AI-generated Python code.
    """
    namespace = {'pd': pd, 'plt': plt, 'sns': sns, 'df': df}
    try:
        with warnings.catch_warnings():
            # Ignore warnings during code execution
            warnings.simplefilter("ignore")
            # Execution of Python code
            exec(code, namespace)
        # If figure created, return fig
        if plt.get_fignums():
            fig = plt.gcf()
            return fig, None
    except Exception as e:
        # If execution fails, return error message 
        return None, str(e)
    return None, None

def parse_response(content):
    """
    Takes raw AI-generated content and separates 
    insights/explanations from code blocks.
    """
    parts = re.split(r'```python\s*\n(.*?)```', content, flags=re.DOTALL)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0 and part.strip():  
            result.append(('text', part.strip()))
        elif i % 2 == 1:  
            result.append(('code', part.strip()))
    return result

# Streamlit App with centered layout
st.set_page_config(page_title="AI Data Explorer", layout="centered")

# Title with custom styling
html_title = """
        <div>
            <h2 style="text-align:center;"> 🧞 DataGenie: An Agentic AI Data Explorer </h2>
        </div>
    """   
st.markdown(html_title, unsafe_allow_html=True)
st.markdown("")

# File upload element
uploaded_file = st.file_uploader("**Upload CSV file:**", type=['csv'])

if uploaded_file:
    # Load uploaded dataset
    df = pd.read_csv(uploaded_file)
    # Dataset overview
    st.write(f"**Dataset:** {uploaded_file.name} ({df.shape[0]} rows × {df.shape[1]} columns)")
    st.dataframe(df.head())
    
    # Analysis button element
    if st.button("**Generate Analysis**"):
        # Dataset information for AI Agent
        dataset_info = f"""
        Filename: {uploaded_file.name}
        Shape: {df.shape}
        Columns: {list(df.columns)}
        Data types: {df.dtypes.to_dict()}
        """
        
        # Spinner to show AI Agent is processing data
        with st.spinner("Generating analysis..."):
            # Initialize workflow
            workflow = create_workflow() 
            # Invoke workflow to execute AI agent and return results
            result = workflow.invoke({"dataset_info": dataset_info}) 

            # Parse and display results
            parsed_content = parse_response(result["message"])
            for content_type, content in parsed_content:
                # Display insight
                if content_type == 'text':
                    st.markdown(content)
                # Display code
                elif content_type == 'code':
                    st.code(content, language='python')
                    # Execute code
                    fig, error = execute_code(content, df)
                    # Show error if execution fails
                    if error:
                        st.error(f"Error: {error}")
                    # Show plot
                    elif fig:
                        st.pyplot(fig)
                        plt.close(fig)
