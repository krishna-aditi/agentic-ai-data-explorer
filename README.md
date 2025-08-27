# Agentic AI Data Explorer App
This application implements a simple, but effective AI agent that automatically analyzes any uploaded CSV dataset and generates visualizations with insightful explanations. 

## Workflow
- **Data Ingestion**: Agent receives dataset through CSV upload
- **AI-powered Analysis using GPT-4o**: LLM analyzes the dataset and decides appropriate visualization types based on data characteristics
- **Python Code Generation**: Agent generates Python code using matplotlib/seaborn for visualizations
- **Execution**: Generated code is executed for the uploaded dataset
- **Result Presentation**: Displays visualizations alongside AI-generated insights about what each visualization reveals about the data

## Technologies
- **Streamlit**: Provides the user interface
- **OpenAI GPT-4o**: Powers intelligent analysis and code generation
- **LangGraph**: Orchestrates the agent workflow
- **LangChain**: Manages LLM interactions

## Installation Instructions
### Prerequisites
- Python 3.11+
- [OpenAI API key](https://openai.com/api/)

## Setup steps
1. **Clone repository**
   ```
   $ git clone https://github.com/krishna-aditi/agentic-ai-data-explorer.git
   $ cd ai-data-explorer/src
   ```
2. **Install dependencies**
   ```
   $ pip install -r requirements.txt
   ```
3. **Set OpenAI API key**: create .streamlit/secrets.toml file to configure secrets
   ```
   OPENAI_API_KEY = "openai-api-key"
   ```
4. **Run the application on Streamlit**
   ```
   $ streamlit run app.py
   ```
5. **Open in browser**: navigate to http://localhost:8501

## Usage
- **Upload Dataset**: Drop your CSV file into the upload widget
- **Review Data**: Check the dataset preview
- **Generate Analysis**: Click "Generate Analysis" button to activate the AI agent
- **Explore Results**: View AI-generated visualizations and insights
- **Copy Code**: Use the generated Python code in your own projects
    
## References
- [Agentic VIS Challenge](https://www.visagent.org)
- [Streamlit.io](https://streamlit.io)
- [LangChain Academy](https://github.com/langchain-ai/langchain-academy)
- [OpenAI](https://openai.com/api/)
- [Medium: Introduction to LangGraph: A Beginner’s Guide](https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141)
