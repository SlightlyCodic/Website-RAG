# Website RAG Chatbot

Website RAG Chatbot is a Python-based Streamlit web app that lets you chat with the content of any public website using Retrieval-Augmented Generation (RAG). It parses websites, builds a vector store from their content, and enables LLM-powered conversations based solely on the extracted data.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install -r requirements.txt
```

## Usage

1. Add your OpenRouter API key to `.streamlit/secrets.toml`:

```toml
API_KEY = "your_openrouter_api_key_here"
```

2. Run the app:

```bash
streamlit run app.py
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
