# RAG-Code-Repository-Assistant

This is a full-stack Retrieval Augmented Generation (RAG) App that helps browse and query new code-bases via chat. We generate and store embeddings for a given repository in advance, and then when a user asks a question, we retrieve relevant code snippets and provide them to the LLM as context.

<img width="964" alt="image" src="https://github.com/user-attachments/assets/f79bea0d-a799-413b-b78f-88f66b385752">




## Setup

1. Clone this repository:
   ```
   git clone https://github.com/iPrSri/RAG-Code-Repository-Assistant
   cd RAG-Code-Repository-Assistant
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the script:
   ```
   streamlit run app.py
   ```

5. When prompted, enter your OpenAI API key and GitHub token.

## Usage

After indexing, you can ask questions about a given repository. Type 'quit' to move to the next repository or exit the program.
