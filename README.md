# GitHub Repository Q&A Tool

This project creates a Q&A system for GitHub repositories using LlamaIndex and OpenAI.

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
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
   python tech16_class3_rag_pramod.py
   ```

5. When prompted, enter your OpenAI API key and GitHub token.

## Usage

The script will index two GitHub repositories:
1. domarps/papers-i-read
2. domarps/llama2.c

After indexing, you can ask questions about each repository. Type 'quit' to move to the next repository or exit the program.