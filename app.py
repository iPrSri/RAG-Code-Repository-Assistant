import streamlit as st

# Set page config at the very beginning
st.set_page_config(layout="wide")

# Custom CSS to reduce the gap between sidebar and main content
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stChatMessage {
        padding-bottom: 20px;
    }
    # Ensure the chat container doesn't exceed the page height
    [data-testid="stVerticalBlock"] {
        max-height: calc(100vh - 200px);
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


import os
import nest_asyncio
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
import requests
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from streamlit_agraph import agraph, Node, Edge, Config
import re
from dotenv import load_dotenv
import base64

# Load environment variables from .env file
load_dotenv()

st.cache_data.clear()
st.cache_resource.clear()

# if st.session_state.messages:
#     st.experimental_rerun()

# Use environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


# Apply nest_asyncio patch
nest_asyncio.apply()

@st.cache_resource
def setup_environment():
    # Set up OpenAI for text generation
    llm = OpenAI(model="gpt-4o-2024-08-06", temperature=0)
    Settings.llm = llm
    
    # Set up OpenAI for embeddings
    Settings.embed_model = OpenAIEmbedding()

def parse_github_url(url):
    parts = url.strip('/').split('/')
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo

def find_potential_files(prompt, repo_structure):
    # Create a set of all file paths in the repo
    all_files = {item['path'] for item in repo_structure if item['type'] != 'tree'}
    
    # Find all word sequences that could be file paths
    potential_files = re.findall(r'\b[\w./\-_]+\b', prompt.lower())
    
    # Return the intersection of potential files and actual files
    return set(potential_files) & all_files


def get_file_content(owner, repo, path, github_token):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = base64.b64decode(response.json()["content"]).decode("utf-8")
        return content
    return None

def create_index_from_github(owner, repo, branch, github_token):
    github_client = GithubClient(github_token=github_token, verbose=True)
    documents = GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        use_parser=False,
        verbose=False
    ).load_data(branch=branch)
    
    index = VectorStoreIndex.from_documents(documents)
    return index, len(documents)

def get_repo_structure(owner, repo, github_token, main_branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{main_branch}?recursive=1"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    return response


def build_graph(files):
    nodes = []
    edges = []
    node_ids = set()

    for file in files:
        path = file["path"]
        parts = path.split("/")
        for i in range(len(parts)):
            node_id = "/".join(parts[:i+1])
            if node_id not in node_ids:
                node_type = "file" if i == len(parts) - 1 and file.get("type") != "tree" else "folder"
                nodes.append(Node(id=node_id, label=parts[i], shape=node_type))
                node_ids.add(node_id)
                if i > 0:
                    parent_id = "/".join(parts[:i])
                    edges.append(Edge(source=parent_id, target=node_id))

    return nodes, edges

def display_file_structure(files):
    def sort_key(item):
        return (1 if item['type'] == 'tree' else 0, item['path'])

    sorted_files = sorted(files, key=sort_key)

    for file in sorted_files:
        path = file['path']
        indent = '  ' * path.count('/')
        icon = '📁 ' if file['type'] == 'tree' else '📄 '
        st.text(f"{indent}{icon}{path.split('/')[-1]}")
        

def visualize_repo_structure(owner, repo, github_token):
    response = get_repo_structure(owner, repo, github_token, "master")

    files = None

    if response.status_code == 404:
        response = get_repo_structure(owner, repo, github_token, "main")

    if response.status_code == 200:
        files = response.json()["tree"]
    else:
        return None

    return files


def main():
    st.title("Understand your Github Repository")    
    col1, col2 = st.columns([8, 2])

    # Move github_url, owner, and repo to a broader scope
    github_url = None
    owner = None
    repo = None



    # Sidebar for API keys
    with st.sidebar:
        st.header("API Keys")

        # Initialize session state for API keys if not present
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = OPENAI_API_KEY or ""
        if 'github_token' not in st.session_state:
            st.session_state.github_token = GITHUB_TOKEN or ""
    
        # Display current API keys (masked)
        st.text_input("Current OpenAI API Key", value="*" * len(st.session_state.openai_api_key), disabled=True)
        st.text_input("Current GitHub Token", value="*" * len(st.session_state.github_token), disabled=True)
    
        # Option to update API keys
        if st.button("Update API Keys"):
            st.session_state.update_keys = True
        
        if st.session_state.get('update_keys', False):
            new_openai_key = st.text_input("New OpenAI API Key", type="password")
            new_github_token = st.text_input("New GitHub Token", type="password")
            
            if st.button("Save New Keys"):
                st.session_state.openai_api_key = new_openai_key
                st.session_state.github_token = new_github_token
                st.session_state.update_keys = False
                st.success("API keys updated successfully!")

        if st.session_state.openai_api_key and st.session_state.github_token:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
            os.environ["GITHUB_TOKEN"] = st.session_state.github_token
            setup_environment()
            st.success("API keys set successfully!")
        else:
            st.warning("Please enter both API keys to proceed.")
            st.stop()

    # Main content
    with col1:
        github_url = st.text_input("Enter GitHub Repository URL")
        if github_url:
            owner, repo = parse_github_url(github_url)
            st.write(f"Repository: {owner}/{repo}")

            if st.button("Index Repository"):
                with st.spinner("Indexing repository..."):
                    try:
                        index, doc_count = create_index_from_github(owner, repo, "master", st.session_state.github_token)
                    except Exception as e:
                        st.warning("No main to index 'master' branch. Trying 'main' branch...")
                        try:
                            index, doc_count = create_index_from_github(owner, repo, "main", st.session_state.github_token)
                        except Exception as e:
                            st.error(f"Failed to index repository: {str(e)}")
                            st.stop()

                    st.session_state['index'] = index
                    st.session_state['chat_memory'] = ChatMemoryBuffer.from_defaults(token_limit=1500)
                    st.success(f"Indexed {doc_count} documents from {owner}/{repo}")
                    
                    # Fetch and store repository structure after successful indexing
                    files = visualize_repo_structure(owner, repo, st.session_state.github_token)
                    if files:
                        st.session_state['repo_structure'] = files

        if 'index' in st.session_state:
            st.subheader(f"Ask me about {repo} project")
            
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            
            # Create a container for the chat with a scrollbar
            chat_container = st.container()

            st.write("")
            st.write("")

            # React to user input
            if prompt := st.chat_input("What would you like to know about your repository?"):
                # Display user message in chat message container
                #st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Generating response..."):
                    query_engine = st.session_state['index'].as_query_engine(
                        memory=st.session_state['chat_memory']
                    )
                    response = query_engine.query(prompt)
                    answer = response.response.strip()

                    potential_files = find_potential_files(prompt, st.session_state['repo_structure'])
                    for file_path in potential_files:
                        file_content = get_file_content(owner, repo, file_path, st.session_state.github_token)
                        if file_content:
                            # snippet = file_content[:500] + ("..." if len(file_content) > 500 else "")
                            # answer += f"\n\nYou mentioned the file '{file_path}'. Here's a snippet:\n\n```\n{snippet}\n```\n\nExplanation of the file contents:\n"
                            try:
                                explanation = query_engine.query(f"Explain the contents of this file: {file_content[:1000]}").response
                                answer += explanation
                            except Exception as e:
                                pass
                                #st.error(f"An error occurred while generating explanation: {str(e)}")
                                #answer += "Unable to generate explanation due to an error."

                # Display assistant response in chat message container
                # with st.chat_message("assistant"):
                #     st.markdown(answer)
                # Add assistant response to chat history

                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Display chat messages from history in the scrollable container
                with chat_container:
                    st.markdown("""
                        <style>
                            .stChatFloatingInputContainer {
                                bottom: 20px;
                            }
                            .stChatMessage {
                                padding-bottom: 20px;
                            }
                        </style>
                    """, unsafe_allow_html=True)

                    for i in range(len(st.session_state.messages) - 1, -1, -2):
                        if i > 0:  # If there's a pair of messages
                            with st.chat_message("user"):
                                st.markdown(st.session_state.messages[i-1]["content"])
                            with st.chat_message("assistant"):
                                st.markdown(st.session_state.messages[i]["content"])
                        else:  # If there's an odd message at the end (should be a user message)
                            with st.chat_message("user"):
                                st.markdown(st.session_state.messages[i]["content"])
                    # for message in reversed(st.session_state.messages):
                    #     with st.chat_message(message["role"]):
                    #         st.markdown(message["content"])

                # Add some space after the chat container
                st.write("")
                st.write("")

    # Repository structure visualization
    with col2:
        if 'repo_structure' in st.session_state:
            st.subheader("Repository Structure")
            
            # Create and store the repository visualization)
            display_file_structure(st.session_state['repo_structure'])

if __name__ == "__main__":
    main()
