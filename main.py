from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
import pathlib
import subprocess
import tempfile
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os

# Get Markdown documents from a repository
def get_repo_docs(repo_path):
    repo = pathlib.Path(repo_path)
    
    # Iterate over all Markdown files in the repo (including subdirectories)
    for md_file in repo.glob("**/*.md"):
        # Read the content of the Markdown file
        with open(md_file, "r") as file:
            rel_path = md_file.relative_to(repo)
            yield Document(page_content=file.read(), metadata={"source": str(rel_path)})

# Get source chunks from a repository
def get_source_chunks(repo_path):
    source_chunks = []
    
    # Create a CharacterTextSplitter object for splitting the text
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    
    # Iterate over the documents in the repository
    for source in get_repo_docs(repo_path):
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    return source_chunks

if __name__ == "__main__":
    # Define the path of the repository and Chroma DB
    REPO_PATH = '<absolute path to the repo>/EIPs'
    CHROMA_DB_PATH = f'./chroma/{os.path.basename(REPO_PATH)}'

    vector_db = None

    # Check if Chroma DB exists
    if not os.path.exists(CHROMA_DB_PATH):
        # Create a new Chroma DB
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks = get_source_chunks(REPO_PATH)
        vector_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        vector_db.persist()
    else:
        # Load an existing Chroma DB
        print(f'Loading Chroma DB from {CHROMA_DB_PATH}...')
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    print('Chroma DB Loaded!')

    # Load a QA chain
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    
    # Create a RetrievalQA object using the QA chain and the retriever from vector_db
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_db.as_retriever())

    while True:
        print('\n\n\033[31m' + 'Ask a question' + '\033[m')
        user_input = input()
        print('\033[31m' + qa.run(user_input) + '\033[m')
