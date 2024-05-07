from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Initialize embedding model for retrieval (sentence similarity)
BATCH_SIZE = 32
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
retriever_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
retriever_model = HuggingFaceEmbeddings(
    model_name=retriever_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': BATCH_SIZE},
)

def retrieve_relevant_documents_using_rag(search_results, content_key, question, chunk_size=512, chunk_overlap=128, top_k=10):
    """
    Takes in search results and a query question, processes and splits the documents,
    and retrieves relevant documents using a RAG approach.

    Args:
        search_results (list of dict): A list of dictionaries containing web-scraped data.
        question (str): The query question for retrieving relevant documents.
        content_key (str): The key in the dictionary containing the text content.
        chunk_size (int): The maximum size of the text chunks.
        chunk_overlap (int): The overlap between consecutive text chunks.
        top_k (int): The number of relevant documents to retrieve.

    Returns:
        list: A list of relevant document chunks.
    """
    # Create LangChain documents from search results
    documents = []
    for result in search_results:
        page_content = result.pop(content_key, None)  # Extract the text content, remaining keys are metadata
        if page_content is not None:
            documents.append(Document(page_content=page_content, metadata=result))

    # Split documents into smaller chunks (if needed, based on document size)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_documents = text_splitter.split_documents(documents)

    # Initialize ChromaDB vector store to index the document chunks
    db = Chroma.from_documents(
        documents=split_documents,
        embedding=retriever_model,
    )

    # Retrieve the most relevant chunks for the given question
    relevant_docs = db.max_marginal_relevance_search(question, k=top_k)

    return relevant_docs