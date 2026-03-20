import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# STEP 1: Extracting Text from PDFs
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("21B91A6145_od.pdf")
docs = loader.load()

# Creating own Metadata for PDF Chunks
for i in docs:
    i.metadata = {"source": "21B91A6145_od.pdf",
                  "developer": "Avinash"}
    
    
# STEP 2: Splitting the Document into CHUNKS
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50)

chunks = splitter.split_documents(docs)
# print(chunks)


# STEP 3: Creating Embeddings for the Chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)


# STEP 4: Create and Store Embeddings in Local Vector Store
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./VectorStore/"
)


#Re-Use the Vector Database
vectorstore_persist = Chroma(
    persist_directory="./VectorStore/",
    embedding_function=embedding_model
)


# STEP 5: Semantic Search
vectorstore_persist.similarity_search("What are the contact numbers in the pdf?", k= 3)


# Talk to LLM
context = vectorstore_persist.similarity_search("What are the contact numbers in the pdf?", k= 3)
response = llm.invoke(f"What are the contact numbers in the pdf? You can answer using the following context: {context}")
print(response.content)