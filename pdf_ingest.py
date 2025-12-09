import os
from  google import genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings  
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback
import shutil
from langchain_core.runnables import RunnableSequence,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GEMINI_API_KEY = "AIzaSyA8hidHwXo8J4HYdmimQadrcub8SUk6hMI"
#api_key=GEMINI_API_KEY
client = genai.Client()
file_path = "/Users/swarajsolanke/Desktop/Updated/AI-Assignment.pdf"

def get_gemini_embeddings(texts):
    """Generate embeddings using Gemini API"""
    
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts,
        config=genai.types.EmbedContentConfig(output_dimensionality=768),
        
    )
    return [emb.values for emb in result.embeddings]

def ingest_pdf_file(file_path):
    # Load PDF
    loader = PyMuPDFLoader(file_path)
    print(f"Loaded PDF: {loader}")
    documents = loader.load()
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )


    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")
    
   
    text_contents = [text.page_content for text in texts]
    embeddings = get_gemini_embeddings(text_contents)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create Document objects with embeddings
    embedding_docs = [
        Document(page_content=text.page_content, metadata=text.metadata)
        for text in texts
    ]
    
    # Create and save FAISS index
    persist_directory = "/Users/swarajsolanke/Chatbot/E_commerce/vect/vector_store"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vectordb = FAISS.from_embeddings(
        [(doc.page_content, emb) for doc, emb in zip(embedding_docs, embeddings)],
        embedding=None,  
        normalize_L2=True,

    )
    vectordb.save_local(persist_directory)
    print(f"FAISS index saved to {persist_directory}")
    return vectordb



persist_directory = "/Users/swarajsolanke/Chatbot/E_commerce/vect/vector_store"
vectordb = FAISS.load_local(
    folder_path=persist_directory,
    embeddings=get_gemini_embeddings 
)


llm=ChatOllama(
    model="gemma2:2b",
    temperature=0.7,
)

reteriver=vectordb.as_retriver(search_kwargs={"k":3})

auth_router=APIRouter()

@auth_router.post("/ingest-pdf")
def ingest_pdf(file: UploadFile = File(...)):
    temp_dir="/Users/swarajsolanke/Chatbot/E_commerce/data"
    os.makedirs(temp_dir,exist_ok=True)
    temp_file_path=os.path.join( temp_dir,file.filename)
    with open(temp_file_path,"wb") as f:
        shutil.copyfileobj(file.file,f)
    try:
        ingest_pdf_file(temp_file_path)
        return JSONResponse({"success":True,"message":"PDF ingested successfully"})
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500,f"Error ingesting PDF:{str(e)}")



def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])



template="""
you are an helpful assistant that helps user to answer the asked query based on the pdf user has shared.do not make up answer give answer what user has asked 
context:{context}
Question:{question}

Answer: """

prompt = ChatPromptTemplate.from_template(template)


chain=(
{"context":reteriver| format_docs,"question":RunnablePassthrough()}
|prompt|llm
|StrOutputParser()

)


def get_response(query:str)->str:
    try:
        response=chain.invoke({"question":query})
        return response
    except Exception as e:
        logging.error(traceback.format_exc())
        return "Sorry,an error occurred while processing your request."
    

# if __name__ == "__main__":
#     vectordb = ingest_pdf_file(file_path)
