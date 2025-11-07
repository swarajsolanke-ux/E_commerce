
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import traceback
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from transformers import pipeline as hf_pipeline
import math
from typing import Tuple, Dict, Any

app = FastAPI(title="E-commerce Chatbot")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}")


FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

print(f"FRONTEND_DIR {FRONTEND_DIR}")
print(f"STATIC_DIR   {STATIC_DIR}")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("static files are loaded ")
else:
    print(f"[WARNING] STATIC_DIR not found: {STATIC_DIR} - CSS/JS will 404")


index_path = os.path.join(FRONTEND_DIR, "home.html")
if os.path.exists(index_path):
    @app.get("/", response_class=HTMLResponse)
    def root():
        return FileResponse(index_path)
else:
    print(f"[WARNING] index.html not found at: {index_path}")


                              
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
print(f"image directory:{IMAGES_DIR}")
DB_PATH = os.path.join(BASE_DIR, "E_commerce", "db", "products.db")
VECTOR_PATH = os.path.join(BASE_DIR, "E_commerce", "vector_store", "index.faiss")

# Device for torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
print(f"vectorstore:{vectorstore}")

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16 if device == "mps" else torch.float32,
    use_safetensors=True,
    device_map="auto" if device == "mps" else None
)


tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.01,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    truncation=True,
    return_full_text=False
)

sentiment_pipe = hf_pipeline("sentiment-analysis")

llm = HuggingFacePipeline(pipeline=pipe)



prompt_template = """
You are a database lookup bot. Your task is to answer EXACTLY what is asked using ONLY the data available in the database context.

Rules:
- Retrieve and answer strictly from the database context provided.
- NEVER add explanations, assumptions, or additional information beyond what the database contains.
- If the question is irrelevant, unclear, or cannot be answered from the database context,
  reply with: "No data found. Sorry, I could not provide an answer. Kindly ask a relevant database-related question."

Context: {context}
Question: {question}

Answer with ONLY the requested value. Examples:
- If asked "rating", answer "3.2"
- If asked "price", answer "159.26"
- If asked "review", answer the full review text
- if asked "other than this say no data found"

Answer:
"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

engine = create_engine(f'sqlite:///{DB_PATH}')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)





origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("middleware added sucessfully")


class QueryRequest(BaseModel):
    query: str




@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "home.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Exact Answer Bot Ready"})



@app.get("/ui")
def ui():
    index_path = os.path.join(FRONTEND_DIR, "home.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h3>UI not found. Put files in the 'frontend' folder.</h3>", status_code=404)





SIMILARITY_THRESHOLD_HIGH = 0.75   
SIMILARITY_THRESHOLD_LOW  = 0.30  

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _clean_metadata(md: dict) -> dict:
    """Return a plain-Python dict with str/float values only."""
    out = {}
    for k, v in (md or {}).items():
        if v is None:
            out[k] = None
            continue
       
        try:
            out[k] = float(v) if (isinstance(v, (int, float)) or hasattr(v, "astype")) else str(v)
        except Exception:
            try:
                out[k] = str(v)
            except Exception:
                out[k] = None
    return out


def is_db_query(query: str, k: int = 1) -> Tuple[bool, Dict[str, Any]]:
   
    try:
        if hasattr(vectorstore, "similarity_search_with_score"):
            results = vectorstore.similarity_search_with_score(query, k=k)
            print(results)
            
            if results:
                top_doc, raw_score = results[0]
                score = _to_float(raw_score)
                metadata = _clean_metadata(getattr(top_doc, "metadata", {}) or {})

              
                explanation = ""
                is_relevant = False
                if score is None:
                    explanation = "no numeric score"
                    is_relevant = False
                elif score > 0.9:
                  
                    is_relevant = score >= SIMILARITY_THRESHOLD_HIGH
                    explanation = "high-is-better"
                else:
                   
                    is_relevant = score <= SIMILARITY_THRESHOLD_LOW
                    explanation = "low-is-better"

                return is_relevant, {"score": score, "metadata": metadata, "explanation": explanation}

    except Exception as e:
      
        print("[DEBUG] similarity_search_with_score error:", e)

    
    try:
        retr = vectorstore.as_retriever()
        docs = retr.get_relevant_documents(query)[:k]
        if not docs:
            return False, {"score": None, "metadata": None, "explanation": "no docs from retriever"}

       
        top = docs[0]
        md = _clean_metadata(getattr(top, "metadata", {}) or {})

       
        qtokens = set([t.lower() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1])
        match_found = False
        fields_to_check = [ "category", "product_id","rating","reviews","cost",]
        for f in fields_to_check:
            val = str(md.get(f, "")).lower() if md.get(f) is not None else ""
            for t in qtokens:
                if t in val:
                    match_found = True
                    break
            if match_found:
                break

        explanation = "retriever_lexical_match" if match_found else "retriever_nomatch_lexical"
        return match_found, {"score": None, "metadata": md, "explanation": explanation}

    except Exception as e:
        print("[DEBUG] retriever fallback error:", e)
        return False, {"score": None, "metadata": None, "explanation": f"error: {str(e)}"}

@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

      
        sentiment_label = None
        sentiment_score = None
        if 'sentiment_pipe' in globals():
            try:
                s = sentiment_pipe(query[:512])[0]
                sentiment_label = s.get("label")
                sentiment_score = float(s.get("score", 0.0))
            except Exception as se:
                print("Sentiment error:", se)

        
        is_relevant, info = is_db_query(query, k=1)
        print(is_relevant,info)
        print(f"[DEBUG] query relevance -> {is_relevant}, info={info}")

        if not is_relevant:
          
            return JSONResponse({
                "response": "No data found. Sorry, I could not provide an answer. Kindly ask a relevant database-related question.",
                "products": [],
                "debug": {
                    "relevance": False,
                    "score": (float(info.get("score")) if info.get("score") is not None else None),
                    "explanation": info.get("explanation"),
                    "sentiment": {"label": sentiment_label, "score": sentiment_score}
                }
            })

        result = qa_chain({"query": query})
        raw_answer = result["result"].strip()
        doc = result["source_documents"][0]
        meta = dict(getattr(doc, "metadata", {}) or {})

        # convert metadata safely to primitives
        def _safe_num(x):
            try:
                return float(x)
            except Exception:
                return x

        product = {
            "name": str(meta.get("name", "Unknown")),
            "cost": _safe_num(meta.get("cost", 0) or 0),
            "rating": _safe_num(meta.get("rating", 0) or 0),
            "review": str(meta.get("review", "No review") or "No review"),
            "image": str(meta.get("image_path")) if meta.get("image_path") else None
        }

        lower_q = query.lower()
        if "rating" in lower_q:
            answer = str(product["rating"])
        elif "price" in lower_q or "cost" in lower_q:
            try:
                answer = f"Rs{float(product['cost']):.2f}"
            except Exception:
                answer = str(product["cost"])
        elif "review" in lower_q:
            answer = product["review"]
        # elif "name" in lower_q:
        #     answer = product["name"]
        else:
            answer = raw_answer.split("Answer:")[-1].strip()

      
        answer = re.sub(r"[^0-9.$a-zA-Z\s]", "", answer).strip()
        print("answer:{answer}")
        if not answer or answer == "0":
            answer = "Not found"

        return {
            "response": answer,
            "products": [product],
            "debug": {
                "relevance": True,
                "score": (float(info.get("score")) if info.get("score") is not None else None),
                "explanation": info.get("explanation"),
                "sentiment": {"label": sentiment_label, "score": sentiment_score}
            }
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")




#running api endpoint
# @app.post("/query")
# def search_products(request: QueryRequest):
#     try:
#         query = request.query.lower().strip()

#         # RAG call
#         result = qa_chain({"query": query})
#         raw_answer = result["result"].strip()
#         doc = result["source_documents"][0]
#         meta = doc.metadata

#         # Extract exact product (guard keys)
#         product = {
#             "name": meta.get("name", "Unknown"),
#             # Some metadata fields may be strings, handle safely
#             "cost": float(meta.get("cost", 0) or 0),
#             "rating": float(meta.get("rating", 0) or 0),
#             "review": meta.get("review", "No review"),
#             "image": meta.get("image_path", None)
#         }

#         # Determine which value to return
#         if "rating" in query:
#             answer = str(product["rating"])
#         elif "price" in query or "cost" in query:
#             answer = f"Rs{product['cost']:.2f}"
#         elif "review" in query:
#             answer = product["review"]
#         elif "name" in query:
#             answer = product["name"]
#         else:
#             # fallback - trust the chain result
#             answer = raw_answer.split("Answer:")[-1].strip()

#         # sanitize and minimal fallback
#         answer = re.sub(r"[^0-9.$a-zA-Z\s]", "", answer).strip()
#         if not answer or answer == "0":
#             answer = "Not found"

#         return {
#             "response": answer,
#             "products": [product]
#         }

#     except Exception as e:
#         tb = traceback.format_exc()
#         print(tb)
     
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/images/{filename}")
def get_image(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(IMAGES_DIR, safe_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    uvicorn.run("backend:app", host="192.168.5.224", port=8000, reload=True)



#  inet 192.168.5.224 netmask 0xffffff00 broadcast 192.168.5.255

