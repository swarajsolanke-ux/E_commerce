
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

import os
import re
import traceback
import json
import logging
from typing import Tuple, Dict, Any, List, Set, Optional
from difflib import SequenceMatcher
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import uvicorn

try:
    from langchain_classic.chains import RetrievalQA
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
DB_PATH = os.path.join(BASE_DIR, "db", "products_DB.db")
VECTOR_DIR = os.path.join(BASE_DIR, "vect", "vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model_path = "/Users/swarajsolanke/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

app = FastAPI(title="E-commerce Chatbot")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, 
                   allow_methods=["*"], allow_headers=["*"])

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = None
try:
    print("Loading FAISS vectorstore...")
    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully")
except Exception as e:
    print(f"Failed to load vectorstore: {e}")

# ============= INTENT DETECTION PATTERNS =============

GREETING_WORDS = {
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
    "good evening", "hii", "helo", "sup", "yo", "howdy"
}

# Comprehensive query patterns
QUERY_PATTERNS = {
    "list_all": [
        r"\b(show|list|display|give)\s+me\s+(all|everything)",
        r"\ball\s+(products?|items?|things?)",
        r"\bwhat\s+(products?|items?)\s+(do\s+you\s+have|are\s+available)",
        r"\bwhat\s+can\s+i\s+buy",
        r"\bavailable\s+items?",
        r"\blist\s+of\s+products?",
        r"\bentire\s+(catalog|inventory)",
    ],
    
    "product_by_name": [
        r"\bdo\s+you\s+have\s+(.+?)\??$",
        r"\bsearch\s+for\s+[\"']?(.+?)[\"']?",
        r"\bshow\s+me\s+(.+?)\s*$",
        r"\bfind\s+(product\s+)?named?\s+[\"']?(.+?)[\"']?",
        r"\blooking\s+for\s+(.+?)\s*$",
        r"\b(nike|iphone|samsung|redmi|adidas|sony|apple)\s+",
    ],
    
    "category": [
        r"\b(show|list|display|give)\s+(me\s+)?(all\s+)?(?:the\s+)?(electronics?|clothing|sports?|home|fashion|tech|apparel)",
        r"\b(electronics?|clothing|sports?|home|fashion|tech)\s+(products?|items?|section)",
        r"\bwhat\s+(electronics?|clothing|sports?|home)\s+(do\s+you\s+have|are\s+available)",
        r"\banything\s+in\s+(electronics?|clothing|sports?|home)",
        r"\b(electronics?|clothing|sports?|home)\s+category",
    ],
    
    "cheapest": [
        r"\b(cheapest|most\s+affordable|lowest\s+price|least\s+expensive)",
        r"\bminimum\s+price",
        r"\bbudget\s+friendly",
        r"\blowest\s+cost",
    ],
    
    "most_expensive": [
        r"\b(most\s+expensive|highest\s+price|premium|luxury|costliest)",
        r"\bmaximum\s+price",
        r"\bpriciest",
    ],
    
    "price_range": [
        r"\bbetween\s+.*?[₹rs\.]*\s*(\d+)\s+(?:and|to|-)\s+[₹rs\.]*\s*(\d+)",
        r"\bunder\s+(?:budget\s+)?[₹rs\.]*\s*(\d+)",
        r"\bbelow\s+[₹rs\.]*\s*(\d+)",
        r"\bless\s+than\s+[₹rs\.]*\s*(\d+)",
        r"\babove\s+[₹rs\.]*\s*(\d+)",
        r"\bover\s+[₹rs\.]*\s*(\d+)",
        r"\bmore\s+than\s+[₹rs\.]*\s*(\d+)",
        r"\bwithin\s+[₹rs\.]*\s*(\d+)",
        r"\bprice\s+range",
        r"\bbudget\s+of\s+[₹rs\.]*\s*(\d+)",
    ],
    
    "highest_rating": [
        r"\b(highest|best|top)\s+(rated?|rating|reviewed?)",
        r"\btop\s+rated",
        r"\bbest\s+quality",
        r"\bmost\s+popular",
        r"\bgood\s+reviews?",
        r"\brated\s+above\s+(\d+)",
        r"\b(\d+)\s+stars?\s+(or\s+)?(above|more|higher)",
    ],
    
    "compare": [
        r"\bcompare\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+?)(?:\?|$)",
        r"\b(.+?)\s+(?:or|vs\.?|versus)\s+(.+?)\??$",
        r"\bwhich\s+is\s+(better|cheaper|best|good)",
        r"\bdifference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\bshould\s+i\s+buy\s+(.+?)\s+or\s+(.+?)\??$",
        r"\bwhich\s+one\s+.*?\:\s*(.+?)\s+or\s+(.+?)\??$",
    ],
    
    "recommend": [
        r"\brecommend\s+(?:a\s+)?(?:good\s+)?(.+?)(?:\s+under\s+[₹rs\.]*\s*(\d+))?",
        r"\bsuggest\s+(?:a\s+)?(.+?)(?:\s+under\s+[₹rs\.]*\s*(\d+))?",
        r"\bwhat\s+should\s+i\s+(buy|get|purchase)",
        r"\bbest\s+(?:for\s+)?(.+?)(?:\s+under\s+[₹rs\.]*\s*(\d+))?",
        r"\bgift\s+for\s+(.+?)(?:\s+under\s+[₹rs\.]*\s*(\d+))?",
        r"\bneed\s+(?:a\s+)?(.+?)(?:\s+under\s+[₹rs\.]*\s*(\d+))?",
        r"\bbest[-\s]selling",
    ],
}

# ============= HELPER FUNCTIONS =============

def _all_docs_from_vectorstore() -> List[Any]:
    """Get all documents from vectorstore"""
    if not vectorstore:
        return []
    try:
        return list(getattr(vectorstore.docstore, "_dict", {}).values())
    except:
        return []

def doc_to_metadata(doc) -> Dict[str, Any]:
    """Extract metadata from document"""
    if doc is None:
        return {}
    if isinstance(doc, dict):
        return doc.get("metadata", doc)
    m = getattr(doc, "metadata", None)
    if isinstance(m, dict):
        return m
    try:
        return dict(doc.__dict__)
    except:
        return {}

def parse_price(value: Any) -> float:
    """Parse price value to float"""
    try:
        if not value:
            return 0.0
        s = re.sub(r"[^\d.]", "", str(value))
        return float(s) if s else 0.0
    except:
        return 0.0

def parse_rating(value: Any) -> float:
    """Parse rating value to float"""
    try:
        return float(str(value).strip())
    except:
        return 0.0

def get_primary_category(metadata: Dict[str, Any]) -> str:
    """Get primary category from metadata"""
    for k in ("category_2", "category_1", "category_3", "product_category", "main_category"):
        v = str(metadata.get(k, "")).strip()
        if v and v.lower() not in ["nan", "none", "", "null"]:
            return v
    return ""

def doc_to_card(doc) -> dict:
    """Convert document to product card"""
    meta = doc_to_metadata(doc)
    image = None
    if meta.get("image_path"):
        image = f"/images/{os.path.basename(meta.get('image_path'))}"
    elif meta.get("image_urls"):
        urls = meta.get("image_urls")
        if isinstance(urls, (list, tuple)) and urls:
            image = urls[0]
        elif isinstance(urls, str) and urls:
            image = urls
    
    return {
        "title": meta.get("title") or meta.get("name") or "Unknown Product",
        "selling_price": meta.get("selling_price") or meta.get("cost") or meta.get("price") or "N/A",
        "product_rating": parse_rating(meta.get("product_rating") or meta.get("rating") or 0),
        "description": (meta.get("description") or meta.get("about") or "No description available")[:280],
        "image": image
    }

def fuzzy_match_product(query: str, docs: List[Any], threshold: float = 0.4) -> List[Any]:
    """Fuzzy match product names"""
    query_lower = query.lower()
    matches = []
    
    for doc in docs:
        title = doc_to_card(doc)["title"].lower()
        ratio = SequenceMatcher(None, query_lower, title).ratio()
        
        # Also check if query words are in title
        query_words = set(query_lower.split())
        title_words = set(title.split())
        word_overlap = len(query_words & title_words) / max(len(query_words), 1)
        
        score = max(ratio, word_overlap)
        
        if score >= threshold:
            matches.append((doc, score))
    
    # Sort by score
    matches.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in matches]

def extract_category(query: str) -> Optional[str]:
    """Extract category from query"""
    query_lower = query.lower()
    
    categories = {
        'electronics': ['electronic', 'electronics', 'tech', 'gadget', 'device', 'phone', 
                       'laptop', 'computer', 'mobile', 'tablet', 'iphone', 'samsung'],
        'sports': ['sport', 'sports', 'fitness', 'gym', 'exercise', 'athletic', 'running'],
        'clothing': ['cloth', 'clothing', 'apparel', 'fashion', 'wear', 'shirt', 'pant', 
                    'jeans', 'dress', 'tshirt', 't-shirt', 'jacket'],
        'home': ['home', 'house', 'kitchen', 'furniture', 'appliance', 'decor']
    }
    
    for cat, keywords in categories.items():
        if any(kw in query_lower for kw in keywords):
            return cat
    
    return None

def extract_price_range(query: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract price range from query"""
    query_lower = query.lower()
    
    # Between X and Y
    between = re.search(r'between\s+.*?[₹rs\.]*\s*(\d+)\s+(?:and|to|-)\s+[₹rs\.]*\s*(\d+)', query_lower)
    if between:
        return float(between.group(1)), float(between.group(2))
    
    # Under/below X
    under = re.search(r'(under|below|less\s+than|within)\s+(?:budget\s+)?[₹rs\.]*\s*(\d+)', query_lower)
    if under:
        return 0, float(under.group(2))
    
    # Above/over X
    above = re.search(r'(above|over|more\s+than)\s+[₹rs\.]*\s*(\d+)', query_lower)
    if above:
        return float(above.group(2)), float('inf')
    
    # Just a number
    just_num = re.search(r'[₹rs\.]\s*(\d+)', query_lower)
    if just_num:
        return 0, float(just_num.group(1))
    
    return None, None

def extract_rating_threshold(query: str) -> Optional[float]:
    """Extract rating threshold from query"""
    match = re.search(r'(?:rated\s+)?(?:above|over|more\s+than)\s+(\d+(?:\.\d+)?)', query.lower())
    if match:
        return float(match.group(1))
    
    match = re.search(r'(\d+(?:\.\d+)?)\s+stars?\s+(?:or\s+)?(?:above|more|higher)', query.lower())
    if match:
        return float(match.group(1))
    
    return None

def detect_query_type(query: str) -> Tuple[str, Dict[str, Any]]:
    """Detect query type and extract parameters"""
    query_lower = query.lower()
    
    # Check each pattern type
    for query_type, patterns in QUERY_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                params = {"match_groups": match.groups() if match.groups() else []}
                logger.info(f"Detected query type: {query_type} with params: {params}")
                return query_type, params
    
    # Default to semantic search
    return "semantic", {}

def build_context(docs: List[Any], max_docs: int = 30) -> str:
    """Build context from documents"""
    lines = []
    for doc in docs[:max_docs]:
        card = doc_to_card(doc)
        cat = get_primary_category(doc_to_metadata(doc))
        lines.append(
            f"Product: {card['title']} | Price: Rs {card['selling_price']} | "
            f"Rating: {card['product_rating']}/5 | Category: {cat}"
        )
    return "\n".join(lines) if lines else "No products available."

# ============= LLM SETUP =============

llm = None
qa_chain = None

if LLM_AVAILABLE and vectorstore:
    try:
        print("Loading LLM...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map="auto" if device == "mps" else None
        )
        tokenizer.pad_token = tokenizer.eos_token

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=600,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True,
            return_full_text=False
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        PROMPT = PromptTemplate(
            template="""You are a helpful e-commerce assistant. Answer in valid JSON format only.

AVAILABLE PRODUCTS:
{context}

USER QUESTION: {question}

Analyze the question and respond with ONE of these JSON formats:

1. SINGLE PRODUCT (specific product inquiry):
{{"type": "single", "response": "Product details...", "product": {{"title": "...", "selling_price": "...", "product_rating": X.X, "description": "...", "image": "..."}}}}

2. MULTIPLE PRODUCTS (list/category/search results):
{{"type": "multiple", "response": "Found X products...", "products": [...]}}

3. CHEAPEST:
{{"type": "cheapest", "response": "Cheapest is...", "product": {{...}}}}

4. MOST EXPENSIVE:
{{"type": "most_expensive", "response": "Most expensive is...", "product": {{...}}}}

5. HIGHEST RATING:
{{"type": "highest_rating", "response": "Highest rated is...", "product": {{...}}}}

6. COMPARISON:
{{"type": "compare", "response": "Comparison result...", "product_a": {{...}}, "product_b": {{...}}, "cheaper": {{...}}, "better_rated": {{...}}}}

7. RECOMMENDATION:
{{"type": "recommend", "response": "I recommend...", "products": [...]}}

8. NOT FOUND:
{{"type": "not_found", "response": "No products found matching your criteria."}}

IMPORTANT RULES:
- Return ONLY valid JSON, no extra text
- Use products from AVAILABLE PRODUCTS list only
- For price queries: filter by price correctly
- For rating queries: filter by rating correctly
- For comparisons: identify both products accurately
- Product format: {{"title": "...", "selling_price": "...", "product_rating": X.X, "description": "...", "image": "..."}}
- Limit products list to 5 items maximum

ANSWER (JSON only):
""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 30}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )
        print("LLM loaded and ready")
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        traceback.print_exc()

# ============= API ROUTES =============

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(400, "Empty query")

        logger.info(f"Processing query: {query}")

        # Handle greetings
        if any(word in query.lower().split() for word in GREETING_WORDS):
            return JSONResponse({
                "response": "Hello! I'm your e-commerce assistant. I can help you find products, compare prices, get recommendations, and more. What are you looking for today?"
            })

        # Get all documents
        all_docs = _all_docs_from_vectorstore()
        
        if not all_docs:
            return JSONResponse({
                "response": "Sorry, no products are currently available.",
                "products": []
            })

        # Detect query type
        query_type, params = detect_query_type(query)
        logger.info(f"Query type: {query_type}")

        # ============= HANDLE DIFFERENT QUERY TYPES =============

        # 1. LIST ALL PRODUCTS
        if query_type == "list_all":
            products = [doc_to_card(d) for d in all_docs[:5]]
            return JSONResponse({
                "response": f"Here are all available products. We have {len(all_docs)} products in total.",
                "products": products,
                "total_count": len(all_docs),
                "debug": {"type": "list_all"}
            })

        # 2. SEARCH BY PRODUCT NAME
        elif query_type == "product_by_name":
            # Extract product name from query
            product_query = query
            for prefix in ["do you have", "search for", "show me", "find", "looking for"]:
                if prefix in query.lower():
                    product_query = re.sub(rf"\b{prefix}\b", "", query, flags=re.IGNORECASE).strip()
            
            product_query = product_query.strip('?"\'')
            
            # Try fuzzy matching first
            matched_docs = fuzzy_match_product(product_query, all_docs, threshold=0.3)
            
            if not matched_docs:
                # Use semantic search
                matched_docs = vectorstore.similarity_search(product_query, k=5)
            
            if not matched_docs:
                return JSONResponse({
                    "response": f"Sorry, I couldn't find '{product_query}' in our inventory.",
                    "products": [],
                    "debug": {"type": "product_by_name", "query": product_query}
                })
            
            products = [doc_to_card(d) for d in matched_docs[:5]]
            
            if len(products) == 1:
                p = products[0]
                return JSONResponse({
                    "response": f"Found: {p['title']} - Rs {p['selling_price']} - Rated {p['product_rating']}/5 stars",
                    "products": products,
                    "debug": {"type": "product_by_name"}
                })
            else:
                return JSONResponse({
                    "response": f"Found {len(products)} products matching '{product_query}':",
                    "products": products,
                    "debug": {"type": "product_by_name"}
                })

        # 3. CATEGORY SEARCH
        elif query_type == "category":
            category = extract_category(query)
            
            if category:
                filtered_docs = [
                    d for d in all_docs
                    if category.lower() in get_primary_category(doc_to_metadata(d)).lower()
                ]
            else:
                # Use semantic search
                filtered_docs = vectorstore.similarity_search(query, k=20)
            
            if not filtered_docs:
                return JSONResponse({
                    "response": f"Sorry, no products found in the '{category}' category.",
                    "products": [],
                    "debug": {"type": "category", "category": category}
                })
            
            products = [doc_to_card(d) for d in filtered_docs[:5]]
            return JSONResponse({
                "response": f"Found {len(filtered_docs)} {category or 'matching'} products:",
                "products": products,
                "total_count": len(filtered_docs),
                "debug": {"type": "category", "category": category}
            })

        # 4. CHEAPEST PRODUCT
        elif query_type == "cheapest":
            # Extract context if any (e.g., "cheapest laptop")
            context_words = query.lower().replace("cheapest", "").replace("most affordable", "").strip()
            
            if context_words and len(context_words) > 3:
                # Filter by context
                relevant_docs = vectorstore.similarity_search(context_words, k=20)
            else:
                relevant_docs = all_docs
            
            if not relevant_docs:
                relevant_docs = all_docs
            
            sorted_docs = sorted(relevant_docs, key=lambda d: parse_price(doc_to_metadata(d).get("selling_price")))
            cheapest = sorted_docs[0] if sorted_docs else None
            
            if not cheapest:
                return JSONResponse({"response": "No products available.", "products": []})
            
            card = doc_to_card(cheapest)
            return JSONResponse({
                "response": f"The cheapest {context_words or 'product'} is '{card['title']}' for Rs {card['selling_price']}",
                "products": [card],
                "debug": {"type": "cheapest"}
            })

        # 5. MOST EXPENSIVE PRODUCT
        elif query_type == "most_expensive":
            context_words = query.lower().replace("most expensive", "").replace("costliest", "").strip()
            
            if context_words and len(context_words) > 3:
                relevant_docs = vectorstore.similarity_search(context_words, k=20)
            else:
                relevant_docs = all_docs
            
            sorted_docs = sorted(relevant_docs, key=lambda d: parse_price(doc_to_metadata(d).get("selling_price")), reverse=True)
            most_exp = sorted_docs[0] if sorted_docs else None
            
            if not most_exp:
                return JSONResponse({"response": "No products available.", "products": []})
            
            card = doc_to_card(most_exp)
            return JSONResponse({
                "response": f"The most expensive {context_words or 'product'} is '{card['title']}' for Rs {card['selling_price']}",
                "products": [card],
                "debug": {"type": "most_expensive"}
            })

        # 6. PRICE RANGE
        elif query_type == "price_range":
            min_price, max_price = extract_price_range(query)
            
            # Check if there's a product type mentioned
            query_clean = re.sub(r'(under|below|above|between|price|budget|₹|rs\.?|\d+)', '', query.lower()).strip()
            
            if query_clean and len(query_clean) > 3:
                relevant_docs = vectorstore.similarity_search(query_clean, k=30)
            else:
                relevant_docs = all_docs
            
            # Filter by price
            filtered_docs = []
            for d in relevant_docs:
                price = parse_price(doc_to_metadata(d).get("selling_price"))
                if min_price is not None and max_price is not None:
                    if max_price == float('inf'):
                        if price >= min_price:
                            filtered_docs.append(d)
                    elif min_price <= price <= max_price:
                        filtered_docs.append(d)
                elif max_price is not None:
                    if price <= max_price:
                        filtered_docs.append(d)
                elif min_price is not None:
                    if price >= min_price:
                        filtered_docs.append(d)
            
            if not filtered_docs:
                return JSONResponse({
                    "response": f"Sorry, no products found in that price range.",
                    "products": [],
                    "debug": {"type": "price_range", "min": min_price, "max": max_price}
                })
            
            products = [doc_to_card(d) for d in filtered_docs[:5]]
            
            if max_price == float('inf'):
                price_text = f"above Rs {min_price}"
            elif min_price == 0:
                price_text = f"under Rs {max_price}"
            else:
                price_text = f"between Rs {min_price} and Rs {max_price}"
            
            return JSONResponse({
                "response": f"Found {len(filtered_docs)} products {price_text}:",
                "products": products,
                "total_count": len(filtered_docs),
                "debug": {"type": "price_range", "min": min_price, "max": max_price}
            })

        # 7. HIGHEST RATING
        elif query_type == "highest_rating":
            rating_threshold = extract_rating_threshold(query)
            
            # Extract context
            context_words = re.sub(r'(highest|best|top|rated?|rating|reviews?|stars?|\d+)', '', query.lower()).strip()
            
            if context_words and len(context_words) > 3:
                relevant_docs = vectorstore.similarity_search(context_words, k=30)
            else:
                relevant_docs = all_docs
            
            # Filter by rating if threshold specified
            if rating_threshold:
                filtered_docs = [
                    d for d in relevant_docs
                    if parse_rating(doc_to_metadata(d).get("product_rating")) >= rating_threshold
                ]
            else:
                filtered_docs = relevant_docs
            
            if not filtered_docs:
                return JSONResponse({
                    "response": "No products found matching your criteria.",
                    "products": []
                })
            
            sorted_docs = sorted(filtered_docs, key=lambda d: parse_rating(doc_to_metadata(d).get("product_rating")), reverse=True)
            
            if rating_threshold:
                products = [doc_to_card(d) for d in sorted_docs[:5]]
                return JSONResponse({
                    "response": f"Found {len(products)} products rated {rating_threshold}+ stars:",
                    "products": products,
                    "debug": {"type": "highest_rating", "threshold": rating_threshold}
                })
            else:
                highest = sorted_docs[0]
                card = doc_to_card(highest)
                return JSONResponse({
                    "response": f"The highest rated {context_words or 'product'} is '{card['title']}' with {card['product_rating']}/5 stars",
                    "products": [card],
                    "debug": {"type": "highest_rating"}
                })

        # 8. COMPARISON
        elif query_type == "compare":
            # Extract product names from comparison
            match = None
            for pattern in QUERY_PATTERNS["compare"]:
                match = re.search(pattern, query.lower())
                if match and len(match.groups()) >= 2:
                    break
            
            if not match or len(match.groups()) < 2:
                # Fallback: try to split by common separators
                parts = re.split(r'\s+(?:or|vs\.?|versus|and)\s+', query.lower())
                if len(parts) >= 2:
                    product_a_query = parts[0].strip()
                    product_b_query = parts[1].strip()
                else:
                    return JSONResponse({
                        "response": "Please specify two products to compare (e.g., 'Compare Product A and Product B')",
                        "products": []
                    })
            else:
                product_a_query = match.group(1).strip() if len(match.groups()) >= 1 else ""
                product_b_query = match.group(2).strip() if len(match.groups()) >= 2 else ""
            
            # Clean up product names
            for prefix in ["product", "between", "the"]:
                product_a_query = re.sub(rf"\b{prefix}\b", "", product_a_query, flags=re.IGNORECASE).strip()
                product_b_query = re.sub(rf"\b{prefix}\b", "", product_b_query, flags=re.IGNORECASE).strip()
            
            # Find both products
            products_a = fuzzy_match_product(product_a_query, all_docs, threshold=0.3)
            products_b = fuzzy_match_product(product_b_query, all_docs, threshold=0.3)
            
            if not products_a:
                products_a = vectorstore.similarity_search(product_a_query, k=1)
            if not products_b:
                products_b = vectorstore.similarity_search(product_b_query, k=1)
            
            if not products_a or not products_b:
                return JSONResponse({
                    "response": f"Sorry, I couldn't find both products. Please check the product names.",
                    "products": [],
                    "debug": {"type": "compare", "product_a": product_a_query, "product_b": product_b_query}
                })
            
            card_a = doc_to_card(products_a[0])
            card_b = doc_to_card(products_b[0])
            
            price_a = parse_price(card_a["selling_price"])
            price_b = parse_price(card_b["selling_price"])
            
            cheaper = card_a if price_a < price_b else card_b
            better_rated = card_a if card_a["product_rating"] > card_b["product_rating"] else card_b
            
            comparison_text = f"Comparing '{card_a['title']}' (Rs {card_a['selling_price']}, {card_a['product_rating']}/5) vs '{card_b['title']}' (Rs {card_b['selling_price']}, {card_b['product_rating']}/5). "
            comparison_text += f"Cheaper: {cheaper['title']}. Better rated: {better_rated['title']}."
            
            return JSONResponse({
                "response": comparison_text,
                "comparison": {
                    "product_a": card_a,
                    "product_b": card_b,
                    "cheaper": cheaper,
                    "better_rated": better_rated,
                    "price_difference": abs(price_a - price_b)
                },
                "debug": {"type": "compare"}
            })

        # 9. RECOMMENDATION
        elif query_type == "recommend":
            # Extract context (what to recommend)
            recommend_query = query.lower()
            for prefix in ["recommend", "suggest", "what should i buy", "what should i get", "what should i purchase"]:
                recommend_query = re.sub(rf"\b{prefix}\b", "", recommend_query, flags=re.IGNORECASE).strip()
            
            # Extract price if mentioned
            min_price, max_price = extract_price_range(query)
            
            # Clean query
            recommend_query = re.sub(r'(under|below|above|between|price|budget|₹|rs\.?|\d+)', '', recommend_query).strip()
            
            # Get relevant products
            if recommend_query and len(recommend_query) > 2:
                relevant_docs = vectorstore.similarity_search(recommend_query, k=30)
            else:
                # No specific context, recommend best overall
                relevant_docs = all_docs
            
            # Filter by price if specified
            if min_price is not None or max_price is not None:
                filtered_docs = []
                for d in relevant_docs:
                    price = parse_price(doc_to_metadata(d).get("selling_price"))
                    if max_price == float('inf'):
                        if price >= min_price:
                            filtered_docs.append(d)
                    elif min_price == 0 or min_price is None:
                        if price <= max_price:
                            filtered_docs.append(d)
                    else:
                        if min_price <= price <= max_price:
                            filtered_docs.append(d)
                relevant_docs = filtered_docs
            
            if not relevant_docs:
                return JSONResponse({
                    "response": "Sorry, I couldn't find any products matching your criteria.",
                    "products": []
                })
            
            # Sort by rating and get top 3
            sorted_docs = sorted(relevant_docs, key=lambda d: parse_rating(doc_to_metadata(d).get("product_rating")), reverse=True)
            recommendations = [doc_to_card(d) for d in sorted_docs[:3]]
            
            price_context = ""
            if max_price and max_price != float('inf'):
                price_context = f" under Rs {max_price}"
            elif min_price:
                price_context = f" above Rs {min_price}"
            
            return JSONResponse({
                "response": f"I recommend these{' top-rated' if recommend_query else ''} products{price_context}:",
                "products": recommendations,
                "debug": {"type": "recommend", "context": recommend_query, "price_filter": (min_price, max_price)}
            })

        # 10. SEMANTIC SEARCH (fallback)
        else:
            if not qa_chain:
                # No LLM, use basic semantic search
                relevant_docs = vectorstore.similarity_search(query, k=5)
                
                if not relevant_docs:
                    return JSONResponse({
                        "response": "Sorry, I couldn't find any relevant products.",
                        "products": []
                    })
                
                products = [doc_to_card(d) for d in relevant_docs]
                return JSONResponse({
                    "response": f"Here are {len(products)} products that might interest you:",
                    "products": products,
                    "debug": {"type": "semantic_fallback"}
                })
            
            # Use LLM for complex queries
            try:
                relevant_docs = vectorstore.similarity_search(query, k=30)
                context = build_context(relevant_docs)
                
                filled = PROMPT.format(context=context, question=query)
                raw = llm.invoke(filled)
                llm_output = raw.strip()
                
                logger.info(f"LLM output preview: {llm_output[:300]}")
                
                # Extract JSON
                json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                if not json_match:
                    # Fallback
                    products = [doc_to_card(d) for d in relevant_docs[:3]]
                    return JSONResponse({
                        "response": "Here are some products that might help:",
                        "products": products,
                        "debug": {"type": "llm_fallback", "reason": "no_json"}
                    })
                
                payload = json.loads(json_match.group(0))
                response_type = payload.get("type", "unknown")
                
                # Handle LLM response
                if response_type == "single":
                    return JSONResponse({
                        "response": payload.get("response"),
                        "products": [payload.get("product")] if payload.get("product") else [],
                        "debug": {"type": "llm_single"}
                    })
                
                elif response_type in ("multiple", "recommend"):
                    return JSONResponse({
                        "response": payload.get("response"),
                        "products": payload.get("products", [])[:5],
                        "debug": {"type": f"llm_{response_type}"}
                    })
                
                elif response_type in ("cheapest", "most_expensive", "highest_rating"):
                    return JSONResponse({
                        "response": payload.get("response"),
                        "products": [payload.get("product")] if payload.get("product") else [],
                        "debug": {"type": f"llm_{response_type}"}
                    })
                
                elif response_type == "compare":
                    return JSONResponse({
                        "response": payload.get("response"),
                        "comparison": {
                            "product_a": payload.get("product_a"),
                            "product_b": payload.get("product_b"),
                            "cheaper": payload.get("cheaper"),
                            "better_rated": payload.get("better_rated")
                        },
                        "debug": {"type": "llm_compare"}
                    })
                
                elif response_type == "not_found":
                    return JSONResponse({
                        "response": payload.get("response", "No products found."),
                        "products": [],
                        "debug": {"type": "llm_not_found"}
                    })
                
                else:
                    # Generic response
                    return JSONResponse({
                        "response": payload.get("response", "Here are some relevant products:"),
                        "products": payload.get("products", [])[:5],
                        "debug": {"type": f"llm_{response_type}"}
                    })
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"LLM output: {llm_output[:500]}")
                
                # Fallback to semantic search
                products = [doc_to_card(d) for d in relevant_docs[:3]]
                return JSONResponse({
                    "response": "Here are some products that might help:",
                    "products": products,
                    "debug": {"type": "json_error_fallback"}
                })
            
            except Exception as e:
                logger.error(f"LLM processing error: {e}")
                logger.error(traceback.format_exc())
                
                # Final fallback
                relevant_docs = vectorstore.similarity_search(query, k=3)
                products = [doc_to_card(d) for d in relevant_docs]
                return JSONResponse({
                    "response": "Here are some relevant products:",
                    "products": products,
                    "debug": {"type": "error_fallback"}
                })

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(FRONTEND_DIR, "product2.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>E-commerce Chatbot API Running</h1><p>Use /query endpoint to search products.</p>")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    path = os.path.join(FRONTEND_DIR, "product2.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("UI not found", status_code=404)

@app.get("/images/{filename}")
def get_image(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(IMAGES_DIR, safe)
    return FileResponse(path) if os.path.exists(path) else HTTPException(404, "Image not found")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if vectorstore and qa_chain else "degraded",
        "vectorstore_loaded": bool(vectorstore),
        "llm_loaded": bool(qa_chain),
        "total_products": len(_all_docs_from_vectorstore()) if vectorstore else 0
    }

@app.get("/categories")
def get_categories():
    """Get all available categories"""
    docs = _all_docs_from_vectorstore()
    categories = set()
    for doc in docs:
        cat = get_primary_category(doc_to_metadata(doc))
        if cat:
            categories.add(cat)
    return {"categories": sorted(list(categories))}

if __name__ == "__main__":
    uvicorn.run("without_reg:app", host="0.0.0.0", port=5000, reload=True)