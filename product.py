
import os
import re
import sys
import traceback
from typing import Tuple, Dict, Any, List, Set
from difflib import SequenceMatcher
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_classic.chains import RetrievalQA
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")

DB_PATH = os.path.join(BASE_DIR, "db", "products_DB.db")
VECTOR_DIR = os.path.join(BASE_DIR, "vect","vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_ID = "gpt2"
print(f"model loaded:{MODEL_ID}")

K_SEMANTIC_CANDIDATES = 10
TOP_RETURN = 1

app = FastAPI(title="E-commerce Chatbot")

print(f"BASE_DIR: {BASE_DIR}")
print(f"FRONTEND_DIR: {FRONTEND_DIR}")
print(f"STATIC_DIR: {STATIC_DIR}")
print(f"IMAGES_DIR: {IMAGES_DIR}")
print(f"DB_PATH: {DB_PATH}")
print(f"VECTOR_DIR: {VECTOR_DIR}")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("Static files loaded")
else:
    print(f"STATIC_DIR not found: {STATIC_DIR}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = None
try:
    print("Loading FAISS vectorstore from directory:", VECTOR_DIR)
    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded")
except Exception as e:
    print("Failed to load vectorstore:", e)
    traceback.print_exc()
print(f"embdeddings:{embeddings}")
print(f"vecorstore:{vectorstore}")

llm = None
qa_chain = None
if LLM_AVAILABLE:
    try:
        print("Loading language model (optional)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
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
        llm = HuggingFacePipeline(pipeline=pipe)
     

        prompt_template = """
You are a strict database lookup assistant for an e-commerce product database.

STRICT RULES:
1. ONLY answer questions about products in the database
2. ONLY provide information that exists in the context below
3. Answer with ONLY the requested value - no explanations or extra text
4. If the question cannot be answered from the context, respond: "No data found"
5. if the question are generalize one then simple answer "Sorry — that doesn't look like a product query. Please ask about product price, rating, reviews, or name."

Context from database:
{context}

Question: {question}

IMPORTANT: Answer with ONLY the specific value requested (e.g., just the number for ratings/prices, just the text for reviews).

Answer:
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        if llm is not None and vectorstore is not None:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        print("Optional LLM loaded")
    except Exception as e:
        print("Failed to load optional LLM:", e)
        traceback.print_exc()
        llm = None
        qa_chain = None

engine = create_engine(f"sqlite:///{DB_PATH}")
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
print("CORS middleware configured")

class QueryRequest(BaseModel):
    query: str

# Greeting words (lowercased)
GREETING_WORDS: Set[str] = {
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening"
}


PRODUCT_KEYWORDS = set([
    "product", "item", "buy", "purchase", "show", "tell", "find", "what", "which","give me","tell me"
    "how", "how much", "price", "cost", "rating", "review", "images", "image", "selling_price","shoes"
    "product_rating", "title", "category", "t-shirt", "shoe", "bat", "cricket", "plastic", "pvc","T-shirt","sarre","sarres"
])


GENERAL_INTENT_PATTERNS = [
    r"^\s*how\s+are\s+you\b",
    r"\bwhat\s+is\s+my\s+name\b",
    r"\bwho\s+am\s+i\b",
    r"\bmeaning\s+of\b",
    r"\bdefine\b",
    r"\btranslate\b",
    r"\bweather\b",
    r"\bnews\b",
    r"\btime\b",
    r"\bdate\b",
    r"\btell\s+me\s+about\s+yourself\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\bopenai\b",
    r"\bi\s+feel\b",
    r"\bmy\s+name\b",
    r"\bmeaning of\b",
    r"\bwhat does\b.*\bmean\b",
    r"\bwho\s+is\b.*",
    r"^(hi|hello|hey)\b\s*$",
]

GENERAL_INTENT_REGEXES = [re.compile(p, re.IGNORECASE) for p in GENERAL_INTENT_PATTERNS]

def is_general_query(query: str) -> bool:
    """
    Detects conversational/general queries that should NOT be answered from the product DB.
    Returns True for chit-chat, personal questions, definitions, weather/news/time, etc.
    """
    if not query or not isinstance(query, str):
        return True
    q = query.strip()
    print(f"general query check for:{q}")
    if q == "":
        return True


    for rx in GENERAL_INTENT_REGEXES:
        if rx.search(q):
            return True

  
    pronoun_patterns = re.compile(r"\b(i|my|me|mine|you)\b", re.IGNORECASE)
    if pronoun_patterns.search(q):
        tokens = set(re.findall(r"\b\w+\b", q.lower()))
        if not (tokens & PRODUCT_KEYWORDS):
            return True


    tokens = re.findall(r"\b\w+\b", q)
    if 0 < len(tokens) <= 3:
        tokset = set(t.lower() for t in tokens)
        if not (tokset & PRODUCT_KEYWORDS):
            return True

    return False

def parse_price(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        s = str(value)
        s_clean = re.sub(r"[^\d.]", "", s)
        if s_clean == "":
            return 0.0
        return float(s_clean)
    except Exception:
        return 0.0

def parse_rating(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(str(value).strip())
    except Exception:
        return 0.0

def is_product_query(query: str) -> bool:
    q = str(query).lower()
    tokens = set(re.findall(r"\b\w+\b", q))
    if tokens & GREETING_WORDS:
        return False
    if tokens & PRODUCT_KEYWORDS:
        return True
    patterns = [
        r"\b(price|cost|rating|review)\b",
        r"\b(show|find|tell|get|give)\b.*\b(product|item|title|t-shirt|shoe|bat|cricket|shoes|sarres|sarre|T-shirt|)\b",
        r"\b(how much|what is|what's)\b.*\b(price|cost|selling_price)\b",
    ]
    return any(re.search(p, q) for p in patterns)

def normalize_vector_score(score) -> float:
    try:
        if score is None:
            return 0.5
        s = float(score)
        print(f"vector score:{s}")
        if 0.0 <= s <= 1.0:
            return s
        return 1.0 / (1.0 + abs(s))
    except Exception:
        return 0.5

def fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

PRICE_RE = re.compile(r"(?:under|below|less than|<|<=)\s*₹?\s*([\d,]+)|(?:above|over|greater than|>|>=)\s*₹?\s*([\d,]+)", re.IGNORECASE)
def parse_price_constraints(query: str) -> Dict[str, float]:
    q = str(query).lower()
    max_price = None
    min_price = None
    m = re.search(r"(?:under|below|less than|<|<=)\s*₹?\s*([\d,]+)", q)
    if m:
        try:
            max_price = float(re.sub(r"[^\d.]", "", m.group(1)))
        except:
            pass
    m2 = re.search(r"(?:above|over|greater than|>|>=)\s*₹?\s*([\d,]+)", q)
    if m2:
        try:
            min_price = float(re.sub(r"[^\d.]", "", m2.group(1)))
        except:
            pass
    return {"max_price": max_price, "min_price": min_price}

def score_candidate(query: str, metadata: Dict[str, Any], vector_score) -> float:
    sim = normalize_vector_score(vector_score)
    title = str(metadata.get("title", "") or "")
    desc = str(metadata.get("description", "") or "")
    fuzzy = max(fuzzy_ratio(query, title), fuzzy_ratio(query, desc))
    q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
    title_tokens = set(re.findall(r"\b\w+\b", title.lower()))
    cat_tokens = set()
    for k in ("category_1","category_2","category_3"):
        cat_tokens |= set(re.findall(r"\b\w+\b", str(metadata.get(k,"")).lower()))
    overlap = 0.0
    if title_tokens:
        overlap += len(q_tokens & title_tokens) / max(len(title_tokens), 1)
    if cat_tokens:
        overlap += len(q_tokens & cat_tokens) / max(len(cat_tokens), 1)
    overlap = min(overlap, 1.0)
    final = 0.5*sim + 0.3*fuzzy + 0.2*overlap
    return max(0.0, min(final, 1.0))

def build_product_card(metadata: Dict[str, Any]) -> Dict[str, Any]:
    image = None
    if metadata.get("image_path"):
        img = str(metadata.get("image_path") or "")
        base = os.path.basename(img)
        image = f"/images/{base}"
    elif metadata.get("image_urls"):
        urls = metadata.get("image_urls")
        if isinstance(urls, (list, tuple)) and len(urls) > 0:
            image = urls[0]
        elif isinstance(urls, str) and urls:
            image = urls
    return {
        "title": metadata.get("title") or metadata.get("name") or "Unknown",
        "selling_price": metadata.get("selling_price") or metadata.get("cost") or "",
        "product_rating": parse_rating(metadata.get("product_rating") or metadata.get("rating")),
        "description": (metadata.get("description") or "")[:280],
        "image": image,
        "metadata": metadata
    }

def get_primary_category(metadata: Dict[str, Any]) -> str:
    for k in ("category_2", "category_1", "category_3"):
        v = str(metadata.get(k, "")).strip()
        if v:
            return v.lower()
    return ""

def safe_convert(value, type_func, default):
    try:
        return type_func(value) if value is not None and value != "" else default
    except:
        return default

def get_recommendations(main_meta: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    try:
        if vectorstore is None:
            return []
        cat = get_primary_category(main_meta)
        main_title = (main_meta.get("title") or "").lower()
        if not cat or not main_title:
            return []

        query_text = f"{main_title} {cat}"
        results = vectorstore.similarity_search_with_score(query_text, k=10)
        print(f"recommendation results:{results}")
        recs = []
        seen_titles = {main_title}
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, _score = item
            else:
                doc = item
            m = dict(getattr(doc, "metadata", {}) or {})
            title = str(m.get("title", "")).lower()
            prod_cat = get_primary_category(m)
            if not title or title in seen_titles or prod_cat != cat:
                continue
            rec = {
                "title": m.get("title", "Unknown"),
                "selling_price": f"Rs{parse_price(m.get('selling_price')):.2f}" if m.get("selling_price") else None,
                "product_rating": parse_rating(m.get("product_rating")),
                "description": m.get("description", ""),
                "category": prod_cat,
                "image": m.get("image_path") or (m.get("image_urls")[0] if m.get("image_urls") else None)
            }
            print(f"recommendation candidate:{rec}")
            recs.append(rec)
            seen_titles.add(title)
            if len(recs) >= k:
                break
        return recs
    except Exception as e:
        print("[REC ERROR]", e)
        traceback.print_exc()
        return []


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(FRONTEND_DIR, "product.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>E-commerce Chatbot</h1><p>Backend is running</p>")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    index_path = os.path.join(FRONTEND_DIR, "product.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h3>UI not found</h3>", status_code=404)

@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        print(f"user query given :{query}")
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        q_lower = query.lower()
        print(f"q_lower:{q_lower}")
        q_tokens = set(re.findall(r"\b\w+\b", q_lower))

        if q_tokens & GREETING_WORDS:
            return JSONResponse({"response": "Hello and welcome to the E-commerce world! I'm your shopping assistant—let me know what product you're looking for, and I'll help you find it. How can I assist you today?"})

  
        if is_general_query(query):
            return JSONResponse({
                "response": "Sorry — I can only answer query related to the E-commerce.kindly asked the question related to this?",
               # "products": [],
                "debug": {"relevance": False, "reason": "General / conversational query detected"}
            })

        # product intent detection
        if not is_product_query(query):
            return JSONResponse({
                "response": "Sorry — that doesn't look like a product query. Please ask about product price, rating, reviews, or name.",
                #"products": [],
                "debug": {"relevance": False, "reason": "Not a product query"}
            })

        if vectorstore is None:
            return JSONResponse({
                "response": "Server vectorstore unavailable.",
               # "products": [],
                "debug": {"relevance": False, "reason": "No vectorstore loaded"}
            })

        
        price_constraints = parse_price_constraints(query)
        print(f"price constraints:{price_constraints}")
        max_price = price_constraints.get("max_price")
        print(f"max_price :{max_price}")
        min_price = price_constraints.get("min_price")
        print(f"minimum price:{min_price}")

        # semantic search candidates
        results = vectorstore.similarity_search_with_score(query, k=K_SEMANTIC_CANDIDATES)
        print(f"results of vector:{results}")
        candidates = []
        print(f"candidates before loop:{candidates}")
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
                score = None
            metadata = dict(getattr(doc, "metadata", {}) or {})
            candidates.append((metadata, score))
            print(f"candidates in loop:{candidates}")

        scored = []
        for meta, vscore in candidates:
            price_ok = True
            if max_price is not None:
                p = parse_price(meta.get("selling_price") or meta.get("cost") or "")
                if p == 0 or p > max_price:
                    price_ok = False
            if min_price is not None:
                p = parse_price(meta.get("selling_price") or meta.get("cost") or "")
                if p < min_price:
                    price_ok = False
            sc = score_candidate(query, meta, vscore)
            if (max_price is not None or min_price is not None) and not price_ok:
                sc *= 0.25
            scored.append((sc, meta, vscore))

        scored.sort(key=lambda x: x[0], reverse=True)

        
        if not scored or scored[0][0] < 0.35:
            fallback_hits = []
            extra = vectorstore.similarity_search_with_score(query, k=50)
            for item in extra:
                if isinstance(item, tuple) and len(item) == 2:
                    doc, vscore = item
                else:
                    doc = item
                    vscore = None
                meta = dict(getattr(doc, "metadata", {}) or {})
                title = str(meta.get("title","")).lower()
                desc = str(meta.get("description","")).lower()
                cats = " ".join([str(meta.get(k,"")).lower() for k in ("category_1","category_2","category_3")])
                qtok = set(re.findall(r"\b\w+\b", query.lower()))
                if qtok & set(re.findall(r"\b\w+\b", title)) or qtok & set(re.findall(r"\b\w+\b", desc)) or qtok & set(re.findall(r"\b\w+\b", cats)):
                    fallback_hits.append((score_candidate(query, meta, vscore) + 0.1, meta, vscore))
            print(f"fallbackes count:{len(fallback_hits)}")
            print(f"fallbackes hits:{fallback_hits}")
            if fallback_hits:
                combined = fallback_hits + scored
                seen = set()
                dedup = []
                for sc, m, vs in combined:
                    t = str(m.get("title","")).lower()
                    if t in seen:
                        continue
                    seen.add(t)
                    dedup.append((sc, m, vs))
                scored = dedup

        
        top_products = []
        for sc, meta, vs in scored[:TOP_RETURN]:
            card = build_product_card(meta)
            card["score"] = round(float(sc), 4)
            top_products.append(card)
            print(f"top_products:{top_products}")

        if not top_products:
            return JSONResponse({
                "response": "No matching products found.",
                #"products": [],
                "recommendations": [],
                "debug": {"relevance": False, "reason": "No candidates returned from vectorstore/fallback"}
            })

        best_meta = dict(top_products[0]["metadata"] or {})
        recs = get_recommendations(best_meta, k=3)
        print(f"recommendations:{recs}")

   
        if any(w in q_lower for w in ["price", "cost", "how much", "selling_price","give me","tell me"]) and len(top_products) > 0:
            best = top_products[0]
            price_str = best.get("selling_price") or best["metadata"].get("selling_price") or ""
            pval = parse_price(price_str)
            answer = f"Rs{pval:.2f}" if pval > 0 else "Price not available"
            print(f"answer price:{answer}")
            return JSONResponse({
                "response": answer,
                "products": [best],
                "recommendations": recs,
                "debug": {"relevance": True, "top_score": top_products[0].get("score")}
            })

        return JSONResponse({
            "response": "Here are the best matches for your query.",
            "products": top_products,
            "recommendations": recs,
            "debug": {"relevance": True, "top_score": top_products[0].get("score")}
        })

    except Exception as e:
        print("Search error:", e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/images/{filename}")
def get_image(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(IMAGES_DIR, safe_name)
    print(f"path of an image requested: {path}")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/health")
def health_check():
    return JSONResponse({
        "status": "healthy" if vectorstore is not None else "degraded",
        "device": device,
        "vector_store_loaded": vectorstore is not None,
        "model_loaded": llm is not None
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("product:app", host="0.0.0.0", port=5000, reload=True)
