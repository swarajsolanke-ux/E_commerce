
import os
import re
import traceback
from typing import Tuple, Dict, Any, List, Set, Optional
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
import logging
import uvicorn
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
VECTOR_DIR = os.path.join(BASE_DIR, "vect", "vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model_path = "/Users/swarajsolanke/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

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

llm = None
qa_chain = None
if LLM_AVAILABLE:
    try:
        print("Loading language model (optional)...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
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


        prompt_template ="""
        You are an intelligent e-commerce assistant chatbot trained to answer questions in the following domains:

1. Product Database Lookup:
- ONLY answer questions about products in the database.
- ONLY provide information that exists in the context below.
- Answer with ONLY the requested value - no explanations or extra text.
- If the question cannot be answered from the context, respond: "No data found".
- If the question is general (not product-specific), respond: "Sorry — that doesn't look like a product query. Please ask about product price, rating, reviews, or name."

2. Order & Shipping:
- Answer queries about order status, tracking numbers, delivery times, shipment status, shipping costs, shipping countries, shipping options, address changes, confirmation emails, and backordered items.

3. Returns, Refunds & Exchanges:
- Answer queries about return policies, initiating returns/exchanges, refund eligibility and processing times, damaged items, exchanges, return shipping, and return conditions.

4. Product Information:
- Answer queries about product functionality, materials, size guides, stock availability, restocking, product features, color variations, compatibility, recommendations, customer reviews, and product certifications (organic/vegan/etc.).

5. Discounts, Payments & Promotions:
- Answer queries about discount codes, payment methods, payment issues, financing options, and upcoming sales.

6. Account Management:
- Answer queries about password resets, login issues, updating personal info, loyalty points, order history, and account deletion.

7. General & Technical Support:
- Answer queries about speaking to human agents, problem reporting, product search help, mobile website, service hours, customer support contacts, and privacy policies.

If the question is outside these categories or requires information not in the product database context (for product-specific questions), politely respond with the above generic replies or "No data found" as appropriate.

Context from database:
{context}

Question: {question}

Answer:
"""

#         prompt_template = """
#     You are a strict database lookup assistant for an e-commerce product database.

#     STRICT RULES:
#     1. ONLY answer questions about products in the database
#     2. ONLY provide information that exists in the context below
#     3. Answer with ONLY the requested value - no explanations or extra text
#     4. If the question cannot be answered from the context, respond: "No data found"
#     5. if the question are generalize one then simple answer "Sorry — that doesn't look like a product query. Please ask about product price, rating, reviews, or name."

#     Context from database:
#     {context}

#     Question: {question}

#     Answer:
# """
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
    "http://localhost:5500", "http://127.0.0.1:5500",
    "http://localhost:8000", "http://127.0.0.1:8000",
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
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening","hii","helo",
    "morning", "afternoon", "evening"
}

PRODUCT_KEYWORDS = set([
    "product", "item", "buy", "purchase", "show", "tell", "find", "what", "which", "give me", "tell me",
    "how", "how much", "price", "cost", "rating", "review", "images", "image", "selling_price", "shoes",
    "product_rating", "title", "category", "t-shirt", "shoe", "bat", "cricket", "plastic", "pvc", "T-shirt", "sarre", "sarres"
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



def _all_docs_from_vectorstore() -> List[Any]:
    """Return all doc objects in vectorstore.docstore. Returns empty list if unavailable."""
    if vectorstore is None:
        return []
    try:
     
        all_docs = list(getattr(vectorstore.docstore, "_dict", {}).values())
        return all_docs
    except Exception as e:
        print("Error reading docs from vectorstore:", e)
        return []


def parse_between_price_constraints(query: str) -> Dict[str, Optional[float]]:
    """Parse "between X and Y" style price queries."""
    q = str(query).lower()
    m = re.search(r"\bbetween\s+₹?\s*([\d,]+)\s*(?:and|-|to)\s*₹?\s*([\d,]+)\b", q)
    if m:
        try:
            p1 = float(re.sub(r"[^\d.]", "", m.group(1)))
            p2 = float(re.sub(r"[^\d.]", "", m.group(2)))
            return {"min_price": min(p1, p2), "max_price": max(p1, p2)}
        except:
            pass
    # also support "X to Y" e.g. "₹50 to ₹200"
    m2 = re.search(r"\b₹?\s*([\d,]+)\s*(?:to|-)\s*₹?\s*([\d,]+)\b", q)
    if m2:
        try:
            p1 = float(re.sub(r"[^\d.]", "", m2.group(1)))
            p2 = float(re.sub(r"[^\d.]", "", m2.group(2)))
            return {"min_price": min(p1, p2), "max_price": max(p1, p2)}
        except:
            pass
    return {"min_price": None, "max_price": None}


def doc_to_metadata(doc) -> Dict[str, Any]:
    """Normalized metadata extraction for doc objects."""
    if doc is None:
        return {}
    if isinstance(doc, dict):
        return doc.get("metadata", doc)
    m = getattr(doc, "metadata", None)
    if isinstance(m, dict):
        return m
    try:
        return dict(doc.__dict__)
    except Exception:
        return {}


def list_all_products(limit: int = 5) -> List[Dict[str, Any]]:
    """Return a list of product cards for all products in the vectorstore (up to limit)."""
    docs = _all_docs_from_vectorstore()
    products = []
    for doc in docs[:limit]:
        m = doc_to_metadata(doc)
        card = build_product_card(m)
        products.append(card)
    return products


def list_products_in_price_range(min_price: float, max_price: float) -> List[Dict[str, Any]]:
    out = []
    docs = _all_docs_from_vectorstore()
    for doc in docs:
        m = doc_to_metadata(doc)
        p = parse_price(m.get("selling_price") or m.get("cost") or "")
        if p >= min_price and p <= max_price:
            out.append(build_product_card(m))
    return out


def get_cheapest_product() -> Optional[Dict[str, Any]]:
    docs = _all_docs_from_vectorstore()
    best = None
    best_price = None
    for doc in docs:
        m = doc_to_metadata(doc)
        p = parse_price(m.get("selling_price") or m.get("cost") or "")
        if p == 0.0:
            continue
        if best_price is None or p < best_price:
            best_price = p
            best = build_product_card(m)
    return best


def get_highest_rating_product() -> Optional[Dict[str, Any]]:
    docs = _all_docs_from_vectorstore()
    best = None
    best_rating = None
    for doc in docs:
        m = doc_to_metadata(doc)
        r = parse_rating(m.get("product_rating") or m.get("rating") or 0)
        if best_rating is None or r > best_rating:
            best_rating = r
            best = build_product_card(m)
    return best


def find_product_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Find first doc whose normalized title/name matches the provided name (case-insensitive)."""
    n = str(name).strip().lower()
    docs = _all_docs_from_vectorstore()
    for doc in docs:
        m = doc_to_metadata(doc)
        title = str(m.get("title") or m.get("name") or "").strip().lower()
        if title == n:
            return m
    
    for doc in docs:
        m = doc_to_metadata(doc)
        title = str(m.get("title") or m.get("name") or "").strip().lower()
        if not title:
            continue
        if SequenceMatcher(None, n, title).ratio() > 0.85:
            return m
    return None


def compare_two_products(name_a: str, name_b: str) -> Dict[str, Any]:
    """Compare two products by price and return a dict with both and the cheaper one."""
    ma = find_product_by_name(name_a)
    mb = find_product_by_name(name_b)
    if not ma and not mb:
        return {"found_a": False, "found_b": False}
    pa = parse_price(ma.get("selling_price") or ma.get("cost") or "") if ma else None
    pb = parse_price(mb.get("selling_price") or mb.get("cost") or "") if mb else None
    result = {
        "product_a": build_product_card(ma) if ma else None,
        "product_b": build_product_card(mb) if mb else None,
        "cheaper": None
    }
    if pa is not None and pb is not None:
        if pa <= pb:
            result["cheaper"] = build_product_card(ma)
        else:
            result["cheaper"] = build_product_card(mb)
    elif pa is not None:
        result["cheaper"] = build_product_card(ma)
    elif pb is not None:
        result["cheaper"] = build_product_card(mb)
    return result



def get_all_categories() -> List[str]:
    if vectorstore is None:
        return []
    try:
        all_docs = vectorstore.docstore._dict.values()
        categories = set()
        for doc in all_docs:
            meta = getattr(doc, "metadata", {})
            for key in ["category_1", "category_2", "category_3", "product_category"]:
                val = str(meta.get(key, "")).strip()
                if val and val.lower() not in ["nan", "none", ""]:
                    categories.add(val)
        return sorted(categories)
    except Exception as e:
        print("Error fetching categories:", e)
        return []


def is_list_categories_query(query: str) -> bool:
    q = query.lower().strip()
    patterns = [
        r"list.*all.*product.*categor",
        r"what.*categor.*available",
        r"show.*categor",
        r"categor.*list",
        r"all.*categor",
        r"can you.*list the *product availables in the *database",
    
    ]

    return any(re.search(p, q) for p in patterns)


def is_general_query(query: str) -> bool:
    if not query or not isinstance(query, str):
        return True
    q = query.strip()
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
        r"\b(show|find|tell|get|give|provide)\b.*\b(product|item|title|t-shirt|shoe|bat|cricket|shoes|sarres|sarre|T-shirt|)\b",
        r"\b(how much|what is|what's)\b.*\b(price|cost|selling_price)\b",
    ]
    return any(re.search(p, q) for p in patterns)

def normalize_vector_score(score) -> float:
    try:
        if score is None:
            return 0.5
        s = float(score)
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
    print(f"price constraint")
    if m2:
        try:
            min_price = float(re.sub(r"[^\d.]", "", m2.group(1)))
        except:
            pass
  
    between = parse_between_price_constraints(query)
    if between.get("min_price") is not None or between.get("max_price") is not None:
        return {"min_price": between.get("min_price"), "max_price": between.get("max_price")}
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

def get_recommendations(main_meta: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    try:
        if vectorstore is None:
            return []
        cat = get_primary_category(main_meta)
        main_title = (main_meta.get("title") or "").lower()
        print(f"title for recommendations:{main_title}")
        if not cat or not main_title:
            return []

        query_text = f"{main_title} {cat}"
        results = vectorstore.similarity_search_with_score(query_text, k=10)
        logging.info(f"recommendation candidates:{results}")
        recs = []
        seen_titles = {main_title}
        logging.info(f"primary category for recommendations:{cat}")
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, _score = item
                logging.info(f"recommendate candidate docs:{doc}")
            else:
                doc = item
            m = dict(getattr(doc, "metadata", {}) or {})
            print(f"recommendate candidate metadata:{m}")
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
            recs.append(rec)
            print(f"rec:{recs}")
            seen_titles.add(title)
            if len(recs) >= k:
                break
        return recs
    
    except Exception as e:
        print("[REC ERROR]", e)
        traceback.print_exc()
        return []

@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        print(f"user query given: {query}")
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        q_lower = query.lower()

        
        list_all_patterns = [
            r"\b(provide me the list of product availables)\b",
            r"\b(can you list all the products available)\b",
            r"\b(list all products)\b",
            r"\b(show me all products)\b",
            r"\b(show all products)\b",
            r"\b(list all the products available)\b",
            r"\b(all products available)\b"
        ]
        if any(re.search(p, q_lower) for p in list_all_patterns):
            products = list_all_products(limit=5)
            print(f"product nameS:{products[0]}")
            if not products:
                return JSONResponse({
                    "response": "No products found in the database.",
                    "products": [],
                    "debug": {"type": "list_all_products", "count": 0}
                })
            return JSONResponse({
                "response": f"Found {len(products)} products.",
                "products": products,
                "debug": {"type": "list_all_products", "count": len(products)}
            })

        
        if is_list_categories_query(query):
            categories = get_all_categories()
            print(len(categories))
            print(f"available categories:{categories}")
            if categories:
                return JSONResponse({
                    "response": f"Available product categories:\n" + "\n".join([f"• {c}" for c in categories]),
                    "categories": categories,
                    "debug": {"type": "list_categories"}
                })
            else:
                return JSONResponse({
                    "response": "No categories found in the database.",
                    "categories": [],
                    "debug": {"type": "list_categories", "error": "empty"}
                })

        q_tokens = set(re.findall(r"\b\w+\b", q_lower))
        if re.search(r"\b(what kinds of products|kinds of products do you have|what products do you have)\b", q_lower):
            categories = get_all_categories()
            sample_cats = categories[:30]
            logging.info(f"sample categories:{sample_cats}")
            return JSONResponse({
                "response": f"We have products across categories: {', '.join(sample_cats)}" if sample_cats else "No categories found.",
                "categories": categories,
                "debug": {"type": "kinds_of_products"}
            })
        if re.search(r"\b(what are the main categories|main categories available|main categories)\b", q_lower):
            categories = get_all_categories()
            return JSONResponse({
                "response": f"Main categories:\n" + "\n".join([f"• {c}" for c in categories]) if categories else "No categories found.",
                "categories": categories,
                "debug": {"type": "main_categories"}
            })

        if q_tokens & GREETING_WORDS:
            return JSONResponse({
                "response": "Hello and welcome to the E-commerce world! I'm your shopping assistant—let me know what product you're looking for, and I'll help you find it. How can I assist you today?"
            })

        if is_general_query(query):
            return JSONResponse({
                "response": "Sorry — I can only answer queries related to e-commerce products. Please ask about product price, rating, reviews, or name.",
                "debug": {"relevance": False, "reason": "General / conversational query detected"}
            })

        if not is_product_query(query):
            return JSONResponse({
                "response": "Sorry — that doesn't look like a product query. Please ask about product price, rating, reviews, or name.",
                "debug": {"relevance": False, "reason": "Not a product query"}
            })

        if vectorstore is None:
            return JSONResponse({
                "response": "Server vectorstore unavailable.",
                "debug": {"relevance": False, "reason": "No vectorstore loaded"}
            })

     
        m_comp = re.search(r"\bcompare\s+(.+?)\s+(?:vs|vs\.|and|with|v)\s+(.+?)(?:\?|$)", q_lower)
        if m_comp:
            name_a = m_comp.group(1).strip()
            name_b = m_comp.group(2).strip()
            comp_result = compare_two_products(name_a, name_b)
            if not comp_result.get("found_a", True) and not comp_result.get("found_b", True):
                return JSONResponse({
                    "response": f"Couldn't find either '{name_a}' or '{name_b}' in the database.",
                    "debug": {"type": "compare", "found_a": False, "found_b": False}
                })
            cheaper = comp_result.get("cheaper")
            return JSONResponse({
                "response": f"Comparison between '{name_a}' and '{name_b}'. Cheaper product: {cheaper['title'] if cheaper else 'Not available'}.",
                "comparison": comp_result,
                "debug": {"type": "compare"}
            })

       
        if re.search(r"\b(highest rating|highest rated|top rated|product_name having highest rating|which product has the highest rating)\b", q_lower):
            best = get_highest_rating_product()
            if not best:
                return JSONResponse({
                    "response": "No rated products found in the database.",
                    "debug": {"type": "highest_rating", "found": False}
                })
            return JSONResponse({
                "response": f"Product with highest rating: {best['title']}",
                "product": best,
                "debug": {"type": "highest_rating", "rating": best.get("product_rating")}
            })

      
        if re.search(r"\b(cheapest product|price of the cheapest product|what is the price of the cheapest product)\b", q_lower):
            cheapest = get_cheapest_product()
            if not cheapest:
                return JSONResponse({
                    "response": "No priced products found in the database.",
                    "debug": {"type": "cheapest", "found": False}
                })
            price = parse_price(cheapest["selling_price"] or cheapest["metadata"].get("cost"))
            return JSONResponse({
                "response": f"The cheapest product is '{cheapest['title']}' priced at Rs{price:.2f}.",
                "product": cheapest,
                "debug": {"type": "cheapest", "price": price}
            })


        between = parse_between_price_constraints(query)
        if between.get("min_price") is not None or between.get("max_price") is not None:
            min_p = between.get("min_price") or 0.0
            max_p = between.get("max_price") or 1e9
            list_in_range = list_products_in_price_range(min_p, max_p)
            print(f"top 5 product")
            return JSONResponse({
                "response": f"Found {len(list_in_range)} products between Rs{min_p:.2f} and Rs{max_p:.2f}.",
                "products": list_in_range[:5],
                "debug": {"type": "price_range", "min": min_p, "max": max_p, "count": len(list_in_range)}
            })

        # Also support "Can I see all products within a price range, e.g., ₹50 to ₹200?" generic phrasing
        if re.search(r"\b(products within a price range|products between|within.*price range|products from)\b", q_lower):
           
            between2 = parse_between_price_constraints(query)
            if between2.get("min_price") is not None or between2.get("max_price") is not None:
                min_p = between2.get("min_price") or 0.0
                max_p = between2.get("max_price") or 1e9
                list_in_range = list_products_in_price_range(min_p, max_p)
                return JSONResponse({
                    "response": f"Found {len(list_in_range)} products between Rs{min_p:.2f} and Rs{max_p:.2f}.",
                    "products": list_in_range[:3],
                    "debug": {"type": "price_range", "min": min_p, "max": max_p, "count": len(list_in_range)}
                })

       
        price_constraints = parse_price_constraints(query)
        max_price = price_constraints.get("max_price")
        min_price = price_constraints.get("min_price")

        results = vectorstore.similarity_search_with_score(query, k=K_SEMANTIC_CANDIDATES)
        candidates = []
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
                score = None
            metadata = dict(getattr(doc, "metadata", {}) or {})
            candidates.append((metadata, score))

        scored = []
        logging.info(f"candidates before scoring:{scored}")
        for meta, vscore in candidates:
            price_ok = True
            p = parse_price(meta.get("selling_price") or meta.get("cost") or "")
            if max_price is not None and (p == 0 or p > max_price):
                price_ok = False
            if min_price is not None and p < min_price:
                price_ok = False
            sc = score_candidate(query, meta, vscore)
            if (max_price is not None or min_price is not None) and not price_ok:
                sc *= 0.25
            scored.append((sc, meta, vscore))
        logging.info(f"scrored candidates before sort :{scored}")

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or scored[0][0] < 0.35:
            fallback_hits = []
            extra = vectorstore.similarity_search_with_score(query, k=50)
            print(f"extra candidate for fallback:{extra}")
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
            if sc < 0.3:  # Strict threshold
                continue
            card = build_product_card(meta)
            card["score"] = round(float(sc), 4)
            top_products.append(card)
            print(f"top_products category:{top_products}")

        if not top_products:
            return JSONResponse({
                "response": "No matching products found.",
                "recommendations": [],
                "debug": {"relevance": False, "reason": "No candidates above threshold"}
            })

        best_meta = dict(top_products[0]["metadata"] or {})
        recs = get_recommendations(best_meta, k=3)

        specific_keywords = ["price", "cost", "how much", "rating", "review", "description"]
        if any(k in q_lower for k in specific_keywords) and qa_chain and top_products:
            context_docs = vectorstore.similarity_search(query, k=1)
            if context_docs:
                context = context_docs[0].page_content
                try:
                    result = qa_chain.invoke({"query": query, "context": context})
                    llm_answer = result["result"].strip()
                    if llm_answer and llm_answer not in ["No data found", ""]:
                        return JSONResponse({
                            "response": llm_answer,
                            "products": top_products,
                            "recommendations": recs,
                            "debug": {"llm_used": True, "top_score": top_products[0].get("score")}
                        })
                except Exception as e:
                    print("LLM failed:", e)

        return JSONResponse({
            "products": top_products,
            "recommendations": recs,
            "debug": {"relevance": True, "top_score": top_products[0].get("score")}
        })

    except Exception as e:
        print("Search error:", e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


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

@app.get("/images/{filename}")
def get_image(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(IMAGES_DIR, safe_name)
    logging.info(f"fetching image path:{path}")
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
    uvicorn.run("product2:app", host="0.0.0.0", port=5000, reload=True)
