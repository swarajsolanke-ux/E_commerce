from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
from typing import Tuple, Dict, Any, List, Set

app = FastAPI(title="E-commerce Chatbot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR resolved to: {BASE_DIR}")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
print(f"frontend dir:{FRONTEND_DIR}")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
print(f"images dir:{IMAGES_DIR}")
DB_PATH = os.path.join(BASE_DIR, "E_commerce", "db", "products.db")
VECTOR_PATH = os.path.join(BASE_DIR, "E_commerce", "vector_store", "index.faiss")

print(f"BASE_DIR: {BASE_DIR}")
print(f"FRONTEND_DIR: {FRONTEND_DIR}")
print(f"STATIC_DIR: {STATIC_DIR}")
print(f"IMAGES_DIR: {IMAGES_DIR}")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("Static files loaded")
else:
    print(f"STATIC_DIR not found: {STATIC_DIR}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
print("Vector store loaded")

print("Loading language model...")
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
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

print(f"llm loaded sucessfully")

prompt_template = """
You are a strict database lookup assistant for an e-commerce product database.

STRICT RULES:
1. ONLY answer questions about products in the database
2. ONLY provide information that exists in the context below
3. Answer with ONLY the requested value - no explanations or extra text
4. If the question cannot be answered from the context, respond: "No data found"

Context from database:
{context}

Question: {question}

IMPORTANT: Answer with ONLY the specific value requested (e.g., just the number for ratings/prices, just the text for reviews).

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
print(f"qa_chain:{qa_chain}")

engine = create_engine(f'sqlite:///{DB_PATH}')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://192.168.5.224:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS middleware configured")

GREETING_WORDS: Set[str] = {
    "hello", "hi", "hey", "greetings","HELLO","HI"
    "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening"
}

class QueryRequest(BaseModel):
    query: str

VALID_PRODUCT_FIELDS = {
    'product_id', 'name', 'category', 'cost', 'price', 'rating', 
    'review', 'reviews', 'image', 'image_path'
}

VALID_CATEGORIES = {
    'clothing', 'sports', 'electronics', 'home', 'garden', 'books'
}

PRODUCT_KEYWORDS = VALID_PRODUCT_FIELDS.union(VALID_CATEGORIES).union({
    'product', 'item', 'buy', 'purchase', 'show', 'tell', 'find',
    'what', 'which', 'how much', 'expensive', 'cheap',"give me "
})

def is_product_query(query: str) -> bool:
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    has_keyword = bool(PRODUCT_KEYWORDS.intersection(query_tokens))
    product_patterns = [
        r'\b(price|cost|rating|review)\s+(of|for)',
        r'\b(show|find|tell|get|give)\s+.*\b(product|item)',
        r'\b(how much|what is)\s+.*\b(price|cost)',
        r'\b(what|which)\s+.*\b(category|product)',
    ]
    has_pattern = any(re.search(pattern, query_lower) for pattern in product_patterns)
    return has_keyword or has_pattern


def calculate_relevance_score(query: str, metadata: Dict[str, Any]) -> float:
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    score = 0.0
    max_score = 0.0
    max_score += 2.0
    category = str(metadata.get('category', '')).lower()
    if category and category in query_lower:
        score += 2.0
    max_score += 3.0
    name = str(metadata.get('name', '')).lower()
    name_tokens = set(re.findall(r'\b\w+\b', name))
    common_tokens = query_tokens.intersection(name_tokens)
    if len(common_tokens) > 0:
        score += 3.0 * (len(common_tokens) / max(len(name_tokens), 1))
    max_score += 1.0
    field_keywords = {'price', 'cost', 'rating', 'review', 'reviews'}
    if query_tokens.intersection(field_keywords):
        score += 1.0
        print(f"score:{score}")
    return score / max_score if max_score > 0 else 0.0



def is_relevant_result(query: str, doc_metadata: Dict[str, Any], vector_score: float = None) -> Tuple[bool, str]:
    if not is_product_query(query):
        return False, "Query is not product-related"
    relevance = calculate_relevance_score(query, doc_metadata)
    if vector_score is not None:
        try:
            score_val = float(vector_score)
            if score_val > 0.95:
                return False, f"Vector distance too high: {score_val:.3f}"
            if score_val < 0.4 and relevance > 0.3:
                return True, f"Strong match - vector: {score_val:.3f}, relevance: {relevance:.2f}"
        except:
            pass
    if relevance > 0.4:
        return True, f"Relevance score: {relevance:.2f}"
    return False, f"Low relevance: {relevance:.2f}"



def extract_answer_from_query(query: str, product: Dict[str, Any]) -> str:
    query_lower = query.lower()
    if any(word in query_lower for word in ['rating', 'rated', 'star', 'stars']):
        rating = product.get('rating', 0)
        return f"{float(rating):.1f}" if rating else "No rating available"
    if any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap', 'how much']):
        cost = product.get('cost', 0)
        return f"Rs{float(cost):.2f}" if cost else "Price not available"
    if any(word in query_lower for word in ['review', 'reviews', 'feedback', 'opinion']):
        review = product.get('review', '')
        return review if review else "No reviews available"
    if any(word in query_lower for word in ['name', 'called', 'title']):
        return product.get('name', 'Unknown product')
    if any(word in query_lower for word in ['category', 'type', 'kind']):
        return product.get('category', 'Unknown category')
    return product.get('name', 'Product information available')


def safe_convert(value, type_func, default):
            try:
                return type_func(value) if value is not None else default
            except:
                return default

def get_recommendations(main: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """
    Return up to *k* products from the **same category** (excluding the main product)
    using vector similarity.
    """
    try:
        cat = main.get("category", "").lower()
        print(f"category to be find:{cat}")
        main_name = main.get("name", "").lower()
        if not cat or not main_name:
            return []

       
        results = vectorstore.similarity_search_with_score(
            f"{main_name} {cat}", k=10
        )

        recs = []
        seen = {main_name}
        for doc, _ in results:
            m = dict(getattr(doc, "metadata", {}) or {})
            name = str(m.get("name", "")).lower()
            prod_cat = str(m.get("category", "")).lower()

            if name == main_name or name in seen or prod_cat != cat:
                continue

            rec = {
                "name": str(m.get("name", "Unknown")),
                "cost": safe_convert(m.get("cost"), float, 0),
                "rating": safe_convert(m.get("rating"), float, 0),
                "review": str(m.get("review", "No review")),
                "category": str(m.get("category", "Unknown")),
                "image": str(m.get("image_path")) if m.get("image_path") else None
            }
            recs.append(rec)
            seen.add(name)
            if len(recs) >= k:
                break
        return recs
    except Exception as e:
        print(f"[REC] error: {e}")
        return []


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(FRONTEND_DIR, "result1.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>E-commerce Chatbot</h1><p>Backend is running</p>")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    index_path = os.path.join(FRONTEND_DIR, "result1.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h3>UI not found</h3>", status_code=404)

@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))
        if words & GREETING_WORDS:
            print(f"GREETING DETECTED → {query!r}")
            return "welcome to ecommerce chatbot , how may i help you?"

        if not is_product_query(query):
            print("Query is not product-related")
            return JSONResponse({
                "response": "sorry i couldn't provide you answer.Please ask relevant query only",
                "products": [],
                "debug": {"relevance": False, "reason": "Not a product query"}
            })

        print("Query is product-related")
        results_with_scores = vectorstore.similarity_search_with_score(query, k=1)
        if not results_with_scores:
            print(" No results from vector store")
            return JSONResponse({
                "response": "Sorry, I couldn't find any matching products in our database.",
                "products": [],
                "debug": {"relevance": False, "reason": "No vector store results"}
            })

        top_doc, vector_score = results_with_scores[0]
        metadata = dict(getattr(top_doc, "metadata", {}) or {})
        print(f"Vector score: {vector_score:.4f}")
        print(f"Metadata: {metadata}")

        is_relevant, reason = is_relevant_result(query, metadata, vector_score)
        if not is_relevant:
            print(f" Result not relevant: {reason}")
            return JSONResponse({
                "response": "Sorry, I couldn't find a relevant product for your query. Please try asking about specific product details like price, rating, or reviews.",
                "products": [],
                "debug": {"relevance": False, "reason": reason, "vector_score": float(vector_score)}
            })

        print(f"Result is relevant: {reason}")

        

        product = {
            "name": str(metadata.get("name", "Unknown")),
            "cost": safe_convert(metadata.get("cost"), float, 0),
            "rating": safe_convert(metadata.get("rating"), float, 0),
            "review": str(metadata.get("review", "No review")),
            "category": str(metadata.get("category", "Unknown")),
            "image": str(metadata.get("image_path")) if metadata.get("image_path") else None
        }

        answer = extract_answer_from_query(query, product)
        print(f"Answer: {answer}")
        print(f"Product: {product['name']}")

     
        recommendations = get_recommendations(product, k=3)

        return JSONResponse({
            "response": answer,
            "products": [product],
            "recommendations": recommendations,
            "debug": {
                "relevance": True,
                "reason": reason,
                "vector_score": float(vector_score)
            }
        })

    except Exception as e:
        print(f"\nError processing query:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/images/{filename}")
def get_image(filename: str):
    safe_name = os.path.basename(filename)
    print(f"safE_name:{safe_name}")
    path = os.path.join(IMAGES_DIR, safe_name)
    print(f"path of an image:{path}")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/health")
def health_check():
    return JSONResponse({
        "status": "healthy",
        "device": device,
        "vector_store_loaded": vectorstore is not None,
        "model_loaded": llm is not None
    })

if __name__ == "__main__":
    uvicorn.run("result1:app", host="192.168.5.155", port=5000, reload=True)
    #192.168.5.155