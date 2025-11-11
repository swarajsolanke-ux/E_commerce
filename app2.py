
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import hashlib
import requests
import re
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import uvicorn

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"base dir:{BASE_DIR}")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images") 
print(f"images dir :{IMAGES_DIR}")
DB_PATH = os.path.join(BASE_DIR, "db", "products.db")
print(f"db path:{DB_PATH}")
VECTOR_PATH = os.path.join(BASE_DIR,"E_commerce","E_commerce","vector_store","index.faiss") 
USER_DATA_PATH = os.path.join(BASE_DIR, "E_commerce", "data", "user_data.json")
print(f"user data path:{USER_DATA_PATH}")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
# STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
# IMAGES_DIR = os.path.join(BASE_DIR, "E_commerce", "data", "images")
# USER_DATA_PATH = os.path.join(BASE_DIR, "E_commerce", "data", "user_data.json")
# VECTOR_PATH = os.path.join(BASE_DIR, "E_commerce", "vector_store")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "mps" else torch.float32, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, temperature=0.01, do_sample=False, pad_token_id=tokenizer.eos_token_id)
llm = HuggingFacePipeline(pipeline=pipe)


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

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def load_users():
    if not os.path.exists(USER_DATA_PATH):
        default = {
            "users": {},
            "orders": [],
            "next_order_id": 1000
        }
        with open(USER_DATA_PATH, "w") as f:
            json.dump(default, f, indent=2)
    with open(USER_DATA_PATH, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USER_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()


class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/register")
def register(req: LoginRequest):
    users = load_users()
    if req.username in users["users"]:
        return {"success": False, "error": "User already exists"}
    users["users"][req.username] = {
        "password": hash_password(req.password),
        "cart": [], "wishlist": [], "favorites": [],
        "conversations": [{"id": "default", "title": "New Chat", "messages": []}],
        "theme": "light",
        "past_queries": []
    }
    save_users(users)
    return {"success": True}

@app.post("/auth/login")
def login(req: LoginRequest):
    users = load_users()
    user = users["users"].get(req.username)
    if user and user["password"] == hash_password(req.password):
        return {"success": True, "username": req.username}
    return {"success": False, "error": "Invalid credentials"}


VALID_PRODUCT_FIELDS = {'product_id', 'name', 'category', 'cost', 'price', 'rating', 'review', 'reviews', 'image', 'image_path'}
VALID_CATEGORIES = {'clothing', 'sports', 'electronics', 'home', 'garden', 'books'}
PRODUCT_KEYWORDS = VALID_PRODUCT_FIELDS.union(VALID_CATEGORIES).union({'product', 'item', 'buy', 'purchase', 'show', 'tell', 'find', 'what', 'which', 'how much', 'expensive', 'cheap'})

def is_product_query(query: str) -> bool:
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    has_keyword = bool(PRODUCT_KEYWORDS.intersection(query_tokens))
    product_patterns = [
        r'\b(price|cost|rating|review)\s+(of|for)',
        r'\b(show|find|tell|get)\s+.*\b(product|item)',
        r'\b(how much|what is)\s+.*\b(price|cost)',
        r'\b(what|which)\s+.*\b(category|product)',
    ]
    has_pattern = any(re.search(p, query_lower) for p in product_patterns)
    return has_keyword or has_pattern

def calculate_relevance_score(query: str, metadata: Dict[str, Any]) -> float:
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    score = 0.0
    max_score = 6.0
    category = str(metadata.get('category', '')).lower()
    if category and category in query_lower:
        score += 2.0
    name = str(metadata.get('name', '')).lower()
    name_tokens = set(re.findall(r'\b\w+\b', name))
    common_tokens = query_tokens.intersection(name_tokens)
    if common_tokens:
        score += 3.0 * (len(common_tokens) / max(len(name_tokens), 1))
    if query_tokens.intersection({'price', 'cost', 'rating', 'review', 'reviews'}):
        score += 1.0
    return score / max_score if max_score > 0 else 0.0

def is_relevant_result(query: str, doc_metadata: Dict[str, Any], vector_score: float = None) -> Tuple[bool, str]:
    if not is_product_query(query):
        return False, "Not a product query"
    relevance = calculate_relevance_score(query, doc_metadata)
    if vector_score is not None and float(vector_score) < 0.4 and relevance > 0.3:
        return True, f"Strong match (vec: {vector_score:.3f}, rel: {relevance:.2f})"
    if relevance > 0.4:
        return True, f"Relevance: {relevance:.2f}"
    return False, f"Low relevance: {relevance:.2f}"

def extract_answer_from_query(query: str, product: Dict[str, Any]) -> str:
    q = query.lower()
    if any(w in q for w in ['rating', 'rated', 'star']): 
        return f"{float(product.get('rating', 0)):.1f}"
    if any(w in q for w in ['price', 'cost', 'how much']): 
        return f"Rs{float(product.get('cost', 0)):.2f}"
    if any(w in q for w in ['review', 'reviews']): 
        return product.get('review', "No reviews")
    if any(w in q for w in ['name', 'called']): 
        return product.get('name', 'Unknown')
    if any(w in q for w in ['category', 'type']): 
        return product.get('category', 'Unknown')
    return product.get('name', 'Product found')


@app.post("/query")
def query(request: Request):
    data = request.json()
    query = data.get("query", "").strip()
    username = data.get("username", "guest")

    if not query:
        return {"response": "Please say something.", "products": []}

    # Save query
    users = load_users()
    user = users["users"].setdefault(username, {"past_queries": []})
    user["past_queries"].append(query)
    save_users(users)

    # Try strict product search
    if is_product_query(query):
        results = vectorstore.similarity_search_with_score(query, k=1)
        if results:
            doc, score = results[0]
            meta = doc.metadata
            is_rel, reason = is_relevant_result(query, meta, score)
            if is_rel:
                product = {
                    "name": str(meta.get("name", "Unknown")),
                    "cost": float(meta.get("cost", 0)),
                    "rating": float(meta.get("rating", 0)),
                    "image": meta.get("image_path")
                }
                answer = extract_answer_from_query(query, product)
                return {"response": answer, "products": [product]}

    # Web search fallback
    try:
        resp = requests.get(f"https://ddg-api.duckduckgo.com/?q={query}&format=json", timeout=6)
        res = resp.json()
        answer = res.get("AbstractText") or (res.get("RelatedTopics")[0].get("Text") if res.get("RelatedTopics") else "No result found.")
        return {"response": answer[:500], "web_search": True}
    except:
        return {"response": "I'm having trouble connecting right now.", "products": []}


@app.post("/user/action")
def user_action(req: dict):
    data = load_users()
    username = req.get("username", "guest")
    user = data["users"].setdefault(username, {"cart": [], "wishlist": [], "favorites": []})
    action = req.get("action")
    product = req.get("product")
    if action == "cart" and product not in user["cart"]:
        user["cart"].append(product)
    elif action == "wishlist" and product not in user["wishlist"]:
        user["wishlist"].append(product)
    elif action == "favorite" and product not in user["favorites"]:
        user["favorites"].append(product)
    save_users(data)
    return {"status": "success"}

@app.get("/user/data")
def get_user_data(username: str):
    users = load_users()
    return users["users"].get(username, {"cart": [], "wishlist": [], "favorites": []})

@app.post("/user/order")
def place_order(req: dict):
    data = load_users()
    username = req.get("username", "guest")
    user = data["users"].get(username)
    if not user or not user.get("cart"):
        return {"error": "Cart is empty"}
    order_id = data["next_order_id"]
    data["next_order_id"] += 1
    order = {
        "order_id": order_id, "user_id": username, "items": user["cart"][:], "status": "Confirmed",
        "order_date": datetime.now().isoformat(), "shipping_date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
        "tracking": f"TRK{order_id}XYZ"
    }
    data["orders"].append(order)
    user["cart"] = []
    save_users(data)
    return {"order": order}

@app.get("/user/orders")
def get_orders(username: str):
    data = load_users()
    return {"orders": [o for o in data["orders"] if o["user_id"] == username]}

@app.get("/recommend")
def recommend(username: str):
    data = load_users()
    user = data["users"].get(username, {})
    terms = " ".join(user.get("past_queries", [])[-5:] + user.get("favorites", []))
    if not terms:
        return {"recommendations": []}
    results = vectorstore.similarity_search(terms, k=6)
    recs = [{"name": d.metadata.get("name"), "cost": d.metadata.get("cost"), "rating": d.metadata.get("rating"), "image": d.metadata.get("image_path")} for d in results]
    return {"recommendations": recs}

# === CHAT HISTORY ===
@app.get("/user/conversations")
def get_convos(username: str):
    users = load_users()
    user = users["users"].get(username, {})
    return {"conversations": user.get("conversations", []), "theme": user.get("theme", "light")}

@app.post("/user/save-chat")
def save_chat(username: str, convo_id: str, messages: list, theme: str = None):
    users = load_users()
    user = users["users"].get(username)
    if user:
        for convo in user["conversations"]:
            if convo["id"] == convo_id:
                convo["messages"] = messages
                convo["title"] = messages[1]["content"][:30] + "..." if len(messages) > 1 else "New Chat"
                break
        if theme:
            user["theme"] = theme
        save_users(users)
    return {"success": True}

@app.get("/user/load-chat")
def load_chat(username: str, convo_id: str):
    users = load_users()
    user = users["users"].get(username)
    if user:
        for convo in user["conversations"]:
            if convo["id"] == convo_id:
                return {"messages": convo["messages"], "theme": user.get("theme", "light")}
    return {"messages": [], "theme": "light"}

# === STATIC FILES ===
@app.get("/images/{filename}")
def get_image(filename: str):
    path = os.path.join(IMAGES_DIR, os.path.basename(filename))
    if os.path.exists(path):
        return FileResponse(path)
    placeholder = os.path.join(STATIC_DIR, "no-image.jpg")
    return FileResponse(placeholder) if os.path.exists(placeholder) else FileResponse(path)

@app.get("/")
def root():
    path = os.path.join(FRONTEND_DIR, "app2.html")
    return FileResponse(path) if os.path.exists(path) else "<h1>Server Running</h1>"

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
   
    uvicorn.run("app2:app", host="0.0.0.0", port=8000, reload=True)