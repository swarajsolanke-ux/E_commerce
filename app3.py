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
from typing import Dict, Any, Tuple, List, Optional
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
        os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)
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


# Pydantic Models
class LoginRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    username: Optional[str] = "guest"

class UserActionRequest(BaseModel):
    username: str
    action: str
    product: str

class OrderRequest(BaseModel):
    username: str

class SaveChatRequest(BaseModel):
    username: str
    convo_id: str
    messages: List[dict]
    theme: Optional[str] = None


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
        return f"The rating is {float(product.get('rating', 0)):.1f} stars ⭐"
    if any(w in q for w in ['price', 'cost', 'how much']): 
        return f"The price is ₹{float(product.get('cost', 0)):.2f}"
    if any(w in q for w in ['review', 'reviews']): 
        return f"Reviews: {product.get('review', 'No reviews available')}"
    if any(w in q for w in ['name', 'called']): 
        return f"The product is called: {product.get('name', 'Unknown')}"
    if any(w in q for w in ['category', 'type']): 
        return f"Category: {product.get('category', 'Unknown')}"
    return f"Here's what I found: {product.get('name', 'Product found')}"


@app.post("/query")
async def query(req: QueryRequest):
    """
    Handle product queries and general questions
    """
    query = req.query.strip()
    username = req.username

    if not query:
        return {"response": "Please say something.", "products": []}

    # Save query to user history
    try:
        users = load_users()
        if username not in users["users"]:
            users["users"][username] = {
                "cart": [], "wishlist": [], "favorites": [],
                "conversations": [{"id": "default", "title": "New Chat", "messages": []}],
                "theme": "light",
                "past_queries": []
            }
        users["users"][username]["past_queries"].append(query)
        save_users(users)
    except Exception as e:
        print(f"Error saving query: {e}")

    # Try strict product search
    if is_product_query(query):
        try:
            results = vectorstore.similarity_search_with_score(query, k=3)
            if results:
                for doc, score in results:
                    meta = doc.metadata
                    is_rel, reason = is_relevant_result(query, meta, score)
                    if is_rel:
                        product = {
                            "name": str(meta.get("name", "Unknown")),
                            "cost": float(meta.get("cost", 0)),
                            "rating": float(meta.get("rating", 0)),
                            "image": meta.get("image_path", ""),
                            "description": str(meta.get("review", ""))[:100],
                            "category": str(meta.get("category", ""))
                        }
                        answer = extract_answer_from_query(query, product)
                        return {"response": answer, "products": [product]}
        except Exception as e:
            print(f"Vector search error: {e}")

    # Web search fallback for non-product queries
    try:
        resp = requests.get(
            f"https://api.duckduckgo.com/?q={query}&format=json",
            timeout=6,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        res = resp.json()
        
       
        answer = None
        if res.get("AbstractText"):
            answer = res["AbstractText"]
        elif res.get("RelatedTopics") and len(res["RelatedTopics"]) > 0:
            first_topic = res["RelatedTopics"][0]
            if isinstance(first_topic, dict) and "Text" in first_topic:
                answer = first_topic["Text"]
        
        if answer:
            return {"response": answer[:500], "web_search": True, "products": []}
        else:
            return {
                "response": "I couldn't find specific information about that. Try asking about products in our catalog!",
                "products": []
            }
    except Exception as e:
        print(f"Web search error: {e}")
        return {
            "response": "I'm having trouble connecting right now. Please try asking about products in our store!",
            "products": []
        }


@app.post("/user/action")
async def user_action(req: UserActionRequest):
    """Add items to cart, wishlist, or favorites"""
    try:
        data = load_users()
        username = req.username
        
        if username not in data["users"]:
            data["users"][username] = {
                "cart": [], "wishlist": [], "favorites": [],
                "conversations": [{"id": "default", "title": "New Chat", "messages": []}],
                "theme": "light",
                "past_queries": []
            }
        
        user = data["users"][username]
        action = req.action
        product = req.product
        
        if action == "cart" and product not in user["cart"]:
            user["cart"].append(product)
        elif action == "wishlist" and product not in user["wishlist"]:
            user["wishlist"].append(product)
        elif action == "favorite" and product not in user["favorites"]:
            user["favorites"].append(product)
        
        save_users(data)
        return {"status": "success"}
    except Exception as e:
        print(f"Error in user_action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/data")
def get_user_data(username: str):
    """Get user's cart, wishlist, and favorites"""
    users = load_users()
    return users["users"].get(username, {"cart": [], "wishlist": [], "favorites": []})


@app.post("/user/order")
async def place_order(req: OrderRequest):
    """Place an order from cart"""
    try:
        data = load_users()
        username = req.username
        user = data["users"].get(username)
        
        if not user or not user.get("cart"):
            return {"error": "Cart is empty"}
        
        order_id = data["next_order_id"]
        data["next_order_id"] += 1
        
        order = {
            "order_id": order_id,
            "user_id": username,
            "items": user["cart"][:],
            "status": "Confirmed",
            "order_date": datetime.now().isoformat(),
            "shipping_date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
            "tracking": f"TRK{order_id}XYZ"
        }
        
        data["orders"].append(order)
        user["cart"] = []
        save_users(data)
        
        return {"order": order}
    except Exception as e:
        print(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/orders")
def get_orders(username: str):
    """Get user's order history"""
    data = load_users()
    return {"orders": [o for o in data["orders"] if o["user_id"] == username]}


@app.get("/recommend")
def recommend(username: str):
    """Get personalized recommendations"""
    try:
        data = load_users()
        user = data["users"].get(username, {})
        
        # Build query from user history
        terms = " ".join(
            user.get("past_queries", [])[-5:] + 
            user.get("favorites", []) + 
            user.get("wishlist", [])
        )
        
        if not terms:
            # Return some default popular items
            results = vectorstore.similarity_search("popular products", k=6)
        else:
            results = vectorstore.similarity_search(terms, k=6)
        
        recs = []
        for d in results:
            recs.append({
                "name": d.metadata.get("name", "Unknown"),
                "cost": float(d.metadata.get("cost", 0)),
                "rating": float(d.metadata.get("rating", 0)),
                "image": d.metadata.get("image_path", "")
            })
        
        return {"recommendations": recs}
    except Exception as e:
        print(f"Error in recommend: {e}")
        return {"recommendations": []}



@app.get("/user/conversations")
def get_convos(username: str):
    """Get user's chat conversations"""
    users = load_users()
    user = users["users"].get(username, {})
    return {
        "conversations": user.get("conversations", []),
        "theme": user.get("theme", "light")
    }


@app.post("/user/save-chat")
async def save_chat(req: SaveChatRequest):
    """Save chat conversation"""
    try:
        users = load_users()
        user = users["users"].get(req.username)
        
        if user:
            # Find and update conversation
            convo_exists = False
            for convo in user["conversations"]:
                if convo["id"] == req.convo_id:
                    convo["messages"] = req.messages
                    if len(req.messages) > 1:
                        convo["title"] = req.messages[0]["content"][:30] + "..."
                    convo_exists = True
                    break
            
            # Create new conversation if doesn't exist
            if not convo_exists:
                title = req.messages[0]["content"][:30] + "..." if req.messages else "New Chat"
                user["conversations"].append({
                    "id": req.convo_id,
                    "title": title,
                    "messages": req.messages
                })
            
            if req.theme:
                user["theme"] = req.theme
            
            save_users(users)
        
        return {"success": True}
    except Exception as e:
        print(f"Error saving chat: {e}")
        return {"success": False, "error": str(e)}


@app.get("/user/load-chat")
def load_chat(username: str, convo_id: str):
    """Load specific chat conversation"""
    users = load_users()
    user = users["users"].get(username)
    
    if user:
        for convo in user["conversations"]:
            if convo["id"] == convo_id:
                return {
                    "messages": convo["messages"],
                    "theme": user.get("theme", "light")
                }
    
    return {"messages": [], "theme": "light"}



@app.get("/images/{filename}")
def get_image(filename: str):
    """Serve product images"""
    path = os.path.join(IMAGES_DIR, os.path.basename(filename))
    if os.path.exists(path):
        return FileResponse(path)
    
    # Return placeholder if image not found
    placeholder = os.path.join(STATIC_DIR, "no-image.jpg")
    if os.path.exists(placeholder):
        return FileResponse(placeholder)
    
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/")
def root():
    """Serve the main HTML file"""
    path = os.path.join(FRONTEND_DIR, "app2.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"status": "Server Running", "message": "Frontend not found"})


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run("app3:app", host="192.168.5.155", port=8000, reload=True)
#192.168.5.255