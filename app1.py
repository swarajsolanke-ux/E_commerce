from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import traceback
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from typing import Tuple, Dict, Any, List, Optional
import json
from datetime import datetime

app = FastAPI(title="E-commerce Chatbot")

# Database Models
Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    product_id = Column(String, unique=True)
    name = Column(String)
    category = Column(String)
    cost = Column(Float)
    rating = Column(Float)
    review = Column(String)
    image_path = Column(String)
    stock = Column(Integer, default=10)  # Stock quantity

class CartItem(Base):
    __tablename__ = 'cart_items'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, default="default_user")
    product_id = Column(String)
    product_name = Column(String)
    quantity = Column(Integer, default=1)
    price = Column(Float)
    added_at = Column(String)

class Favorite(Base):
    __tablename__ = 'favorites'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, default="default_user")
    product_id = Column(String)
    product_name = Column(String)
    added_at = Column(String)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
DB_PATH = os.path.join(BASE_DIR, "E_commerce", "db", "products.db")
VECTOR_PATH = os.path.join(BASE_DIR, "E_commerce", "vector_store", "index.faiss")

print(f"BASE_DIR: {BASE_DIR}")
print(f"FRONTEND_DIR: {FRONTEND_DIR}")
print(f"STATIC_DIR: {STATIC_DIR}")
print(f"IMAGES_DIR: {IMAGES_DIR}")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("✓ Static files loaded")
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
print("Language model loaded")

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

engine = create_engine(f'sqlite:///{DB_PATH}')
Base.metadata.create_all(engine)
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

# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"

class CartActionRequest(BaseModel):
    product_id: str
    product_name: str
    price: float
    quantity: int = 1
    user_id: Optional[str] = "default_user"

class FavoriteRequest(BaseModel):
    product_id: str
    product_name: str
    user_id: Optional[str] = "default_user"

VALID_PRODUCT_FIELDS = {
    'product_id', 'name', 'category', 'cost', 'price', 'rating', 
    'review', 'reviews', 'image', 'image_path', 'stock'
}

VALID_CATEGORIES = {
    'clothing', 'sports', 'electronics', 'home', 'garden', 'books'
}

PRODUCT_KEYWORDS = VALID_PRODUCT_FIELDS.union(VALID_CATEGORIES).union({
    'product', 'item', 'buy', 'purchase', 'show', 'tell', 'find',
    'what', 'which', 'how much', 'expensive', 'cheap', 'cart', 'add',
    'favorite', 'like', 'wishlist', 'stock', 'available'
})

CART_KEYWORDS = {'cart', 'add', 'basket', 'buy', 'purchase', 'order'}
FAVORITE_KEYWORDS = {'favorite', 'like', 'wishlist', 'save', 'love'}
STOCK_KEYWORDS = {'stock', 'available', 'availability', 'in stock'}

def is_product_query(query: str) -> bool:
    """Check if query is related to products in the database."""
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    
    has_keyword = bool(PRODUCT_KEYWORDS.intersection(query_tokens))
    
    product_patterns = [
        r'\b(price|cost|rating|review)\s+(of|for)',
        r'\b(show|find|tell|get)\s+.*\b(product|item)',
        r'\b(how much|what is)\s+.*\b(price|cost)',
        r'\b(what|which)\s+.*\b(category|product)',
        r'\b(add|cart|buy)\b',
        r'\b(favorite|like|wishlist)\b',
        r'\b(stock|available)\b'
    ]
    
    has_pattern = any(re.search(pattern, query_lower) for pattern in product_patterns)
    
    return has_keyword or has_pattern

def detect_intent(query: str) -> str:
    """Detect user intent from query."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['add to cart', 'add cart', 'put in cart', 'buy', 'purchase']):
        return 'add_to_cart'
    elif any(word in query_lower for word in ['show cart', 'my cart', 'view cart', 'cart items']):
        return 'view_cart'
    elif any(word in query_lower for word in ['remove from cart', 'delete from cart', 'remove cart']):
        return 'remove_from_cart'
    elif any(word in query_lower for word in ['add to favorite', 'add favorite', 'like', 'add to wishlist']):
        return 'add_to_favorite'
    elif any(word in query_lower for word in ['show favorites', 'my favorites', 'wishlist', 'liked items']):
        return 'view_favorites'
    elif any(word in query_lower for word in ['stock', 'available', 'availability', 'in stock']):
        return 'check_stock'
    else:
        return 'search_product'

def calculate_relevance_score(query: str, metadata: Dict[str, Any]) -> float:
    """Calculate relevance score based on query-metadata matching."""
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
    field_keywords = {'price', 'cost', 'rating', 'review', 'reviews', 'stock'}
    if query_tokens.intersection(field_keywords):
        score += 1.0
    
    return score / max_score if max_score > 0 else 0.0

def is_relevant_result(query: str, doc_metadata: Dict[str, Any], 
                       vector_score: float = None) -> Tuple[bool, str]:
    """Determine if the retrieved document is relevant to the query."""
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
    """Extract specific answer based on query intent."""
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
    
    if any(word in query_lower for word in ['stock', 'available', 'availability']):
        stock = product.get('stock', 0)
        return f"{stock} units available" if stock > 0 else "Out of stock"
    
    if any(word in query_lower for word in ['name', 'called', 'title']):
        return product.get('name', 'Unknown product')
    
    if any(word in query_lower for word in ['category', 'type', 'kind']):
        return product.get('category', 'Unknown category')
    
    return product.get('name', 'Product information available')

# API Endpoints
@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(FRONTEND_DIR, "app1.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>E-commerce Chatbot</h1><p>Backend is running</p>")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    index_path = os.path.join(FRONTEND_DIR, "app1.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h3>UI not found</h3>", status_code=404)

@app.post("/query")
def search_products(request: QueryRequest):
    """Main query endpoint with intent detection."""
    try:
        query = request.query.strip()
        user_id = request.user_id
        
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"User: {user_id}")
        print(f"{'='*60}")
        
        intent = detect_intent(query)
        print(f"Detected intent: {intent}")
        
        # Handle cart operations
        if intent == 'view_cart':
            db = SessionLocal()
            cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()
            db.close()
            
            if not cart_items:
                return JSONResponse({
                    "response": "Your cart is empty. Start shopping by asking about products!",
                    "products": [],
                    "cart": [],
                    "intent": "view_cart"
                })
            
            cart_data = [{
                "product_name": item.product_name,
                "quantity": item.quantity,
                "price": item.price,
                "total": item.price * item.quantity
            } for item in cart_items]
            
            total = sum(item.price * item.quantity for item in cart_items)
            
            return JSONResponse({
                "response": f"You have {len(cart_items)} item(s) in your cart. Total: Rs{total:.2f}",
                "products": [],
                "cart": cart_data,
                "cart_total": total,
                "intent": "view_cart"
            })
        
        # Handle favorites view
        if intent == 'view_favorites':
            db = SessionLocal()
            favorites = db.query(Favorite).filter(Favorite.user_id == user_id).all()
            db.close()
            
            if not favorites:
                return JSONResponse({
                    "response": "You haven't added any favorites yet. Like products to save them here!",
                    "products": [],
                    "favorites": [],
                    "intent": "view_favorites"
                })
            
            fav_data = [{"product_name": fav.product_name} for fav in favorites]
            
            return JSONResponse({
                "response": f"You have {len(favorites)} favorite item(s).",
                "products": [],
                "favorites": fav_data,
                "intent": "view_favorites"
            })
        
        # Product search
        if not is_product_query(query):
            print("Query is not product-related")
            return JSONResponse({
                "response": "Sorry, I couldn't provide an answer. Please ask about products, cart, or favorites!",
                "products": [],
                "intent": "unknown"
            })
        
        print("✓ Query is product-related")
        
        results_with_scores = vectorstore.similarity_search_with_score(query, k=1)
        
        if not results_with_scores:
            print(" No results from vector store")
            return JSONResponse({
                "response": "Sorry, I couldn't find any matching products in our database.",
                "products": [],
                "intent": intent
            })
        
        top_doc, vector_score = results_with_scores[0]
        metadata = dict(getattr(top_doc, "metadata", {}) or {})
        
        print(f"Vector score: {vector_score:.4f}")
        print(f"Metadata: {metadata}")
        
        is_relevant, reason = is_relevant_result(query, metadata, vector_score)
        
        if not is_relevant:
            print(f" Result not relevant: {reason}")
            return JSONResponse({
                "response": "Sorry, I couldn't find a relevant product for your query. Please try asking about specific products!",
                "products": [],
                "intent": intent
            })
        
        print(f"✓ Result is relevant: {reason}")
        
        def safe_convert(value, type_func, default):
            try:
                return type_func(value) if value is not None else default
            except:
                return default
        
        # Get stock from database if available
        stock = 10  # Default stock
        product_id = str(metadata.get("product_id", metadata.get("name", "unknown")))
        
        product = {
            "product_id": product_id,
            "name": str(metadata.get("name", "Unknown")),
            "cost": safe_convert(metadata.get("cost"), float, 0),
            "rating": safe_convert(metadata.get("rating"), float, 0),
            "review": str(metadata.get("review", "No review")),
            "category": str(metadata.get("category", "Unknown")),
            "image": str(metadata.get("image_path")) if metadata.get("image_path") else None,
            "stock": stock
        }
        
        answer = extract_answer_from_query(query, product)
        
        # Add contextual suggestions based on intent
        if intent == 'add_to_cart':
            answer = f"{answer}. Would you like to add this to your cart?"
        elif intent == 'check_stock':
            answer = f"{product['name']}: {stock} units available"
        
        print(f"✓ Answer: {answer}")
        print(f"✓ Product: {product['name']}")
        
        return JSONResponse({
            "response": answer,
            "products": [product],
            "intent": intent,
            "suggestions": ["Add to cart", "Add to favorites", "Check similar products"]
        })
    
    except Exception as e:
        print(f"\nError processing query:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/cart/add")
def add_to_cart(request: CartActionRequest):
    """Add product to cart."""
    try:
        db = SessionLocal()
        
        # Check if item already exists
        existing = db.query(CartItem).filter(
            CartItem.user_id == request.user_id,
            CartItem.product_id == request.product_id
        ).first()
        
        if existing:
            existing.quantity += request.quantity
            db.commit()
            message = f"Updated {request.product_name} quantity to {existing.quantity}"
        else:
            cart_item = CartItem(
                user_id=request.user_id,
                product_id=request.product_id,
                product_name=request.product_name,
                quantity=request.quantity,
                price=request.price,
                added_at=datetime.now().isoformat()
            )
            db.add(cart_item)
            db.commit()
            message = f"Added {request.product_name} to your cart!"
        
        cart_count = db.query(CartItem).filter(CartItem.user_id == request.user_id).count()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": message,
            "cart_count": cart_count
        })
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cart/{user_id}")
def get_cart(user_id: str = "default_user"):
    """Get cart items for user."""
    try:
        db = SessionLocal()
        cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()
        
        cart_data = [{
            "id": item.id,
            "product_id": item.product_id,
            "product_name": item.product_name,
            "quantity": item.quantity,
            "price": item.price,
            "total": item.price * item.quantity
        } for item in cart_items]
        
        total = sum(item.price * item.quantity for item in cart_items)
        db.close()
        
        return JSONResponse({
            "cart": cart_data,
            "total": total,
            "count": len(cart_data)
        })
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cart/{user_id}/{product_id}")
def remove_from_cart(user_id: str, product_id: str):
    """Remove product from cart."""
    try:
        db = SessionLocal()
        item = db.query(CartItem).filter(
            CartItem.user_id == user_id,
            CartItem.product_id == product_id
        ).first()
        
        if not item:
            raise HTTPException(status_code=404, detail="Item not found in cart")
        
        db.delete(item)
        db.commit()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": "Item removed from cart"
        })
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/favorites/add")
def add_to_favorites(request: FavoriteRequest):
    """Add product to favorites."""
    try:
        db = SessionLocal()
        
        # Check if already favorited
        existing = db.query(Favorite).filter(
            Favorite.user_id == request.user_id,
            Favorite.product_id == request.product_id
        ).first()
        
        if existing:
            db.close()
            return JSONResponse({
                "success": True,
                "message": f"{request.product_name} is already in your favorites!",
                "already_exists": True
            })
        
        favorite = Favorite(
            user_id=request.user_id,
            product_id=request.product_id,
            product_name=request.product_name,
            added_at=datetime.now().isoformat()
        )
        db.add(favorite)
        db.commit()
        
        fav_count = db.query(Favorite).filter(Favorite.user_id == request.user_id).count()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": f"Added {request.product_name} to your favorites!",
            "favorites_count": fav_count
        })
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favorites/{user_id}")
def get_favorites(user_id: str = "default_user"):
    """Get favorite items for user."""
    try:
        db = SessionLocal()
        favorites = db.query(Favorite).filter(Favorite.user_id == user_id).all()
        
        fav_data = [{
            "id": fav.id,
            "product_id": fav.product_id,
            "product_name": fav.product_name,
            "added_at": fav.added_at
        } for fav in favorites]
        
        db.close()
        
        return JSONResponse({
            "favorites": fav_data,
            "count": len(fav_data)
        })
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/favorites/{user_id}/{product_id}")
def remove_from_favorites(user_id: str, product_id: str):
    """Remove product from favorites."""
    try:
        db = SessionLocal()
        fav = db.query(Favorite).filter(
            Favorite.user_id == user_id,
            Favorite.product_id == product_id
        ).first()
        
        if not fav:
            raise HTTPException(status_code=404, detail="Item not found in favorites")
        
        db.delete(fav)
        db.commit()
        db.close()
        
        return JSONResponse({
            "success": True,
            "message": "Item removed from favorites"
        })
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{filename}")
def get_image(filename: str):
    """Serve product images"""
    safe_name = os.path.basename(filename)
    path = os.path.join(IMAGES_DIR, safe_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "device": device,
        "vector_store_loaded": vectorstore is not None,
        "model_loaded": llm is not None
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting E-commerce Chatbot Backend")
    print("="*60)
    print(f"Device: {device}")
    print(f"Vector store: {' Loaded' if vectorstore else 'Failed'}")
    print(f"LLM: {'✓ Loaded' if llm else 'Failed'}")
    print("="*60 + "\n")
    
    uvicorn.run(
        "app1:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )