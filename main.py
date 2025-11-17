
import os
import re
import traceback
import json
import logging
from typing import Tuple, Dict, Any, List, Set, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from database import (
    init_user_db, hash_password, verify_password, get_user_by_email,
    create_user, save_chat_history, get_chat_history, user_exists
)
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
DB_PATH = os.path.join(BASE_DIR, "db", "products_DB.db")
VECTOR_DIR = os.path.join(BASE_DIR, "vect", "vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model_path = "gpt2"

# Initialize user database
init_user_db()

K_SEMANTIC_CANDIDATES = 20
TOP_RETURN = 1

app = FastAPI(title="E-commerce Chatbot")

# Static & CORS
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = None
try:
    print("Loading FAISS vectorstore...")
    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded")
except Exception as e:
    print("Failed to load vectorstore:", e)

# Greeting detection
GREETING_WORDS: Set[str] = {
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening","hiiiii","helo","hii"
    "hiii", "helo", "morning", "afternoon", "evening", "sup", "yo", "howdy","hiiiiii","helllo","heyyyyy","HELLOOO"
}

# Query intent patterns - Enhanced to handle all query types
INTENT_PATTERNS = {
    "list_all": [
        r"\b(all|every|entire|complete)\s+(products?|items?|things?)",
        r"\b(show|display|list|give)\s+me\s+(all|everything)",
        r"\bwhat\s+(products?|items?)\s+do\s+you\s+have",
        r"\bwhat'?s?\s+available",
        r"\bshow\s+catalog",
        r"\bview\s+all",
        r"\bcan\s+you\s+show\s+me\s+all\s+products?\s+availables?",
        r"\bcan\s+you\s+show\s+me\s+all\s+products?",
        r"\bcan\s+you\s+show\s+all\s+available\s+products?",
        r"\bshow\s+me\s+all\s+products?\s+availables?",
        r"\bshow\s+me\s+all\s+the\s+products?",
        r"\blist\s+all\s+available\s+items?",
        r"\bgive\s+me\s+the\s+list\s+of\s+products?\s+available",
        r"\bwhat\s+can\s+i\s+buy\s+from\s+your\s+store",
        r"\bshow\s+all\s+available\s+products?",
    ],
    "list_categories": [
        r"\b(show|list|display|give)\s+me\s+(all\s+)?(product\s+)?categor(ies|y)",
        r"\bwhat\s+(are\s+)?(the\s+)?(product\s+)?categor(ies|y)",
        r"\ball\s+(product\s+)?categor(ies|y)",
        r"\bcan\s+you\s+show\s+me\s+all\s+product\s+categor(ies|y)",
        r"\bshow\s+me\s+all\s+product\s+categor(ies|y)",
        r"\bcategor(ies|y)\s+list",
    ],
    "product_by_name": [
        r"\bdo\s+you\s+have\s+(.+?)\??$",
        r"\bsearch\s+for\s+[\"']?(.+?)[\"']?",
        r"\bshow\s+me\s+(.+?)\s*$",
        r"\bfind\s+(product\s+)?named?\s+[\"']?(.+?)[\"']?",
        r"\blooking\s+for\s+(.+?)\s*$",
        r"\b(nike|iphone|samsung|redmi|adidas|sony|apple|denim|jeans)\s+",
    ],
    "category": [
        r"\b(show|list|display|find|get|can\s+you\s+show)\s+.{0,30}?(electronics?|clothing|sports?|home|fashion|tech|appliances?)",
        r"\b(electronics?|clothing|sports?|home|fashion|tech|appliances?)\s+(products?|items?|section)",
        r"\bin\s+the\s+(electronics?|clothing|sports?|home|fashion|tech|appliances?)",
        r"\bcategory\s+of\s+",
        r"\bshow\s+me\s+electronics?\s+products?",
        r"\bcan\s+you\s+show\s+me\s+(electronics?|clothing|sports?|home)\s+products?",
        r"\bwhat\s+clothing\s+items?\s+are\s+available",
        r"\blist\s+all\s+sports?\s+items?",
        r"\bdo\s+you\s+have\s+anything\s+in\s+home\s+appliances?",
        r"\bshow\s+me\s+sports?\s+products?",
        r"\bshow\s+me\s+sports?\s+product",
        r"\bshow\s+me\s+Bady and Kids?\s+product?",
        r"\bshow\s+me\s+Home and Furniture?\s+product?",
        r"\bcan\s+you\s+show\s+me all product categories?",
    ],
    "price_range": [
        r"\b(under|below|less\s+than|cheaper\s+than|within)\s+.*?(\d+)",
        r"\b(above|over|more\s+than|expensive\s+than)\s+.*?(\d+)",
        r"\bbetween\s+.*?(\d+)\s*[-–]\s*(\d+)",
        r"\bbetween\s+.*?(\d+)\s+and\s+(\d+)",
        r"\b(₹|rs\.?|rupees?)\s*(\d+)",
        r"\b(\d+)\s*(₹|rs\.?|rupees?)",
        r"\bbudget\s+of\s+(\d+)",
        r"\bprice\s+range",
        r"\bshow\s+products?\s+between\s+",
        r"\blist\s+items?\s+under\s+",
        r"\bshow\s+me\s+products?\s+above\s+",
        r"\bi\s+want\s+something\s+under\s+budget\s+",
        r"\b(suggest|recommend|show|find)\s+.*?\s+under\s+.*?(\d+)",
        r"\b(suggest|recommend|show|find)\s+.*?\s+below\s+.*?(\d+)",
        r"\b.*?\s+under\s+.*?(\d+)\s*(₹|rs\.?|rupees?)",
        r"\b.*?\s+below\s+.*?(\d+)\s*(₹|rs\.?|rupees?)",
    ],
    "cheapest": [
        r"\b(cheapest|most\s+affordable|lowest\s+price|least\s+expensive|budget\s+friendly)",
        r"\bwhat'?s?\s+the\s+cheapest",
        r"\bminimum\s+price",
        r"\bwhich\s+is\s+the\s+cheapest\s+product",
        r"\bwhich\s+is\s+the\s+cheapest",
        r"\bshow\s+me\s+the\s+most\s+affordable",
        r"\bwhich\s+one\s+is\s+more\s+affordable\s+between",
    ],
    "most_expensive": [
        r"\b(most\s+expensive|highest\s+price|premium|luxury|costliest)",
        r"\bwhat'?s?\s+the\s+most\s+expensive",
        r"\bmaximum\s+price",
        r"\bshow\s+me\s+the\s+most\s+expensive\s+item",
    ],
    "highest_rating": [
        r"\b(highest|best|top)\s+(rated?|rating|reviewed?)",
        r"\bbest\s+quality",
        r"\bmost\s+popular",
        r"\btop\s+product",
        r"\bwhich\s+product\s+has\s+the\s+highest\s+rating",
        r"\bshow\s+me\s+top[- ]rated\s+items?", 
        r"\blist\s+products?\s+rated\s+above\s+(\d+)\s+stars?",
        r"\bgive\s+me\s+products?\s+with\s+good\s+reviews?",
    ],
    "compare": [
        r"\bcompare\s+.+\s+(and|vs|versus|with)",
        r"\b.+\s+(or|vs|versus)\s+.+\?",
        r"\bwhich\s+is\s+(better|cheaper|best)",
        r"\bdifference\s+between",
        r"\bcompare\s+product\s+[a-z]\s+and\s+product\s+[a-z]",
        r"\bwhich\s+one\s+should\s+i\s+buy",
        r"\bcompare\s+prices?\s+of\s+",
        r"\bbetween\s+product\s+[a-z]\s+and\s+[a-z],?\s+which\s+is\s+cheaper",
        r"\bgive\s+me\s+the\s+differences?\s+between",
    ],
    "recommend": [
        r"\b(recommend|suggest|advice|help\s+me\s+choose)",
        r"\bwhat\s+should\s+i\s+(buy|get|purchase)",
        r"\bbest\s+for\s+",
        r"\bgift\s+for",
        r"\blooking\s+for",
        r"\bneed\s+a\s+",
        r"\brecommend\s+a\s+good\s+",
        r"\bwhat\s+should\s+i\s+buy\s+for\s+",
        r"\bsuggest\s+a\s+gift\s+for\s+",
        r"\brecommend\s+top\s+",
        r"\bsuggest\s+something\s+best[- ]selling",
    ],
}

# Compile all patterns
COMPILED_PATTERNS = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in INTENT_PATTERNS.items()
}

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
            max_new_tokens=512,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True,
            return_full_text=False
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"llm is callling:{llm}")

        PROMPT = PromptTemplate(
            template="""You are an e-commerce assistant. Answer ONLY in valid JSON format.

CONTEXT:
{context}

QUESTION: {question}

Determine the query type and respond with the appropriate JSON schema:

1. LIST ALL - User wants to see all products
{{"type": "list_all", "response": "Found X products.", "products": [...]}}

2. LIST CATEGORIES - User wants to see categories
{{"type": "list_categories", "response": "Available categories: ...", "categories": [...]}}

3. SINGLE PRODUCT - User asks about a specific product
{{"type": "single", "response": "<title> costs Rs<price> with <rating> stars", "product": {{...}}}}

4. PRICE RANGE - User mentions price constraints
{{"type": "price_range", "response": "Found X products in your budget.", "min": X, "max": Y, "products": [...]}}

5. CHEAPEST - User wants the cheapest option
{{"type": "cheapest", "response": "Cheapest: '<title>' for Rs<price>.", "product": {{...}}}}

6. MOST EXPENSIVE - User wants the most expensive option
{{"type": "most_expensive", "response": "Most expensive: '<title>' for Rs<price>.", "product": {{...}}}}

7. HIGHEST RATING - User wants best rated product
{{"type": "highest_rating", "response": "Highest rated: '<title>' with <rating> stars.", "product": {{...}}}}

8. COMPARE - User wants to compare products
{{"type": "compare", "response": "Comparing...", "product_a": {{...}}, "product_b": {{...}}, "cheaper": {{...}}}}

9. RECOMMEND - User wants recommendations
{{"type": "recommend", "response": "I recommend...", "products": [...]}}

10. NOT FOUND - No matching products
{{"type": "not_found", "response": "Sorry, no matching products found."}}

RULES:
- Parse price from: ₹, Rs, rupees, numbers
- For price queries: extract min/max values
- For comparisons: identify product names
- For categories: match to available categories
- Return ONLY valid JSON, no explanations
- Product schema: {{"title": "...", "selling_price": "...", "product_rating": X.X, "description": "...", "image": "..."}}

ANSWER (JSON only):
""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )
        print("LLM loaded and ready")
    except Exception as e:
        print("Failed to load LLM:", e)
        traceback.print_exc()
        llm = None
        qa_chain = None


def _all_docs_from_vectorstore() -> List[Any]:
    if not vectorstore:
        return []
    try:
        return list(getattr(vectorstore.docstore, "_dict", {}).values())
    except:
        return []

def doc_to_metadata(doc) -> Dict[str, Any]:
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
    try:
        if not value:
            return 0.0
        s = re.sub(r"[^\d.]", "", str(value))
        return float(s) if s else 0.0
    except:
        return 0.0

def parse_rating(value: Any) -> float:
    try:
        return float(str(value).strip())
    except:
        return 0.0

def get_primary_category(metadata: Dict[str, Any]) -> str:
    for k in ("category_2", "category_1", "category_3", "product_category"):
        v = str(metadata.get(k, "")).strip()
        if v and v.lower() not in ["nan", "none", ""]:
            return v
    return ""

def doc_to_card(doc) -> dict:
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
        "title": meta.get("title") or meta.get("name") or "Unknown",
        "selling_price": meta.get("selling_price") or meta.get("cost") or "",
        "product_rating": parse_rating(meta.get("product_rating") or meta.get("rating")),
        "description": (meta.get("description") or "")[:280],
        "image": image
    }

def detect_intent(query: str) -> Optional[str]:
    """Detect the primary intent of the query"""
    query_lower = query.lower()
    print(f"detect the intent:{query_lower}")
    
    # Priority order: check specific intents first
    priority_intents = ["cheapest", "most_expensive", "highest_rating", "list_categories", "list_all", "price_range", "category", "compare", "recommend", "product_by_name"]
    
    for intent in priority_intents:
        if intent in COMPILED_PATTERNS:
            for pattern in COMPILED_PATTERNS[intent]:
                if pattern.search(query_lower):
                    return intent
    
    # Fallback: check all patterns
    for intent, patterns in COMPILED_PATTERNS.items():
        if intent not in priority_intents:
            for pattern in patterns:
                print(f"pattern:{pattern}")
                if pattern.search(query_lower):
                    return intent
    
    return None

def extract_price_values(query: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract min and max price values from query"""
    query_lower = query.lower()
    
    # Between X and Y
    between_match = re.search(r'between\s+.*?(\d+)\s+and\s+(\d+)', query_lower)
    if between_match:
        return float(between_match.group(1)), float(between_match.group(2))
    
    # Between X-Y (with dash)
    between_dash = re.search(r'(\d+)\s*[-–]\s*(\d+)', query_lower)
    if between_dash:
        return float(between_dash.group(1)), float(between_dash.group(2))
    
    # Under/below X (with rs/rupees)
    under_match = re.search(r'(under|below|less\s+than|within)\s+.*?(\d+)', query_lower)
    if under_match:
        return 0, float(under_match.group(2))
    
    # Above/over X
    above_match = re.search(r'(above|over|more\s+than)\s+.*?(\d+)', query_lower)
    if above_match:
        return float(above_match.group(2)), float('inf')
    
    # Price with rs/rupees/₹ (e.g., "under 400rs", "below 500 rupees")
    price_rs = re.search(r'(under|below|less\s+than)\s+.*?(\d+)\s*(₹|rs\.?|rupees?)', query_lower)
    if price_rs:
        return 0, float(price_rs.group(2))
    
    # Single price mention (only if it's clearly a price constraint)
    if any(word in query_lower for word in ['under', 'below', 'less', 'within', 'budget', 'price', 'rs', 'rupees', '₹']):
        price_match = re.search(r'(\d+)', query_lower)
        if price_match:
            price = float(price_match.group(1))
            return 0, price
    
    return None, None

def extract_product_name_from_query(query: str) -> Optional[str]:
    """Extract product name from query, removing price-related words"""
    query_lower = query.lower()
    
    # Remove common price-related phrases
    price_phrases = [
        r'\b(under|below|less\s+than|within|above|over|more\s+than)\s+.*?\d+.*?(₹|rs\.?|rupees?)?',
        r'\b\d+\s*(₹|rs\.?|rupees?)',
        r'\b(₹|rs\.?|rupees?)\s*\d+',
        r'\bbudget\s+of\s+\d+',
        r'\bprice\s+range',
    ]
    
    cleaned = query_lower
    print(f"cleaned query:{cleaned}")
    for phrase in price_phrases:
        cleaned = re.sub(phrase, '', cleaned, flags=re.IGNORECASE)
    
    # Remove common action words
    action_words = ['show', 'me', 'can', 'you', 'suggest', 'recommend', 'find', 'get', 'give', 'list', 'display']
    words = cleaned.split()
    product_words = [w for w in words if w not in action_words and len(w) > 2]
    
    if product_words:
        return ' '.join(product_words).strip()
    
    return None

def extract_category(query: str) -> Optional[str]:
    """Extract category from query"""
    query_lower = query.lower()
    
    categories = {
        'electronics': ['electronic', 'electronics', 'tech', 'gadget', 'device', 'laptop', 'phone', 'mobile'],
        'sports': ['sport', 'sports', 'fitness', 'gym', 'exercise', 'cricket', 'bat', 'ball', 'football'],
        'clothing': ['cloth', 'clothing', 'apparel', 'fashion', 'wear', 'shirt', 'pant', 'jeans', 'dress'],
        'home': ['home', 'house', 'kitchen', 'furniture', 'appliance', 'appliances']
    }
    
    for cat, keywords in categories.items():
        if any(kw in query_lower for kw in keywords):
            return cat
    
    return None

def build_filtered_context(docs: List[Any], max_products: int =20) -> str:
    """Build context string from documents"""
    lines = []
    for doc in docs[:max_products]:
        card = doc_to_card(doc)
        cat = get_primary_category(doc_to_metadata(doc))
        lines.append(
            f"title:{card['title']} price:{card['selling_price']} rating:{card['product_rating']} category:{cat}"
        )
    return "\n".join(lines) if lines else "No products available."

def get_recommendations(main_meta: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """Get recommended products based on main product metadata"""
    try:
        if not vectorstore:
            return []
        
        cat = get_primary_category(main_meta)
        print(f"product categories:{cat}")
        main_title = (main_meta.get("title") or main_meta.get("name") or "").lower()
        
        if not cat or not main_title:
            return []
        
        # Search for similar products in the same category
        query_text = f"{main_title} {cat}"
        results = vectorstore.similarity_search_with_score(query_text, k=10)
        
        recs = []
        seen_titles = {main_title}
        
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, _score = item
            else:
                doc = item
            
            m = doc_to_metadata(doc)
            title = str(m.get("title") or m.get("name") or "").lower()
            prod_cat = get_primary_category(m)
            
        
            if not title or title in seen_titles or prod_cat.lower() != cat.lower():
                continue
            
            rec_card = doc_to_card(doc)
            recs.append(rec_card)
            seen_titles.add(title)
            
            if len(recs) >= k:
                break
        
        return recs
    except Exception as e:
        print(f"[REC ERROR] {e}")
        traceback.print_exc()
        return []

def get_all_categories() -> List[str]:
    """Get all unique product categories from vectorstore"""
    if not vectorstore:
        return []
    try:
        all_docs = _all_docs_from_vectorstore()
        categories = set()
        for doc in all_docs:
            meta = doc_to_metadata(doc)
            for key in ["category_1", "category_2", "category_3", "product_category"]:
                val = str(meta.get(key, "")).strip()
                if val and val.lower() not in ["nan", "none", ""]:
                    categories.add(val)
        return sorted(list(categories))
    except Exception as e:
        print(f"Error fetching categories: {e}")
        return []

def is_list_all_products_query(query: str) -> bool:
    """Check if query is asking for all products"""
    query_lower = query.lower().strip()
    patterns = [
        r"can\s+you\s+show\s+me\s+all\s+products?\s+availables?",
        r"can\s+you\s+show\s+all\s+available\s+products?",
        r"show\s+me\s+all\s+products?\s+availables?",
        r"show\s+all\s+available\s+products?",
        r"list\s+all\s+products?\s+availables?",
        r"all\s+products?\s+availables?",
        r"show\s+.*?\s+all\s+.*?products?",
    ]
    for pattern in patterns:
        if re.search(pattern, query_lower):
            return True
    
    if "show" in query_lower and "all" in query_lower and ("product" in query_lower or "available" in query_lower):
        return True
    return False

def is_list_categories_query(query: str) -> bool:
    """Check if query is asking for product categories"""
    query_lower = query.lower().strip()
    patterns = [
        r"can\s+you\s+show\s+me\s+all\s+product\s+categor(ies|y)",
        r"show\s+me\s+all\s+product\s+categor(ies|y)",
        r"list\s+all\s+product\s+categor(ies|y)",
        r"what\s+(are\s+)?(the\s+)?(product\s+)?categor(ies|y)",
    ]
    for pattern in patterns:
        if re.search(pattern, query_lower):
            return True
    return False




class QueryRequest(BaseModel):
    query: str
    user_id: Optional[int] = None  
class LoginRequest(BaseModel):
    email: str
    username: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str

class LogoutRequest(BaseModel):
    user_id: int


@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(400, "Empty query")

        user_id = request.user_id
        response_text = ""
        response_products = []

        # Handle greetings
        if any(word in query.lower().split() for word in GREETING_WORDS):
            response_text = "welcome to the E-commerce world! I'm your shopping assistant—let me know what product you're looking for, and I'll help you find it. How can I assist you today?"
            response_data = {"response": response_text}
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, None)
            
            return JSONResponse(response_data)

        
        query_lower = query.lower()
        if is_list_all_products_query(query) or "show" in query_lower and "all" in query_lower and ("product" in query_lower or "available" in query_lower):
            all_docs = _all_docs_from_vectorstore()
            if not all_docs:
                response_text = "Sorry, no products are currently available in our database."
                response_data = {
                    "response": response_text,
                    "main_product": None,
                    "recommendations": []
                }
            else:
                # Format as list
                product_list = []
                for doc in all_docs[:20]:  
                    card = doc_to_card(doc)
                    product_list.append(f"• {card['title']} - Rs {card['selling_price']} (Rating: {card['product_rating']})")
                
                response_text = f"Here are all available products:\n\n" + "\n".join(product_list)
                # Return first product with recommendations
                if all_docs:
                    main_doc = all_docs[0]
                    main_product = doc_to_card(main_doc)
                    main_meta = doc_to_metadata(main_doc)
                    recommendations = get_recommendations(main_meta, k=3)
                    response_data = {
                        "response": response_text,
                        "main_product": main_product,
                        "recommendations": recommendations,
                        "list_format": True
                    }
                else:
                    response_data = {
                        "response": response_text,
                        "main_product": None,
                        "recommendations": []
                    }
            
            
            if user_id:
                save_chat_history(user_id, query, response_text, [response_data.get("main_product")] if response_data.get("main_product") else [])
            
            return JSONResponse(response_data)

        # Check for specific queries: "show me all product categories"
        if is_list_categories_query(query) or detect_intent(query) == "list_categories":
            categories = get_all_categories()
            print(f"categories available in the database:{categories}")
            if not categories:
                response_text = "No product categories found in the database."
                response_data = {
                    "response": response_text,
                    "categories":[20],
                }
            else:
                
                category_list = [f"• {cat}" for cat in categories]
                response_text = "Here are all available product categories:\n\n" + "\n".join(category_list)
                response_data = {
                    "response": response_text,
                    "categories": categories,
                    "list_format": True
                }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, None)
            
            return JSONResponse(response_data)

        # Detect intent - check for price_range first (has priority for combined queries)
        intent = detect_intent(query)
        
        # Special handling: if query has both price constraint and product name, prioritize price_range
        query_lower = query.lower()
        has_price = any(word in query_lower for word in ['under', 'below', 'less', 'within', 'budget', 'price', 'rs', 'rupees', '₹']) and re.search(r'\d+', query_lower)
        has_product_name = extract_product_name_from_query(query) is not None
        
        if has_price and has_product_name and intent != "price_range":
            # Force price_range intent for queries like "cricket bat under 400rs"
            intent = "price_range"
        
        print(f"Query: {query}")
        print(f"Detected intent: {intent}")

        # Get all documents
        all_docs = _all_docs_from_vectorstore()
        
        if not all_docs:
            response_text = "Sorry, no products are currently available in our database."
            response_data = {
                "response": response_text,
                "main_product": None,
                "recommendations": []
            }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [])
            
            return JSONResponse(response_data)

        # Handle price_range with product name (e.g., "cricket bat under 400rs")
        if intent == "price_range":
            min_price, max_price = extract_price_values(query)
            product_name = extract_product_name_from_query(query)
            
          
            filtered_docs = []
            if min_price is not None or max_price is not None or product_name:
                # If product name is specified, search for it first
                if product_name:
                    # Use semantic search to find products matching the name
                    name_docs = vectorstore.similarity_search(product_name, k=30)
                else:
                    name_docs = all_docs
                
                # Then filter by price
                for d in name_docs:
                    price = parse_price(doc_to_metadata(d).get("selling_price"))
                    price_ok = True
                    
                    if min_price is not None and max_price is not None:
                        if not (min_price <= price <= max_price):
                            price_ok = False
                    elif max_price is not None:
                        if price > max_price or price == 0:
                            price_ok = False
                    elif min_price is not None:
                        if price < min_price:
                            price_ok = False
                    
                    if price_ok:
                        filtered_docs.append(d)
                
                if not filtered_docs:
                    if product_name:
                        response_text = f"Sorry, no products matching '{product_name}' found in that price range."
                    else:
                        response_text = f"Sorry, no products found in that price range."
                    response_data = {
                        "response": response_text,
                        "main_product": None,
                        "recommendations": []
                    }
                    
                    if user_id:
                        save_chat_history(user_id, query, response_text, [])
                    
                    return JSONResponse(response_data)
            else:
                
                filtered_docs = vectorstore.similarity_search(query, k=20)
            
            # Get first product as main product
            main_doc = filtered_docs[0] if filtered_docs else None
            if main_doc:
                main_product = doc_to_card(main_doc)
                main_meta = doc_to_metadata(main_doc)
                recommendations = get_recommendations(main_meta, k=3)
                
                if product_name:
                    response_text = f"Found '{main_product['title']}'."
                else:
                    response_text = f"Found products in your price range. Here's one: '{main_product['title']}'"
                response_data = {
                    "response": response_text,
                    "main_product": main_product,
                    "recommendations": recommendations
                }
            else:
                response_text = "No products found."
                response_data = {
                    "response": response_text,
                    "main_product": None,
                    "recommendations": []
                }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [main_product] if main_doc else [])
            
            return JSONResponse(response_data)

        # Handle product search by name
        if intent == "product_by_name":
            # Use semantic search to find products by name
            relevant_docs = vectorstore.similarity_search(query, k=10)
            if not relevant_docs:
                response_text = f"Sorry, I couldn't find any products matching '{query}'."
                print(response_text)
                response_data = {
                    "response": response_text,
                    "main_product": None,
                    "recommendations": []
                }
            else:
                # Get first product as main product
                main_doc = relevant_docs[0]
                main_product = doc_to_card(main_doc)
                main_meta = doc_to_metadata(main_doc)
                
                # Get 3 recommendations
                recommendations = get_recommendations(main_meta, k=3)
                print(f"recommendate products:{recommendations}")
                
                response_text = f"I found '{main_product['title']}'."
                response_data = {
                    "response": response_text,
                    "main_product": main_product,
                    "recommendations": recommendations
                }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [response_data.get("main_product")] if response_data.get("main_product") else [])
            
            return JSONResponse(response_data)

        # Handle based on intent
        if intent == "list_all":
            # Format as list
            product_list = []
            for doc in all_docs[:20]:  # Limit to 50 for display
                card = doc_to_card(doc)
                product_list.append(f"• {card['title']} - Rs {card['selling_price']} (Rating: {card['product_rating']})")
            
            response_text = f"Here are all available products:\n\n" + "\n".join(product_list)
            response_products = [doc_to_card(d) for d in all_docs]
            
            response_data = {
                "response": response_text,
                "products": response_products,
                "list_format": True
            }
            
            print(f"response data:{response_data}")
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, response_products)
            
            return JSONResponse(response_data)

        elif intent == "category":
            category = extract_category(query)
            if not category:
                # Use semantic search
                relevant_docs = vectorstore.similarity_search(query, k=20)
            else:
                # Filter by category
                relevant_docs = [
                    d for d in all_docs
                    if category in get_primary_category(doc_to_metadata(d)).lower()
                ][:20]
            
            if not relevant_docs:
                return JSONResponse({
                    "response": f"Sorry, I couldn't find any products matching '{query}'.",
                    "main_product": None,
                    "recommendations": []
                })
            
            # Get first product as main product
            main_doc = relevant_docs[0]
            main_product = doc_to_card(main_doc)
            main_meta = doc_to_metadata(main_doc)
            recommendations = get_recommendations(main_meta, k=3)
            
            response_text = f"Found products in this category. Here's one: '{main_product['title']}'"
            response_data = {
                "response": response_text,
                "main_product": main_product,
                "recommendations": recommendations,
                "debug": {"intent": "category", "category": category}
            }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [main_product])
            
            return JSONResponse(response_data)

        elif intent == "cheapest":
            # Filter out products with invalid prices (0 or None)
            valid_docs = [
                d for d in all_docs
                if parse_price(doc_to_metadata(d).get("selling_price")) > 0
            ]
            
            if not valid_docs:
                return JSONResponse({
                    "response": "No products with valid prices available.",
                    "main_product": None,
                    "recommendations": []
                })
            
            # Sort by price (ascending)
            sorted_docs = sorted(valid_docs, key=lambda d: parse_price(doc_to_metadata(d).get("selling_price")))
            cheapest = sorted_docs[0]
            
            main_product = doc_to_card(cheapest)
            main_meta = doc_to_metadata(cheapest)
            recommendations = get_recommendations(main_meta, k=3)
            
            response_text = f"The cheapest product is '{main_product['title']}' for Rs{main_product['selling_price']}."
            response_data = {
                "response": response_text ,
                "main_product": main_product,
                "recommendations": recommendations
            }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [main_product])
            
            return JSONResponse(response_data)

        elif intent == "most_expensive":
            sorted_docs = sorted(all_docs, key=lambda d: parse_price(doc_to_metadata(d).get("selling_price")), reverse=True)
            most_exp = sorted_docs[0] if sorted_docs else None
            
            if not most_exp:
                return JSONResponse({"response": "No products available.", "main_product": None, "recommendations": []})
            
            main_product = doc_to_card(most_exp)
            main_meta = doc_to_metadata(most_exp)
            recommendations = get_recommendations(main_meta, k=3)
            
            response_text = f"The most expensive product is '{main_product['title']}' for Rs{main_product['selling_price']}."
            response_data = {
                "response": response_text,
                "main_product": main_product,
                "recommendations": recommendations
            }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [main_product])
            
            return JSONResponse(response_data)

        elif intent == "highest_rating":
            sorted_docs = sorted(all_docs, key=lambda d: parse_rating(doc_to_metadata(d).get("product_rating")), reverse=True)
            highest = sorted_docs[0] if sorted_docs else None
            
            if not highest:
                return JSONResponse({"response": "No products available.", "main_product": None, "recommendations": []})
            
            main_product = doc_to_card(highest)
            main_meta = doc_to_metadata(highest)
            recommendations = get_recommendations(main_meta, k=3)
            
            response_text = f"The highest rated product is '{main_product['title']}' with {main_product['product_rating']} stars."
            response_data = {
                "response": response_text,
                "main_product": main_product,
                "recommendations": recommendations
            }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [main_product])
            
            return JSONResponse(response_data)

        # For all other cases, use semantic search + LLM
        if not qa_chain:
            raise HTTPException(500, "LLM not available")

        # Use semantic search to get relevant products
        relevant_docs = vectorstore.similarity_search(query, k=30)
        context = build_filtered_context(relevant_docs)
        
        filled = PROMPT.format(context=context, question=query)
        print(f"filled prompt:{filled}")
        raw = llm.invoke(filled)
        print(f"llm raw output:{raw}")
        llm_output = raw.strip()
       
        
        print(f"LLM Output: {llm_output[:10]}")

        json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not json_match:
            # Fallback response - return first product with recommendations
            if relevant_docs:
                main_doc = relevant_docs[0]
                main_product = doc_to_card(main_doc)
                print(f"main_product:{main_product}")
                main_meta = doc_to_metadata(main_doc)
                print(f"main_meta:{main_meta}")
                recommendations = get_recommendations(main_meta, k=3)
                
                response_text = "Here are some products that might interest you:"
                response_data = {
                    "response": response_text,
                    "main_product": main_product,
                    "recommendations": recommendations,
                    "debug": {"fallback": True, "intent": intent}
                }
                print(f"response_data:{response_data}")
            else:
                response_text = "I couldn't find any products matching your query."
                response_data = {
                    "response": response_text,
                    "main_product": None,
                    "recommendations": []
                }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [response_data.get("main_product")] if response_data.get("main_product") else [])
            
            return JSONResponse(response_data)

        payload = json.loads(json_match.group(0))
        typ = payload.get("type", "not_found")

        # Handle different response types
        if typ in ("single", "cheapest", "most_expensive", "highest_rating"):
            prod = payload.get("product")
            response_text = payload.get("response", "")
            
            if prod:
                # Convert product dict to metadata format for recommendations
                main_meta = {
                    "title": prod.get("title") or prod.get("name"),
                    "category_1": prod.get("category") or "",
                    "category_2": "",
                    "category_3": ""
                }
                recommendations = get_recommendations(main_meta, k=3)
                response_data = {
                    "response": response_text,
                    "main_product": prod,
                    "recommendations": recommendations,
                    "debug": {"type": typ, "intent": intent}
                }
            else:
                response_data = {
                    "response": response_text,
                    "main_product": None,
                    "recommendations": [],
                    "debug": {"type": typ, "intent": intent}
                }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, [prod] if prod else [])
            
            return JSONResponse(response_data)
        
        elif typ == "compare":
            response_text = payload.get("response", "")
            response_data = {
                "response": response_text,
                "comparison": {
                    "product_a": payload.get("product_a"),
                    "product_b": payload.get("product_b"),
                    "cheaper": payload.get("cheaper")
                },
                "debug": {"type": "compare", "intent": intent}
            }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, None)
            
            return JSONResponse(response_data)
        
        elif typ in ("list_all", "list_categories", "price_range", "recommend"):
            prods = payload.get("products", [])
            response_text = payload.get("response", "")
            
            # For product-related queries, return first product with recommendations
            if prods and len(prods) > 0 and typ != "list_categories":
                main_product = prods[0]
                main_meta = {
                    "title": main_product.get("title") or main_product.get("name"),
                    "category_1": main_product.get("category") or "",
                    "category_2": "",
                    "category_3": ""
                }
                recommendations = get_recommendations(main_meta, k=3)
                response_data = {
                    "response": response_text,
                    "main_product": main_product,
                    "recommendations": recommendations,
                    "debug": {"type": typ, "intent": intent}
                }
            else:
                response_data = {
                    "response": response_text,
                    "main_product": None,
                    "recommendations": [],
                    "debug": {"type": typ, "intent": intent}
                }
            
            # Save to chat history if user is logged in
            if user_id:
                save_chat_history(user_id, query, response_text, prods[:1] if prods else [])
            
            return JSONResponse(response_data)
        
        else:
            response_text = payload.get("response", "No matching products found.")
            response_data = {
                "response": response_text,
                "main_product": None,
                "recommendations": [],
                "debug": {"type": typ, "intent": intent}
            }
            
            
            if user_id:
                print(f"chat  histroy stored")
                save_chat_history(user_id, query, response_text, [])
            
            return JSONResponse(response_data)

    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        logging.error(f"Raw output: {llm_output if 'llm_output' in locals() else 'N/A'}")
        response_text = "I had trouble processing that request. Could you rephrase it?"
        response_data = {
            "response": response_text,
            "products": []
        }
        
        # Save to chat history if user is logged in
        user_id = request.user_id if hasattr(request, 'user_id') else None
        query = request.query.strip() if hasattr(request, 'query') else ""
        if user_id and query:
            save_chat_history(user_id, query, response_text, [])
        
        return JSONResponse(response_data)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint - redirects to login page"""
    path = os.path.join(FRONTEND_DIR, "login.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>Backend Running - Please login at /login</h1>")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    """UI endpoint - serves the chatbot page"""
    path = os.path.join(FRONTEND_DIR, "main.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("UI not found", status_code=404)

@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page():
    """Chatbot page endpoint"""
    path = os.path.join(FRONTEND_DIR, "main.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("Chatbot page not found", status_code=404)

@app.get("/login", response_class=HTMLResponse)
def login_page():
    path = os.path.join(FRONTEND_DIR, "login.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("Login page not found", status_code=404)

@app.get("/register", response_class=HTMLResponse)
def register_page():
    path = os.path.join(FRONTEND_DIR, "register.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("Register page not found", status_code=404)

@app.get("/images/{filename}")
def get_image(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(IMAGES_DIR, safe_name)
    logging.info(f"fetching image path:{path}")
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

@app.post("/register")
def register(request: RegisterRequest):
    """Register a new user"""
    try:
 
        if not request.email or not request.username or not request.password:
            raise HTTPException(400, "Email, username, and password are required")
        
        
        existing_user = get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(400, "User with this email already exists")
        
        # Create user
        user_id = create_user(request.email, request.username, request.password)
        if not user_id:
            raise HTTPException(500, "Failed to create user")
        
        return JSONResponse({
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id,
            "email": request.email,
            "username": request.username
        })
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/login")
def login(request: LoginRequest):
    """Login user"""
    try:
        # Validate input
        if not request.email or not request.password:
            raise HTTPException(400, "Email and password are required")
        
        # Get user by email
        user = get_user_by_email(request.email)
        if not user:
            raise HTTPException(401, "Invalid email or password")
        
        print(f"called verify_password function:{verify_password}")
        if not verify_password(request.password, user['password_hash']):
            raise HTTPException(401, "Invalid email or password")
        
     
        if request.username and request.username.lower() != user['username'].lower():
            raise HTTPException(401, "Username does not match")
        
        return JSONResponse({
            "success": True,
            "message": "Login successful",
            "user_id": user['id'],
            "email": user['email'],
            "username": user['username']
        })
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/logout")
def logout(request: LogoutRequest):
    """Logout user (mainly for client-side session management)"""
    try:
        # Verify user exists
        if not user_exists(request.user_id):
            raise HTTPException(404, "User not found")
        
        return JSONResponse({
            "success": True,
            "message": "Logout successful"
        })
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/chat_history/{user_id}")
def get_user_chat_history(user_id: int, limit: int = 50):
    """Get chat history for a specific user"""
    try:
        # Verify user exists
        if not user_exists(user_id):
            raise HTTPException(404, "User not found")
        
        # Get chat history
        history = get_chat_history(user_id, limit)
        
        return JSONResponse({
            "success": True,
            "user_id": user_id,
            "chat_history": history,
            "count": len(history)
        })
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if vectorstore and qa_chain else "degraded",
        "vectorstore": bool(vectorstore),
        "llm": bool(qa_chain)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)