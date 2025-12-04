import json
import sqlite3
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.base import BaseCallbackHandler
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import uvicorn
import logging
import traceback
from database import (
    init_user_db, hash_password, verify_password, get_user_by_email,
    create_user, save_chat_history, get_chat_history, user_exists
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database paths
DB_PATH = "/Users/swarajsolanke/Chatbot/E_commerce_project/db/products_DB.db"
DB_PATH_ORDER = "/Users/swarajsolanke/Chatbot/E_commerce_project/db/users_chat.db"

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")

# Initialize database
init_user_db()

# FastAPI app
app = FastAPI(title="E-commerce Chatbot")

# Pydantic models
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

class QueryRequest(BaseModel):
    query: str
    user_id: int

# Static files and CORS
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Database connection helpers
def get_chat_db():
    """Get database connection for chat history"""
    conn = sqlite3.connect(DB_PATH_ORDER)
    conn.row_factory = sqlite3.Row
    return conn

def get_db():
    """Get database connection for orders"""
    conn = sqlite3.connect(DB_PATH_ORDER)
    conn.row_factory = sqlite3.Row
    return conn

def get_db_connection():
    """Get database connection for products"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def rows_to_dicts(rows) -> List[Dict[str, Any]]:
    """Convert SQLite Row objects to dictionaries"""
    return [dict(row) for row in rows]

# LangChain Tools
@tool
def search_products(user_query: str) -> str:
    """
    Search products in the SQLite database based on a natural language query.
    Returns products with 3 recommended products from the same category.
    
    Handles queries like:
    - "list all products"
    - "show me the cheapest product"
    - "products in electronics category"
    - "cricket bats under 1000"
    - "products between 300 and 600"
    """
    if not isinstance(user_query,str):
        user_query=json.dumps(user_query)
    conn = get_db_connection()
    cur = conn.cursor()
    q = user_query.lower()
    
    logger.info(f"Searching products with query: {q}")

    def attach_recommendations(products):
        """Attach 3 recommended products for each product based on category"""
        enriched = []
        for p in products:
            category = p.get("category_3", "")
            product_title = p.get("title", "")
            
            # Get recommendations from same category, excluding current product
            if category:
                cur.execute("""
                    SELECT title, category_1, category_2, category_3,
                           selling_price, mrp, product_rating, image_path
                    FROM products
                    WHERE category_3 = ? AND title != ?
                    ORDER BY product_rating DESC, selling_price ASC
                    LIMIT 3;
                """, (category, product_title))
            else:
                # If no category, get top rated products
                cur.execute("""
                    SELECT title, category_1, category_2, category_3,
                           selling_price, mrp, product_rating, image_path
                    FROM products
                    WHERE title != ?
                    ORDER BY product_rating DESC
                    LIMIT 3;
                """, (product_title,))
            
            recs = rows_to_dicts(cur.fetchall())
            print(f"recs found:{recs}")
            enriched.append({
                "product": p,
                "recommendations": recs
            })
        return enriched

    # List all products
    if any(word in q for word in ["list", "show all", "display all", "all products"]):
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            ORDER BY product_rating DESC
            LIMIT 20;
        """)
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({"mode": "list_all", "results": result}, default=str)

    # Find cheapest products
    if any(word in q for word in ["cheapest", "lowest price", "most affordable", "cheap"]):
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE selling_price > 0
            ORDER BY selling_price ASC
            LIMIT 10;
        """)
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({"mode": "cheapest", "results": result}, default=str)

    # Find most expensive products
    if any(word in q for word in ["expensive", "highest price", "premium", "costly"]):
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            ORDER BY selling_price DESC
            LIMIT 10;
        """)
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({"mode": "expensive", "results": result}, default=str)

    # Price range search
    range_match = re.search(r"between\s+(\d+)\s*(?:rs|rupees|₹)?\s*and\s*(\d+)", q)
    under_match = re.search(r"under\s+(\d+)", q)
    above_match = re.search(r"(?:above|over)\s+(\d+)", q)
    
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE selling_price BETWEEN ? AND ?
            ORDER BY product_rating DESC
            LIMIT 20;
        """, (low, high))
        
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({
            "mode": "price_range",
            "price_low": low,
            "price_high": high,
            "results": result
        }, default=str)
    
    elif under_match:
        max_price = float(under_match.group(1))
        
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE selling_price <= ?
            ORDER BY product_rating DESC
            LIMIT 20;
        """, (max_price,))
        
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({
            "mode": "under_price",
            "max_price": max_price,
            "results": result
        }, default=str)
    
    elif above_match:
        min_price = float(above_match.group(1))
        
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE selling_price >= ?
            ORDER BY product_rating DESC
            LIMIT 20;
        """, (min_price,))
        
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({
            "mode": "above_price",
            "min_price": min_price,
            "results": result
        }, default=str)

    # Category search - expanded keywords
    category_keywords = {
        "electronics": ["electronics", "electronic", "gadget", "device"],
        "sports": ["sports", "sport", "athletic", "fitness"],
        "books": ["books", "book", "reading", "literature"],
        "cricket": ["cricket", "bat", "ball", "wicket"],
        "badminton": ["badminton", "shuttlecock", "racket"],
        "cycling": ["cycling", "bicycle", "bike", "cycle"],
        "football": ["football", "soccer"],
        "tennis": ["tennis", "racquet"],
        "clothing": ["clothing", "clothes", "apparel", "wear"],
        "shoes": ["shoes", "footwear", "sneakers"],
        "accessories": ["accessories", "accessory"]
    }
    
    matched_cat = None
    for main_cat, keywords in category_keywords.items():
        if any(keyword in q for keyword in keywords):
            matched_cat = main_cat
            break
    
    if matched_cat:
        # Search across all category columns
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE LOWER(category_1) LIKE ? 
               OR LOWER(category_2) LIKE ?
               OR LOWER(category_3) LIKE ?
               OR LOWER(title) LIKE ?
            ORDER BY product_rating DESC
            LIMIT 20;
        """, (f"%{matched_cat}%", f"%{matched_cat}%", f"%{matched_cat}%", f"%{matched_cat}%"))
        
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({
            "mode": "category_search",
            "category": matched_cat,
            "results": result
        }, default=str)

    # Title/keyword search (fallback)
    # Extract meaningful words (longer than 2 characters)
    words = [w for w in q.split() if len(w) > 2 and w not in ["the", "and", "for", "with"]]
    
    if words:
        # Create LIKE pattern for each word
        like_conditions = " OR ".join(["LOWER(title) LIKE ?" for _ in words])
        like_params = [f"%{word}%" for word in words]
        
        query = f"""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE {like_conditions}
            ORDER BY product_rating DESC
            LIMIT 20;
        """
        
        cur.execute(query, like_params)
        rows = rows_to_dicts(cur.fetchall())
        
        if rows:
            result = attach_recommendations(rows)
            conn.close()
            return json.dumps({
                "mode": "keyword_search",
                "keywords": words,
                "results": result
            }, default=str)
    
    # If no results found, return top rated products
    cur.execute("""
        SELECT title, category_1, category_2, category_3,
               selling_price, mrp, product_rating, image_path
        FROM products
        ORDER BY product_rating DESC
        LIMIT 10;
    """)
    rows = rows_to_dicts(cur.fetchall())
    result = attach_recommendations(rows)
    conn.close()
    
    return json.dumps({
        "mode": "default_results",
        "message": "Here are some top-rated products",
        "results": result
    }, default=str)

@tool
def track_orderid(order_id: int) -> str:
    """
    Fetch order status based on order_id.
    Returns order details including status and date.
    """
    if not isinstance(order_id,(str,int)):
        order_id=str(order_id)
    try:
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT orderid, product_name, orderdate, status
            FROM orders
            WHERE orderid = ?
        """, (order_id,))
        
        row = cur.fetchone()
        print(row)
        conn.close()
        
        if row is None:
            return json.dumps({
                "found": False,
                "error": f"No order found with order_id {order_id}"
            })
        
        result = {
            "found": True,
            "orderid": row["orderid"],
            "product_name": row["product_name"],
            "status": row["status"],
            "orderdate": row["orderdate"]
        }
        print(result)
        return json.dumps(result)
    
    except Exception as e:
        logger.error(f"Error tracking order: {str(e)}")
        return json.dumps({
            "found": False,
            "error": str(e)
        })

@tool
def product_with_recommendations(product_identifier: str) -> str:
    """
    Get a specific product by title with 3 related recommendations.
    
    Handles queries like:
    - "show me cricket bat"
    - "details of Nike shoes"
    - "tell me about this laptop"
    """
    if not isinstance(product_identifier,str):
        try:
            product_identifier=str(product_identifier)
        except:
            product_identifier = str(product_identifier)

    conn = get_db_connection()
    cur = conn.cursor()
    
    # Search by title (no ID column exists)
    like_pattern = f"%{product_identifier.lower()}%"
    cur.execute("""
        SELECT title, category_1, category_2, category_3,
               selling_price, mrp, product_rating, image_path
        FROM products
        WHERE LOWER(title) LIKE ?
        ORDER BY product_rating DESC
        LIMIT 1;
    """, (like_pattern,))
    
    product_row = cur.fetchone()
    
    if not product_row:
        conn.close()
        return json.dumps({
            "found": False,
            "message": "No matching product found."
        })
    
    product = dict(product_row)
    category3 = product.get("category_3", "")
    product_title = product.get("title", "")
    
    # Fetch recommendations from same category
    if category3:
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE category_3 = ? AND title != ?
            ORDER BY product_rating DESC, selling_price ASC
            LIMIT 3;
        """, (category3, product_title))
    else:
        # Fallback to top rated products
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE title != ?
            ORDER BY product_rating DESC
            LIMIT 3;
        """, (product_title,))
    
    rec_rows = cur.fetchall()
    conn.close()
    
    return json.dumps({
        "found": True,
        "product": product,
        "recommendations": rows_to_dicts(rec_rows)
    }, default=str)

# System prompt for the chatbot
SYSTEM_PROMPT = """
You are an E-commerce AI Assistant for an online store. You help customers find products and track orders.

Your capabilities:
1. **Greet users warmly** - Be friendly and professional
2. **Product Search**:
   - List all products
   - Find products by category (electronics, sports, books, cricket, badminton, cycling, etc.)
   - Filter by price (cheapest, under X, between X and Y, above X)
   - Search by product name or keywords
3. **Product Details** - Show specific product information with recommendations
4. **Order Tracking** - Check order status by order ID

TOOL USAGE RULES:

Use `search_products` for queries like:
- "list all products" / "show all products" / "what products do you have"
- "show me the cheapest product" / "most affordable items"
- "products in electronics category" / "sports items" / "cricket equipment"
- "products under 1000" / "items between 300 and 600"
- "expensive products" / "premium items"
- General search: "cricket bat" / "Nike shoes" / "laptop"

Use `product_with_recommendations` for specific product details:
- "show details of [product name]"
- "tell me more about [product]"
- "I want to see [specific product]"

Use `track_orderid` for order tracking:
- "track my order 1003"
- "where is my order?"
- "status of order 55"
- "track order id 123"

IMPORTANT RESPONSE GUIDELINES:
1. **Always be conversational** - Don't just list data, explain it naturally
2. **Format responses nicely**:
   - Use bullet points for multiple products
   - Include price, rating, and category
   - Mention if recommendations are included
3. **Handle empty results gracefully**:
   - If no products found, suggest alternatives
   - Guide users to refine their search
4. **For greetings** (hi, hello, good morning):
   - Respond warmly
   - Ask how you can help
   - Don't call any tools

5. **Convert tool outputs to natural language**:
   - Don't show raw JSON
   - Explain what you found
   - Highlight key features

Example responses:
- "I found 5 great cricket bats for you! Here are the details..."
- "The cheapest product is the XYZ at ₹299. Would you like to see more affordable options?"
- "Your order #1003 for 'Nike Cricket Bat' is currently in transit and should arrive by Dec 10th."

Remember: Your goal is to provide excellent shopping assistance and make users feel helped!
"""

# Initialize LLM
llm = ChatOllama(
    model="qwen3-nothink:latest",
    temperature=0.2
)

# Query endpoint
@app.post("/query")
async def query(request: Request):
    """Handle user queries with tool calling"""
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        print(f"user_query:{user_query}")
        user_id = data.get("user_id")
        
        if not user_query or not user_id:
            raise HTTPException(400, detail="Missing query or user_id")
        
        logger.info(f"Processing query: {user_query} for user: {user_id}")
    
    
        
        
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(
            tools=[search_products, product_with_recommendations, track_orderid],
            tool_choice="auto"
        )
        
        # Create messages with system prompt
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_query)
        ]
        
        # Invoke LLM
        response = llm_with_tools.invoke(messages)
        logger.info(f"LLM response received")
        print(f"response:{response}")

        # Extract response text
        response_text = response.content.strip() if hasattr(response, "content") else str(response)
        
        # Initialize response variables
        main_product = None
        recommendations = []
        products = []
        list_format = False
        categories = []
        
        # Check for tool calls
        tool_calls = getattr(response, "tool_calls", [])
        print(f"tool_calls:{tool_calls}")
        if tool_calls:
            logger.info(f"Tool calls detected: {len(tool_calls)}")
            
            for tool_call in tool_calls:
                print(f"tool_call:{tool_call}")
                name = tool_call["name"]
                print(name)
                args = tool_call["args"]
                print(args)
                
                logger.info(f"Executing tool: {name} with args: {args}")
               

                if name == "search_products":
                    uq=args["user_query"]
                    if isinstance(uq,list):
                        uq=" ".join(str(i) for i in uq)
                    elif isinstance(uq,dict):
                        uq=" ".join(str(v) for v in uq.values())
                    uq=str(uq)
                    raw_result = search_products.invoke(uq)
                    print(f"raw_result:{raw_result}")
                    result = json.loads(raw_result)
                    print(f"result of search products:{result}")
                    res_list = result.get("results", [])
                    products.extend(res_list)
                    
                    if result.get("mode") == "list_all":
                        list_format = True
                        categories = [p["product"].get("title", "Unknown") for p in res_list]
                    
                    # Save to chat history
                    save_chat_history(user_id, user_query, json.dumps(result))
                    
                    return JSONResponse({
                        "response": response_text,
                        "products": products,
                        "list_format": list_format,
                        "categories": categories
                    })
                
                elif name == "product_with_recommendations":
                    uq=args["user_query"]
                    if isinstance(uq,list):
                        uq=" ".join(str(i) for i in uq)
                    elif isinstance(uq,dict):
                        uq=" ".join(str(v) for v in uq.values())
                    uq=str(uq)
                   
                    raw_result = product_with_recommendations.invoke(uq)
                    result = json.loads(raw_result)
                    
                    if result.get("found"):
                        main_product = result["product"]
                        recommendations = result["recommendations"]
                    
                    # Save to chat history
                    save_chat_history(user_id, user_query, json.dumps(result))
                    
                    return JSONResponse({
                        "response": response_text,
                        "main_product": main_product,
                        "recommendations": recommendations
                    })
                
                elif name == "track_orderid":
                    raw_result = track_orderid.invoke({"order_id": args["order_id"]})
                    result = json.loads(raw_result)
                    
                    if result.get("found"):
                        response_text = (
                            f"Your order #{result['orderid']} for '{result['product_name']}' "
                            f"is currently {result['status']}. Ordered on {result['orderdate']}."
                        )
                    else:
                        response_text = result.get("error", "Order not found.")
                    
                    # Save to chat history
                    save_chat_history(user_id, user_query, json.dumps(result))
                    
                    return JSONResponse({
                        "response": response_text
                    })
        
        # No tool calls - just text response
        save_chat_history(user_id, user_query, response_text)
        
        return JSONResponse({
            "response": response_text or "How can I help you today?"
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, detail=f"Error: {str(e)}")

# Authentication endpoints
@app.post("/register")
def register(request: RegisterRequest):
    """Register a new user"""
    try:
        if not request.email or not request.username or not request.password:
            raise HTTPException(400, "Email, username, and password are required")
        
        existing_user = get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(400, "User with this email already exists")
        
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
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/login")
def login(request: LoginRequest):
    """Login user"""
    try:
        if not request.email or not request.password:
            raise HTTPException(400, "Email and password are required")
        
        user = get_user_by_email(request.email)
        if not user:
            raise HTTPException(401, "Invalid email or password")
        
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
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/logout")
def logout(request: LogoutRequest):
    """Logout user"""
    try:
        if not user_exists(request.user_id):
            raise HTTPException(404, "User not found")
        
        return JSONResponse({
            "success": True,
            "message": "Logout successful"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/chat_history/{user_id}")
def get_user_chat_history(user_id: int, limit: int = 50):
    """Get chat history for a user"""
    try:
        if not user_exists(user_id):
            raise HTTPException(404, "User not found")
        
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
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")

# Frontend endpoints
@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint"""
    path = os.path.join(FRONTEND_DIR, "login.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(
        "<h1>Backend Running - Please login at /login</h1>"
    )

@app.get("/ui", response_class=HTMLResponse)
def ui():
    """UI endpoint"""
    path = os.path.join(FRONTEND_DIR, "main.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(
        "UI not found", status_code=404
    )

@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page():
    """Chatbot page"""
    path = os.path.join(FRONTEND_DIR, "main.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(
        "Chatbot page not found", status_code=404
    )

@app.get("/login", response_class=HTMLResponse)
def login_page():
    """Login page"""
    path = os.path.join(FRONTEND_DIR, "login.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(
        "Login page not found", status_code=404
    )

@app.get("/register", response_class=HTMLResponse)
def register_page():
    """Register page"""
    path = os.path.join(FRONTEND_DIR, "register.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse(
        "Register page not found", status_code=404
    )

@app.get("/images/{filename}")
def get_image(filename: str):
    """Serve product images"""
    safe_name = os.path.basename(filename)
    path = os.path.join(IMAGES_DIR, safe_name)
    
    if os.path.exists(path):
        return FileResponse(path)
    
    raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    uvicorn.run("agent:app", host="0.0.0.0", port=5000, reload=True)