
import json
import sqlite3
from typing import List, Dict, Any
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.base import BaseCallbackHandler
import uvicorn 
from pydantic import BaseModel
import traceback
import logging
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from database import (
    init_user_db, hash_password, verify_password, get_user_by_email,
    create_user, save_chat_history, get_chat_history, user_exists
)
import re



DB_PATH = "/Users/swarajsolanke/Chatbot/E_commerce_project/db/products_DB.db"
print(DB_PATH)
DB_PATH_order="/Users/swarajsolanke/Chatbot/E_commerce_project/db/users_chat.db"
print(DB_PATH_order)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")


init_user_db()



app = FastAPI(title="E-commerce Chatbot")


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

# Static & CORS
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#added origin as well 

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

#databse 
def get_chat_db():
    conn = sqlite3.connect(DB_PATH_order)
    conn.row_factory = sqlite3.Row
    # Create tables if not exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            products TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    return conn


#db for the order tables.
def get_db():
    conn=sqlite3.connect(DB_PATH_order)
    print(f"connection for the order table is created")
    conn.row_factory=sqlite3.Row
    return conn
print(get_db())


#this is for product
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    print(f"conn is sucessfully done:{conn}")
    conn.row_factory = sqlite3.Row
    return conn



def rows_to_dicts(rows) -> List[Dict[str, Any]]:
    print(f"rows created :{rows}")
    return [dict(row) for row in rows]





# @tool
# def search_products(user_query: str) -> str:
#     """
#     Search products in the SQLite database based on a natural language query.

#     Handles:
#     - 'list all products'
#     - 'cheapest product'
#     - 'products under X rupees'
#     - 'products in category Cricket'
#     - 'show me Cricket nets under 2000' etc.
#     -'show me most-expensive product'
#     -'show me affordable products'

#     Returns a JSON string with a 'results' list.
#     """
#     conn = get_db_connection()
#     cur = conn.cursor()
#     q = user_query.lower()
#     print(f"query asked by user:{q}")

#     # 1. List all products
#     if "list" in q and "product" in q:
#         cur.execute("""
#             SELECT id, title, category_1, category_2, category_3,
#                    selling_price, mrp, product_rating, image_path
#             FROM products
#             LIMIT 20;
#         """)
#         rows = cur.fetchall()
#         conn.close()
#         return json.dumps({"mode": "list_all", "results": rows_to_dicts(rows)}, default=str)

#     # 2. Cheapest product
#     if "cheapest" in q or "lowest price" in q:
#         cur.execute("""
#             SELECT id, title, category_1, category_2, category_3,
#                    selling_price, mrp, product_rating, image_path
#             FROM products
#             ORDER BY selling_price ASC
#             LIMIT 5;
#         """)
#         rows = cur.fetchall()
#         conn.close()
#         return json.dumps({"mode": "cheapest", "results": rows_to_dicts(rows)}, default=str)

#     range_match = re.search(r"between\s+(\d+)\s*(?:rs|rupees)?\s*and\s*(\d+)", q)
#     if range_match:
#         low = float(range_match.group(1))
#         high = float(range_match.group(2))
#         cur.execute("""
#             SELECT id, title, category_1, category_2, category_3,
#                    selling_price, mrp, product_rating, image_path
#             FROM products
#             WHERE selling_price BETWEEN ? AND ?
#             ORDER BY selling_price ASC
#             LIMIT 20;
#         """, (low, high))
#         rows = cur.fetchall()
#         conn.close()
#         return json.dumps({
#             "mode": "price_range",
#             "price_low": low,
#             "price_high": high,
#             "results": rows_to_dicts(rows)
#         }, default=str)

#     # 4. Category-based search (simple heuristic: look for known category words)
#     # You can expand this list or even fetch distinct categories from DB.
#     category_keywords = ["sports", "books", "cricket","Batminton", "cycling",]
#     matched_cat = None
#     for c in category_keywords:
#         if c in q:
#             matched_cat = c
#             break

#     if matched_cat:
#         cur.execute("""
#             SELECT id, title, category_1, category_2, category_3,
#                    selling_price, mrp, product_rating, image_path
#             FROM products
#             WHERE LOWER(category_1) LIKE ? OR LOWER(category_2) LIKE ? OR LOWER(category_3) LIKE ?
#             LIMIT 20;
#         """, (f"%{matched_cat}%", f"%{matched_cat}%", f"%{matched_cat}%"))
#         rows = cur.fetchall()
#         conn.close()
#         return json.dumps({
#             "mode": "category_search",
#             "category": matched_cat,
#             "results": rows_to_dicts(rows)
#         }, default=str)

#     # 5. Fallback: title keyword search
#     # Take 2–3 most important words from query
#     words = [w for w in q.split() if len(w) > 3]
#     if not words:
#         words = q.split()

#     like_pattern = "%" + "%".join(words) + "%"
#     cur.execute("""
#         SELECT id, title, category_1, category_2, category_3,
#                selling_price, mrp, product_rating, image_path
#         FROM products
#         WHERE LOWER(title) LIKE ?
#         LIMIT 20;
#     """, (like_pattern,))
#     rows = cur.fetchall()
#     conn.close()
#     return json.dumps({
#         "mode": "title_search",
#         "query_like": like_pattern,
#         "results": rows_to_dicts(rows)
#     }, default=str)



@tool
def search_products(user_query: str) -> str:
    """
    Search products in the SQLite database based on a natural language query.
    And ALWAYS return:
       - product
       - 3 recommended products (same category_3)
       - include image_path
    """

 
    conn = get_db_connection()
    cur = conn.cursor()
    q = user_query.lower()

    print(f"query asked by user: {q}")

 
    def attach_recommendations(products):
        enriched = []
        for p in products:
            category = p.get("category_3")
            

            cur.execute("""
                SELECT  title, category_1, category_2, category_3,
                       selling_price, mrp, product_rating, image_path
                FROM products
                WHERE category_3 = ?
                ORDER BY product_rating DESC, selling_price ASC
                LIMIT 3;
            """, (category))

            recs = rows_to_dicts(cur.fetchall())

            enriched.append({
                "product": p,
                "recommendations": recs
            })
        return enriched

    
    if "list" in q and "product" in q:
        cur.execute("""
            SELECT title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            LIMIT 20;
        """)
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({"mode": "list_all", "results": result}, default=str)

    
    if "cheapest" in q or "lowest price" in q:
        cur.execute("""
            SELECT  title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            ORDER BY selling_price ASC
            LIMIT 5;
        """)
        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({"mode": "cheapest", "results": result}, default=str)

   
    range_match = re.search(r"between\s+(\d+)\s*(?:rs|rupees|₹)?\s*and\s*(\d+)", q)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))

        cur.execute("""
            SELECT  title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE selling_price BETWEEN ? AND ?
            ORDER BY selling_price ASC
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

    
    category_keywords = ["sports", "books", "cricket", "batminton", "cycling"]

    matched_cat = next((c for c in category_keywords if c.lower() in q), None)

    if matched_cat:
        cur.execute("""
            SELECT  title, category_1, category_2, category_3,
                   selling_price, mrp, product_rating, image_path
            FROM products
            WHERE LOWER(category_1) LIKE ?
               OR LOWER(category_2) LIKE ?
               OR LOWER(category_3) LIKE ?
            LIMIT 20;
        """, (f"%{matched_cat}%", f"%{matched_cat}%", f"%{matched_cat}%"))

        rows = rows_to_dicts(cur.fetchall())
        result = attach_recommendations(rows)
        conn.close()
        return json.dumps({
            "mode": "category_search",
            "category": matched_cat,
            "results": result
        }, default=str)

    
    words = [w for w in q.split() if len(w) > 3] or q.split()
    like_pattern = "%" + "%".join(words) + "%"

    cur.execute("""
        SELECT title, category_1, category_2, category_3,
               selling_price, mrp, product_rating, image_path
        FROM products
        WHERE LOWER(title) LIKE ?
        LIMIT 20;
    """, (like_pattern,))

    rows = rows_to_dicts(cur.fetchall())
    result = attach_recommendations(rows)
    conn.close()

    return json.dumps({
        "mode": "title_search",
        "query_like": like_pattern,
        "results": result
    }, default=str)


@tool
def track_orderid(order_id: int) -> str:
    """
    Fetch order status based on order_id.
    Returns:
        - orderid
        - product_name
        - status
        - order_date
    
    Useful when user asks: 
        'Track my order 1234', 
        'Where is my package?', 
        'What is the status of order id 567?'
    """

    try:
        conn = get_db()   # your SQLite connection function
        cur = conn.cursor()

        print(f"pointer i created:{cur}")

        cur.execute("""
            SELECT orderid, product_name,order_date,status
            FROM orders
            WHERE orderid = ?
        """, (order_id,))

        row = cur.fetchone()
        print(f"row is fetch from the db")

        if row is None:
            return json.dumps({
                "found": False,
                "error": f"No order found with order_id {order_id}"
            })

        result = {
            "found": True,
            "orderid": row[0],
            "product_name": row[1],
            "status": row[2],
            "order_date": row[3],
        
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "found": False,
            "error": str(e)
        })


@tool
def product_with_recommendations(product_identifier: str) -> str:
    """
    Get a specific product (by ID or title keywords) and 3 related recommendations.

    - First, tries to interpret product_identifier as integer ID.
    - If that fails, searches by title (LIKE).
    - Then finds 3 products in the same category_3, sorted by rating DESC and price ASC.
    Returns JSON with 'product' and 'recommendations'.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Try by ID
    product_row = None
    try:
        product_id = int(product_identifier)
        cur.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        product_row = cur.fetchone()
    except ValueError:
        # treat as title search
        like_pattern = f"%{product_identifier.lower()}%"
        cur.execute("""
            SELECT *
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
            "message": "No matching product found for the given identifier."
        })

    product = dict(product_row)
    category3 = product.get("category_3", "")

    # Fetch 3 recommendations in same category_3
    cur.execute("""
        SELECT *
        FROM products
        WHERE category_3 = ? OR id != ?
        ORDER BY product_rating DESC, selling_price ASC
        LIMIT 3;
    """, (category3, product["id"]))
    rec_rows = cur.fetchall()
    conn.close()

    return json.dumps({
        "found": True,
        "product": product,
        "recommendations": rows_to_dicts(rec_rows)
    }, default=str)



SYSTEM_PROMPT = """
You are an E-commerce AI Assistant for an online store.

Your capabilities:
1. Greet users politely and assist them naturally.
2. Answer any product-related queries:
   - List all products
   - Find products by category
   - Filter by price range
   - Show cheapest items
   - Search products using title keywords
3. When the user asks for details of a specific product, use the 
   `product_with_recommendations` tool to return the product +
   3 related recommended products.
4. When the user asks to track an order, check the order status using
   the `track_orderid` tool.

TOOL USAGE RULES:
- Use `search_products` for general product queries such as:
    * "list all products"
    * "show me the cheapest product"
    * "products under 1000"
    * "cricket nets"
    * "products between 300 and 600"
    * "show all categories"

- Use `product_with_recommendations` when:
    * The user mentions a specific product ID
    * The user mentions a specific product name/title
    * The user says: "show details of …" or "I want this product"

- Use `track_orderid` when:
    * User says "track my order"
    * User gives an order ID (like "track 1003", "order id 55", 
      "status of order 88", "where is my package?")

ORDER TABLE DETAILS:
The order table contains:
    - order_id
    - product_name
    - status
    - order_date
    - delivery_date
Use this information to help the user track their order.

RESPONSE STYLE:
- Always respond in clean, friendly, human-like wording.
- After calling a tool, convert its JSON output into a readable summary.
- Use bullet points when listing products.
- Include prices, ratings, and image links when available.
- Be concise but helpful.

If the user just greets you with "hi", "hello", "good morning", etc.,
reply with a warm greeting and ask how you can assist with shopping.

If the user's request cannot be solved using tools,
answer naturally without calling any tool.

Your goal: Provide the best possible shopping assistance experience.
"""




llm =ChatOllama(
    model="qwen3-nothink:latest",
    temperature=0.2
)
print(f"ollama model is called:{llm}")

class ToolTracker(BaseCallbackHandler):
    def __init__(self):
        self.outputs = []
        self.names = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.names.append(serialized.get("name"))

    def on_tool_end(self, output, **kwargs):
        self.outputs.append(output)






@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()
    user_id = data.get("user_id")
    if not user_query or not user_id:
        raise HTTPException(400, detail="Missing query or user_id")

    # Step 1: Bind tools to LLM (this injects tool schema into prompt)
    llm_with_tools = llm.bind_tools(
        tools=[search_products, product_with_recommendations, track_orderid],
        tool_choice="auto"
    )
    print(f"llm sucessfully called with the tool", llm_with_tools)

    # Step 2: Create a simple chain: user → LLM with tools
    chain = llm_with_tools

    # Step 3: Invoke and get response
    try:
        response = chain.invoke(user_query)
        print(f"response:{response}")
    except Exception as e:
        raise HTTPException(500, detail=f"LLM Error: {str(e)}")

    # Step 4: Check if tool was called
    tool_calls = getattr(response, "tool_calls", [])
    print(f"tool_calls is sucessfully done:{tool_calls}")
    response_text = response.content.strip() if hasattr(response, "content") else str(response)
    #print(f"response_text:{response_text}")

    main_product = None
    recommendations = []
    products = []
    list_format = False
    categories = []

    # Step 5: Execute tools if any were requested
    if tool_calls:
        for tool_call in tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]

            try:
                if name == "search_products":
                    raw_result = search_products.invoke(args["user_query"])
                    print(f"raw_result:{raw_result}")
                    result = json.loads(raw_result)
                    res_list = result.get("results", [])
                    products.extend(res_list)
                    if result.get("mode") == "list_all":
                        list_format = True
                        categories = [p.get("title", "Unknown") for p in res_list]

                elif name == "product_with_recommendations":
                    raw_result = product_with_recommendations(args["product_identifier"])
                    result = json.loads(raw_result)
                    if result.get("found"):
                        main_product = result["product"]
                        recommendations = result["recommendations"]

                elif name == "track_orderid":
                    raw_result = track_orderid(args["order_id"])
                    result = json.loads(raw_result)
                    if result.get("found"):
                        response_text = (
                            f"Your order #{result['orderid']} for {result['product_name']} "
                            f"is currently **{result['status']}**. Ordered on {result['order_date']}."
                        )
                    else:
                        response_text = result.get("error", "Order not found.")

            except Exception as e:
                response_text = f"Sorry, I had trouble processing that request: {str(e)}"

    # Step 6: Save to chat history
    conn = get_chat_db()
    cur = conn.cursor()
    products_json = None
    if main_product or products:
        all_prods = ([main_product] if main_product else []) + recommendations + products[:10]
        products_json = json.dumps(all_prods, default=str)

    cur.execute(
        "INSERT INTO chat_history (user_id, query, response, products) VALUES (?, ?, ?, ?)",
        (user_id, user_query, response_text, products_json)
    )
    conn.commit()
    conn.close()

    # Step 7: Return response to frontend
    resp = {"response": response_text or "How can I help you today?"}
    if main_product:
        resp["main_product"] = main_product
        resp["recommendations"] = recommendations
    elif products:
        resp["products"] = products
    if list_format:
        resp["list_format"] = True
        resp["categories"] = categories

    return resp



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
        
        
        existing_user =  get_user_by_email(request.email)
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
        user =  get_user_by_email(request.email)
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
            "count":len(history)
        })
    except HTTPException:
        raise
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("agent:app", host="0.0.0.0", port=5000, reload=True)
#192.168.5.255 192.168.5.146 

#settingup the endpoints



