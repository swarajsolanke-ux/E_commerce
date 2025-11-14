# import os
# import re
# import traceback
# import json
# import logging
# from typing import Tuple, Dict, Any, List, Set, Optional
# from difflib import SequenceMatcher
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import torch
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import uvicorn
# try:
#     from langchain_classic.chains import RetrievalQA
#     from langchain_huggingface import HuggingFacePipeline
#     from langchain_core.prompts import PromptTemplate
#     from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
#     LLM_AVAILABLE = True
# except Exception:
#     LLM_AVAILABLE = False


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
# STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
# IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
# DB_PATH = os.path.join(BASE_DIR, "db", "products_DB.db")
# VECTOR_DIR = os.path.join(BASE_DIR, "vect", "vector_store")
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# model_path = "/Users/swarajsolanke/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

# K_SEMANTIC_CANDIDATES = 10
# TOP_RETURN = 1

# app = FastAPI(title="E-commerce Chatbot")

# # Static & CORS
# if os.path.isdir(STATIC_DIR):
#     app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# origins = ["*"]
# app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# device = "mps" if torch.backends.mps.is_available() else "cpu"


# print("Loading embeddings...")
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# vectorstore = None
# try:
#     print("Loading FAISS vectorstore...")
#     vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
#     print("Vector store loaded")
# except Exception as e:
#     print("Failed to load vectorstore:", e)


# LIST_ALL_PATTERNS = [
#     r"\b(all\s+products?|products?\s+available|what\s+products?\s+do\s+you\s+have)\b",
#     r"\bshow\s+me\s+.*\s+products?\b",
#     r"\bcan\s+you\s+give\s+me\s+all\s+products?\b",
# ]
# LIST_ALL_REGEXES = [re.compile(p, re.IGNORECASE) for p in LIST_ALL_PATTERNS]

# CATEGORY_PATTERNS = [
#     r"\b(show|list)\s+.*\s+(sports?|electronics?|clothing|home)\s+products?\b",
#     r"\b(sports?|electronics?|clothing|home)\s+products?\s+available\b",
# ]
# CATEGORY_REGEXES = [re.compile(p, re.IGNORECASE) for p in CATEGORY_PATTERNS]
# llm = None
# qa_chain = None
# if LLM_AVAILABLE and vectorstore:
#     try:
#         print("Loading LLM...")
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16 if device == "mps" else torch.float32,
#             device_map="auto" if device == "mps" else None
#         )
#         tokenizer.pad_token = tokenizer.eos_token

#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=512,
#             temperature=0.01,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             truncation=True,
#             return_full_text=False
#         )
#         llm = HuggingFacePipeline(pipeline=pipe)

      
#         PROMPT = PromptTemplate(
#             template="""
# You are an e-commerce assistant. Answer **only in valid JSON** using one of the schemas below.
# Use data from context only. Never add explanations.

# CONTEXT:
# {context}

# QUESTION: {question}

# --- SCHEMAS ---

# 1. LIST ALL
# {{
#   "type": "list_all",
#   "response": "Found X products.",
#   "products": [{{ "title": "...", "selling_price": "...", "product_rating": 0.0, "description": "...", "image": "..." }}, ...]  // up to 5
# }}

# 2. LIST CATEGORIES
# {{
#   "type": "list_categories",
#   "response": "Available categories: ...",
#   "categories": ["Electronics", "Clothing", ...]
# }}

# 3. SINGLE PRODUCT
# {{
#   "type": "single",
#   "response": "<title> – Rs<price> – <rating> stars",
#   "product": {{ "title": "...", "selling_price": "...", "product_rating": 0.0, "description": "...", "image": "..." }}
# }}

# 4. PRICE RANGE
# {{
#   "type": "price_range",
#   "response": "Found X products between Rs<min> and Rs<max>.",
#   "min": <float>, "max": <float>,
#   "products": [{{...}}, ...]  // up to 5
# }}

# 5. CHEAPEST
# {{
#   "type": "cheapest",
#   "response": "Cheapest: '<title>' for Rs<price>.",
#   "product": {{...}}
# }}

# 6. MOST EXPENSIVE
# {{
#   "type": "most_expensive",
#   "response": "Most expensive: '<title>' for Rs<price>.",
#   "product": {{...}}
# }}

# 7. HIGHEST RATING
# {{
#   "type": "highest_rating",
#   "response": "Highest rated: '<title>' (<rating> stars).",
#   "product": {{...}}
# }}

# 8. COMPARE TWO
# {{
#   "type": "compare",
#   "response": "Between '<A>' and '<B>', cheaper is '<title>'.",
#   "product_a": {{...}},
#   "product_b": {{...}},
#   "cheaper": {{...}}
# }}

# 9. RECOMMEND
# {{
#   "type": "recommend",
#   "response": "I recommend: ...",
#   "products": [{{...}}, ...] 
# }}

# 10. NOT FOUND
# {{
#   "type": "not_found",
#   "response": "No matching products found."
# }}

# --- INSTRUCTIONS ---
# - For "all products", "list all", "what do you have" → list_all
# - For "category" → list_categories
# - For exact name → single
# - For "under ₹X", "below", "between ₹X and ₹Y", "above" → price_range
# - For "cheapest", "most affordable" → cheapest
# - For "most expensive" → most_expensive
# - For "highest rating", "top rated" → highest_rating
# - For "compare A vs B", "A or B" → compare
# - For "recommend", "suggest", "gift", "under budget" → recommend
# - If price mentioned in recommend → price_range + recommend
# - Parse ₹, Rs, numbers correctly
# - Return **exact JSON**, no markdown


# ANSWER (JSON only):
# """,
#             input_variables=["context", "question"]
#         )

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),  # more context
#             chain_type_kwargs={"prompt": PROMPT},
#             return_source_documents=False
#         )
#         print("LLM loaded and ready")
#     except Exception as e:
#         print("Failed to load LLM:", e)
#         traceback.print_exc()
#         llm = None
#         qa_chain = None


# GREETING_WORDS: Set[str] = {
#     "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
#     "hii", "helo", "morning", "afternoon", "evening"
# }

# def _all_docs_from_vectorstore() -> List[Any]:
#     if not vectorstore:
#         return []
#     try:
#         return list(getattr(vectorstore.docstore, "_dict", {}).values())
#     except:
#         return []

# def doc_to_metadata(doc) -> Dict[str, Any]:
#     if doc is None:
#         return {}
#     if isinstance(doc, dict):
#         return doc.get("metadata", doc)
#     m = getattr(doc, "metadata", None)
#     if isinstance(m, dict):
#         return m
#     try:
#         return dict(doc.__dict__)
#     except:
#         return {}

# def parse_price(value: Any) -> float:
#     try:
#         if not value:
#             return 0.0
#         s = re.sub(r"[^\d.]", "", str(value))
#         return float(s) if s else 0.0
#     except:
#         return 0.0

# def parse_rating(value: Any) -> float:
#     try:
#         return float(str(value).strip())
#     except:
#         return 0.0

# def get_primary_category(metadata: Dict[str, Any]) -> str:
#     for k in ("category_2", "category_1", "category_3", "product_category"):
#         v = str(metadata.get(k, "")).strip()
#         if v and v.lower() not in ["nan", "none", ""]:
#             return v
#     return ""

# def doc_to_card(doc) -> dict:
#     meta = doc_to_metadata(doc)
#     image = None
#     if meta.get("image_path"):
#         image = f"/images/{os.path.basename(meta.get('image_path'))}"
#     elif meta.get("image_urls"):
#         urls = meta.get("image_urls")
#         if isinstance(urls, (list, tuple)) and urls:
#             image = urls[0]
#         elif isinstance(urls, str) and urls:
#             image = urls
#     return {
#         "title": meta.get("title") or meta.get("name") or "Unknown",
#         "selling_price": meta.get("selling_price") or meta.get("cost") or "",
#         "product_rating": parse_rating(meta.get("product_rating") or meta.get("rating")),
#         "description": (meta.get("description") or "")[:280],
#         "image": image
#     }

# def build_context() -> str:
#     docs = _all_docs_from_vectorstore()
#     lines = []
#     for doc in docs:
#         card = doc_to_card(doc)
#         cat = get_primary_category(doc_to_metadata(doc))
#         lines.append(
#             f"title:{card['title']} price:{card['selling_price']} rating:{card['product_rating']} category:{cat}"
#         )
#     return "\n".join(lines) if lines else "No products in database."


# class QueryRequest(BaseModel):
#     query: str


# @app.post("/query")
# def search_products(request: QueryRequest):
#     try:
#         query = request.query.strip()
#         if not query:
#             raise HTTPException(400, "Empty query")

      
#         if any(word in query.lower() for word in GREETING_WORDS):
#             return JSONResponse({
#                 "response": "Hello! How can I help you find a product today?"
#             })

       
#         q_lower = query.lower()
#         print(f"q_lower:{q_lower}")
#         print(f"q_lower:{q_lower}")

     
#         if any(rx.search(q_lower) for rx in LIST_ALL_REGEXES):
#             docs = _all_docs_from_vectorstore()
#             if not docs:
#                 return JSONResponse({"response": "No products in database.", "products": []})

#             context_lines = [
#                 f"title:{doc_to_card(d)['title']} price:{doc_to_card(d)['selling_price']} "
#                 f"rating:{doc_to_card(d)['product_rating']} category:{get_primary_category(doc_to_metadata(d))}"
#                 for d in docs
#             ]
#             context = "\n".join(context_lines)

          
#             filled = PROMPT.format(context=context, question="List all available products.")
#             raw = llm.invoke(filled)
#             llm_out = raw.strip()
#             print(f"llm_out:{llm_out}")

            
#             m = re.search(r"\{.*\}", llm_out, re.DOTALL)
#             payload = json.loads(m.group(0)) if m else {"type": "list_all", "response": "Here are all products.", "products": []}
#             prods = payload.get("products", [doc_to_card(d) for d in docs[:5]])
#             return JSONResponse({
#                 "response": payload.get("response", f"Found {len(docs)} products."),
#                 "products": prods[:5],
#                 "debug": {"type": "list_all", "total": len(docs)}
#             })

      
#         if any(rx.search(q_lower) for rx in CATEGORY_REGEXES):
           
#             cat_match = re.search(r"\b(sports?|electronics?|clothing|home)\b", q_lower)
#             target_cat = cat_match.group(1).lower() if cat_match else ""

#             docs = _all_docs_from_vectorstore()
#             filtered = [
#                 d for d in docs
#                 if target_cat in get_primary_category(doc_to_metadata(d)).lower()
#             ]

#             if not filtered:
#                 return JSONResponse({"response": f"No {target_cat} products found.", "products": []})

#             context_lines = [
#                 f"title:{doc_to_card(d)['title']} price:{doc_to_card(d)['selling_price']} "
#                 f"rating:{doc_to_card(d)['product_rating']} category:{get_primary_category(doc_to_metadata(d))}"
#                 for d in filtered
#             ]
#             context = "\n".join(context_lines)

#             filled = PROMPT.format(context=context, question=f"List {target_cat} products.")
#             raw = llm.invoke(filled)
#             llm_out = raw.strip()

#             m = re.search(r"\{.*\}", llm_out, re.DOTALL)
#             payload = json.loads(m.group(0)) if m else {"type": "list_categories", "products": []}
#             prods = payload.get("products", [doc_to_card(d) for d in filtered[:5]])
#             return JSONResponse({
#                 "response": payload.get("response", f"Found {len(filtered)} {target_cat} products."),
#                 "products": prods[:5],
#                 "debug": {"type": "category_list", "category": target_cat, "total": len(filtered)}
#             })

        
      
#         if not qa_chain:
#             raise HTTPException(500, "LLM not available")

#         result = qa_chain.invoke({"query": query})      
#         llm_output = result["result"].strip()

      
#         json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
#         if not json_match:
#             return JSONResponse({
#                 "response": "I couldn't understand that request.",
#                 "debug": {"raw": llm_output}
#             })

#         payload = json.loads(json_match.group(0))
#         typ = payload.get("type", "not_found")

       
#         if typ == "list_all":
#             prods = payload.get("products", [])[:5]
#             return JSONResponse({
#                 "response": payload.get("response", f"Found {len(prods)} products."),
#                 "products": prods,
#                 "debug": {"type": "list_all"}
#             })

#         elif typ == "list_categories":
#             return JSONResponse({
#                 "response": payload.get("response", "Available categories:"),
#                 "categories": payload.get("categories", []),
#                 "debug": {"type": "list_categories"}
#             })

#         elif typ == "single":
#             prod = payload.get("product")
#             return JSONResponse({
#                 "response": payload.get("response"),
#                 "products": [prod] if prod else [],
#                 "debug": {"type": "single"}
#             })

#         elif typ == "price_range":
#             prods = payload.get("products", [])[:5]
#             return JSONResponse({
#                 "response": payload.get("response"),
#                 "products": prods,
#                 "debug": {"type": "price_range", "min": payload.get("min"), "max": payload.get("max")}
#             })

#         elif typ in ("cheapest", "most_expensive", "highest_rating"):
#             return JSONResponse({
#                 "response": payload.get("response"),
#                 "product": payload.get("product"),
#                 "debug": {"type": typ}
#             })

#         elif typ == "compare":
#             return JSONResponse({
#                 "response": payload.get("response"),
#                 "comparison": {
#                     "product_a": payload.get("product_a"),
#                     "product_b": payload.get("product_b"),
#                     "cheaper": payload.get("cheaper")
#                 },
#                 "debug": {"type": "compare"}
#             })

#         elif typ == "recommend":
#             prods = payload.get("products", [])[:3]
#             return JSONResponse({
#                 "response": payload.get("response"),
#                 "recommendations": prods,
#                 "debug": {"type": "recommend"}
#             })

#         else:
#             return JSONResponse({
#                 "response": payload.get("response", "No matching products found."),
#                 "products": [],
#                 "debug": {"type": typ}
#             })

#     except Exception as e:
#         logging.error(traceback.format_exc())
#         raise HTTPException(500, f"Error: {str(e)}")

# @app.get("/", response_class=HTMLResponse)
# def root():
#     path = os.path.join(FRONTEND_DIR, "product2.html")
#     return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>Backend Running</h1>")

# @app.get("/ui", response_class=HTMLResponse)
# def ui():
#     path = os.path.join(FRONTEND_DIR, "product2.html")
#     return FileResponse(path) if os.path.exists(path) else HTMLResponse("UI not found", status_code=404)

# @app.get("/images/{filename}")
# def get_image(filename: str):
#     safe = os.path.basename(filename)
#     path = os.path.join(IMAGES_DIR, safe)
#     return FileResponse(path) if os.path.exists(path) else HTTPException(404, "Image not found")

# @app.get("/health")
# def health_check():
#     return {
#         "status": "healthy" if vectorstore and qa_chain else "degraded",
#         "vectorstore": bool(vectorstore),
#         "llm": bool(qa_chain)
#     }

# if __name__ == "__main__":
#     uvicorn.run("without_reg:app", host="0.0.0.0", port=5000, reload=True)
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
model_path = "/Users/swarajsolanke/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

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
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "hii", "helo", "morning", "afternoon", "evening", "sup", "yo", "howdy"
}

# Query intent patterns
INTENT_PATTERNS = {
    "list_all": [
        r"\b(all|every|entire|complete)\s+(products?|items?|things?)",
        r"\b(show|display|list|give)\s+me\s+(all|everything)",
        r"\bwhat\s+(products?|items?)\s+do\s+you\s+have",
        r"\bwhat'?s?\s+available",
        r"\bshow\s+catalog",
        r"\bview\s+all",
    ],
    "category": [
        r"\b(show|list|display|find|get)\s+.{0,20}?(sports?|electronics?|clothing|home|fashion|tech)",
        r"\b(sports?|electronics?|clothing|home|fashion|tech)\s+(products?|items?|section)",
        r"\bin\s+the\s+(sports?|electronics?|clothing|home)",
        r"\bcategory\s+of\s+",
    ],
    "price_range": [
        r"\b(under|below|less\s+than|cheaper\s+than|within)\s+.*?(\d+)",
        r"\b(above|over|more\s+than|expensive\s+than)\s+.*?(\d+)",
        r"\bbetween\s+.*?(\d+)\s+and\s+(\d+)",
        r"\b(₹|rs\.?|rupees?)\s*(\d+)",
        r"\b(\d+)\s*(₹|rs\.?|rupees?)",
        r"\bbudget\s+of\s+(\d+)",
        r"\bprice\s+range",
    ],
    "cheapest": [
        r"\b(cheapest|most\s+affordable|lowest\s+price|least\s+expensive|budget\s+friendly)",
        r"\bwhat'?s?\s+the\s+cheapest",
        r"\bminimum\s+price",
    ],
    "most_expensive": [
        r"\b(most\s+expensive|highest\s+price|premium|luxury|costliest)",
        r"\bwhat'?s?\s+the\s+most\s+expensive",
        r"\bmaximum\s+price",
    ],
    "highest_rating": [
        r"\b(highest|best|top)\s+(rated?|rating|reviewed?)",
        r"\bbest\s+quality",
        r"\bmost\s+popular",
        r"\btop\s+product",
    ],
    "compare": [
        r"\bcompare\s+.+\s+(and|vs|versus|with)",
        r"\b.+\s+(or|vs|versus)\s+.+\?",
        r"\bwhich\s+is\s+(better|cheaper|best)",
        r"\bdifference\s+between",
    ],
    "recommend": [
        r"\b(recommend|suggest|advice|help\s+me\s+choose)",
        r"\bwhat\s+should\s+i\s+(buy|get|purchase)",
        r"\bbest\s+for\s+",
        r"\bgift\s+for",
        r"\blooking\s+for",
        r"\bneed\s+a\s+",
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
    
    for intent, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
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
    
    # Under/below X
    under_match = re.search(r'(under|below|less\s+than|within)\s+.*?(\d+)', query_lower)
    if under_match:
        return 0, float(under_match.group(2))
    
    # Above/over X
    above_match = re.search(r'(above|over|more\s+than)\s+.*?(\d+)', query_lower)
    if above_match:
        return float(above_match.group(2)), float('inf')
    
    # Single price mention
    price_match = re.search(r'(\d+)', query_lower)
    if price_match:
        price = float(price_match.group(1))
        return 0, price
    
    return None, None

def extract_category(query: str) -> Optional[str]:
    """Extract category from query"""
    query_lower = query.lower()
    
    categories = {
        'electronics': ['electronic', 'electronics', 'tech', 'gadget', 'device'],
        'sports': ['sport', 'sports', 'fitness', 'gym', 'exercise'],
        'clothing': ['cloth', 'clothing', 'apparel', 'fashion', 'wear', 'shirt', 'pant'],
        'home': ['home', 'house', 'kitchen', 'furniture', 'appliance']
    }
    
    for cat, keywords in categories.items():
        if any(kw in query_lower for kw in keywords):
            return cat
    
    return None

def build_filtered_context(docs: List[Any], max_products: int = 50) -> str:
    """Build context string from documents"""
    lines = []
    for doc in docs[:max_products]:
        card = doc_to_card(doc)
        cat = get_primary_category(doc_to_metadata(doc))
        lines.append(
            f"title:{card['title']} price:{card['selling_price']} rating:{card['product_rating']} category:{cat}"
        )
    return "\n".join(lines) if lines else "No products available."


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(400, "Empty query")

        # Handle greetings
        if any(word in query.lower().split() for word in GREETING_WORDS):
            return JSONResponse({
                "response": "Hello! I'm here to help you find products. You can ask me about products, prices, categories, or get recommendations!"
            })

        # Detect intent
        intent = detect_intent(query)
        print(f"Query: {query}")
        print(f"Detected intent: {intent}")

        # Get all documents
        all_docs = _all_docs_from_vectorstore()
        
        if not all_docs:
            return JSONResponse({
                "response": "Sorry, no products are currently available in our database.",
                "products": []
            })

        # Handle based on intent
        if intent == "list_all":
            context = build_filtered_context(all_docs, max_products=50)
            filled = PROMPT.format(context=context, question="List all available products.")
            raw = llm.invoke(filled)
            llm_out = raw.strip()
            
            m = re.search(r"\{.*\}", llm_out, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                prods = payload.get("products", [doc_to_card(d) for d in all_docs[:5]])
            else:
                prods = [doc_to_card(d) for d in all_docs[:5]]
                payload = {"response": f"Found {len(all_docs)} products."}
            
            return JSONResponse({
                "response": payload.get("response", f"Found {len(all_docs)} products."),
                "products": prods[:5],
                "debug": {"intent": "list_all", "total": len(all_docs)}
            })

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
                    "products": []
                })
            
            context = build_filtered_context(relevant_docs)
            filled = PROMPT.format(context=context, question=query)
            raw = llm.invoke(filled)
            llm_out = raw.strip()
            
            m = re.search(r"\{.*\}", llm_out, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                prods = payload.get("products", [doc_to_card(d) for d in relevant_docs[:5]])
            else:
                prods = [doc_to_card(d) for d in relevant_docs[:5]]
                payload = {"response": f"Found {len(relevant_docs)} products."}
            
            return JSONResponse({
                "response": payload.get("response"),
                "products": prods[:5],
                "debug": {"intent": "category", "category": category}
            })

        elif intent == "price_range":
            min_price, max_price = extract_price_values(query)
            
            # Filter by price
            if min_price is not None or max_price is not None:
                filtered_docs = []
                for d in all_docs:
                    price = parse_price(doc_to_metadata(d).get("selling_price"))
                    if min_price is not None and max_price is not None:
                        if min_price <= price <= max_price:
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
                        "products": []
                    })
                
                context = build_filtered_context(filtered_docs[:20])
            else:
                # Use semantic search
                context = build_filtered_context(vectorstore.similarity_search(query, k=20))
            
            filled = PROMPT.format(context=context, question=query)
            raw = llm.invoke(filled)
            llm_out = raw.strip()
            
            m = re.search(r"\{.*\}", llm_out, re.DOTALL)
            if m:
                payload = json.loads(m.group(0))
                prods = payload.get("products", [doc_to_card(d) for d in filtered_docs[:5]])
            else:
                prods = [doc_to_card(d) for d in filtered_docs[:5]]
                payload = {"response": f"Found {len(filtered_docs)} products."}
            
            return JSONResponse({
                "response": payload.get("response"),
                "products": prods[:5],
                "debug": {"intent": "price_range", "min": min_price, "max": max_price}
            })

        elif intent == "cheapest":
            sorted_docs = sorted(all_docs, key=lambda d: parse_price(doc_to_metadata(d).get("selling_price")))
            cheapest = sorted_docs[0] if sorted_docs else None
            
            if not cheapest:
                return JSONResponse({"response": "No products available.", "products": []})
            
            card = doc_to_card(cheapest)
            return JSONResponse({
                "response": f"The cheapest product is '{card['title']}' for Rs{card['selling_price']}.",
                "product": card,
                "debug": {"intent": "cheapest"}
            })

        elif intent == "most_expensive":
            sorted_docs = sorted(all_docs, key=lambda d: parse_price(doc_to_metadata(d).get("selling_price")), reverse=True)
            most_exp = sorted_docs[0] if sorted_docs else None
            
            if not most_exp:
                return JSONResponse({"response": "No products available.", "products": []})
            
            card = doc_to_card(most_exp)
            return JSONResponse({
                "response": f"The most expensive product is '{card['title']}' for Rs{card['selling_price']}.",
                "product": card,
                "debug": {"intent": "most_expensive"}
            })

        elif intent == "highest_rating":
            sorted_docs = sorted(all_docs, key=lambda d: parse_rating(doc_to_metadata(d).get("product_rating")), reverse=True)
            highest = sorted_docs[0] if sorted_docs else None
            
            if not highest:
                return JSONResponse({"response": "No products available.", "products": []})
            
            card = doc_to_card(highest)
            return JSONResponse({
                "response": f"The highest rated product is '{card['title']}' with {card['product_rating']} stars.",
                "product": card,
                "debug": {"intent": "highest_rating"}
            })

        # For all other cases, use semantic search + LLM
        if not qa_chain:
            raise HTTPException(500, "LLM not available")

        # Use semantic search to get relevant products
        relevant_docs = vectorstore.similarity_search(query, k=30)
        context = build_filtered_context(relevant_docs)
        
        filled = PROMPT.format(context=context, question=query)
        raw = llm.invoke(filled)
        llm_output = raw.strip()
        
        print(f"LLM Output: {llm_output[:200]}")

        json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not json_match:
            # Fallback response
            return JSONResponse({
                "response": "Here are some products that might interest you:",
                "products": [doc_to_card(d) for d in relevant_docs[:3]],
                "debug": {"fallback": True, "intent": intent}
            })

        payload = json.loads(json_match.group(0))
        typ = payload.get("type", "not_found")

        # Handle different response types
        if typ in ("single", "cheapest", "most_expensive", "highest_rating"):
            prod = payload.get("product")
            return JSONResponse({
                "response": payload.get("response"),
                "products": [prod] if prod else [],
                "debug": {"type": typ, "intent": intent}
            })
        
        elif typ == "compare":
            return JSONResponse({
                "response": payload.get("response"),
                "comparison": {
                    "product_a": payload.get("product_a"),
                    "product_b": payload.get("product_b"),
                    "cheaper": payload.get("cheaper")
                },
                "debug": {"type": "compare", "intent": intent}
            })
        
        elif typ in ("list_all", "list_categories", "price_range", "recommend"):
            prods = payload.get("products", [])[:5]
            return JSONResponse({
                "response": payload.get("response"),
                "products": prods,
                "debug": {"type": typ, "intent": intent}
            })
        
        else:
            return JSONResponse({
                "response": payload.get("response", "No matching products found."),
                "products": [],
                "debug": {"type": typ, "intent": intent}
            })

    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        logging.error(f"Raw output: {llm_output if 'llm_output' in locals() else 'N/A'}")
        return JSONResponse({
            "response": "I had trouble processing that request. Could you rephrase it?",
            "products": []
        })
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(FRONTEND_DIR, "product2.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>Backend Running</h1>")

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
        "vectorstore": bool(vectorstore),
        "llm": bool(qa_chain)
    }

if __name__ == "__main__":
    uvicorn.run("without_reg:app", host="0.0.0.0", port=5000, reload=True)