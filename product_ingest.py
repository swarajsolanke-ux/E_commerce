
# import os
# import re
# import sys
# import json
# import traceback
# import requests
# import pandas as pd
# from PIL import Image
# from sqlalchemy import create_engine
# from langchain_core.documents import Document
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS


# CSV_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/data/dataset.csv"
# DB_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/db/products_DB.db"
# VECTOR_DIR = "/Users/swarajsolanke/Chatbot/E_commerce/vect/vector_store/local"
# IMG_DIR = "/Users/swarajsolanke/Chatbot/E_commerce/data/images"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 0
# DOWNLOAD_IMAGES = False
# TIMEOUT = 10



# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
# os.makedirs(VECTOR_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)

# URL_RE = re.compile(r"https?://[^\s\"',]+", flags=re.IGNORECASE)

# def extract_urls(text):
#     if pd.isna(text) or text is None:
#         return []
#     matches = URL_RE.findall(str(text))
#     cleaned = [m.rstrip(').,;\'"') for m in matches]
#     return cleaned

# def safe_filename_from_url(url, prefix=None):
#     name = url.split("/")[-1].split("?")[0]
#     if not name:
#         name = "image"
#     name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
#     if prefix:
#         name = f"{prefix}_{name}"
#     return name

# def download_image(url, out_dir, prefix=None):
#     try:
#         resp = requests.get(url, stream=True, timeout=TIMEOUT)
#         resp.raise_for_status()
#         fname = safe_filename_from_url(url, prefix=prefix)
#         local_path = os.path.join(out_dir, fname)
#         with open(local_path, "wb") as f:
#             for chunk in resp.iter_content(chunk_size=8192):
#                 if chunk:
#                     f.write(chunk)
#         # verify image
#         try:
#             with Image.open(local_path) as img:
#                 img.verify()
#         except Exception:
#             try:
#                 os.remove(local_path)
#             except Exception:
#                 pass
#             return None
#         return local_path
#     except Exception:
#         return None

# def compose_text(row):
#     parts = []
#     for k in ["category_1","category_2","category_3","title","description"]:
#         v = (row.get(k) or "").strip()
#         if v:
#             parts.append(v)
#     return " \n ".join(parts)

# def main():
#     try:
#         print("Loading CSV:", CSV_PATH)
#         df = pd.read_csv(CSV_PATH, dtype=str)   # read strings to avoid mixed types
#         df.fillna("", inplace=True)

#         expected_cols = ["category_1","category_2","category_3","title","product_rating","selling_price","description","image_links"]
#         present = [c for c in expected_cols if c in df.columns]
#         if len(present) < len(expected_cols):
#             missing = set(expected_cols) - set(df.columns)
#             print("Warning: missing columns in CSV:", missing, file=sys.stderr)

#         # Extract URLs into list
#         df["image_urls"] = df["image_links"].apply(extract_urls)

#         # Download first valid image (optional)
#         local_paths = []
#         for idx, urls in enumerate(df["image_urls"].tolist()):
#             local_path = ""
#             if urls and DOWNLOAD_IMAGES:
#                 success = None
#                 for i, url in enumerate(urls):
#                     prefix = f"r{idx}"
#                     try:
#                         p = download_image(url, IMG_DIR, prefix=prefix)
#                         if p:
#                             success = p
#                             break
#                     except Exception:
#                         continue
#                 if success:
#                     local_path = os.path.abspath(success)
#             local_paths.append(local_path)
#         df["image_path"] = local_paths

#         # IMPORTANT: convert non-serializable columns (lists) to JSON strings before to_sql
#         df["image_urls_json"] = df["image_urls"].apply(lambda v: json.dumps(v) if isinstance(v, (list, tuple)) else json.dumps([]))

#         # Optionally keep original image_links as text, ensure image_path is string
#         df["image_path"] = df["image_path"].fillna("").astype(str)
#         df["image_links"] = df["image_links"].fillna("").astype(str)

#         # Save table to SQLite — now image_urls_json is plain text (JSON)
#         print("Saving table to SQLite:", DB_PATH)
#         engine = create_engine(f"sqlite:///{DB_PATH}")
#         df.to_sql("products", engine, if_exists="replace", index=False)
#         print("Saved to DB, rows:", len(df))

#         # Prepare embedding texts and metadata
#         texts = [compose_text(row) for _, row in df.iterrows()]

#         metadatas = []
#         for _, r in df.iterrows():
#             meta = {
#                 "category_1": r.get("category_1",""),
#                 "category_2": r.get("category_2",""),
#                 "category_3": r.get("category_3",""),
#                 "title": r.get("title",""),
#                 "product_rating": r.get("product_rating",""),
#                 "selling_price": r.get("selling_price",""),
#                 "description": r.get("description",""),
#                 "image_links": r.get("image_links",""),
#                 # store the JSON string and the list in metadata (list is fine for vectorstore metadata)
#                 "image_urls": json.loads(r.get("image_urls_json") or "[]"),
#                 "image_path": r.get("image_path","")
#             }
#             metadatas.append(meta)

#         # Create Documents
#         docs = [Document(page_content=t, metadata=m) for t,m in zip(texts, metadatas)]

#         # Split into chunks (metadata preserved)
#         splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#         split_docs = splitter.split_documents(docs)
#         print("Created document chunks:", len(split_docs))

#         # Embeddings
#         print("Loading embedding model:", EMBEDDING_MODEL)
#         embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

#         # Build FAISS
#         print("Building FAISS vectorstore (this may take a while)...")
#         vectorstore = FAISS.from_documents(split_docs, embeddings)

#         print("Saving FAISS to:", VECTOR_DIR)
#         vectorstore.save_local(VECTOR_DIR)
#         print("INGEST DONE")
#         print("Sample metadata ->", metadatas[0])

#         # Example retrieval (debug)
#         try:
#             print("\nExample retrieval for query: 'cricket bat'")
#             results = vectorstore.similarity_search_with_score("cricket bat", k=3)
#             for i, item in enumerate(results):
#                 if isinstance(item, tuple) and len(item) == 2:
#                     doc, score = item
#                 else:
#                     doc = item
#                     score = None
#                 meta = doc.metadata or {}
#                 print(f"#{i+1} score={score} title={meta.get('title')}")
#                 print("   image_path:", meta.get("image_path"))
#                 print("   image_urls (extracted):", meta.get("image_urls"))
#         except Exception as e:
#             print("Retrieval example failed:", e)
#             traceback.print_exc()

#     except Exception as e:
#         print("Fatal error in ingest:", e)
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

#8-12-25
# E_commerce/ingest.py
import os
import re
import sys
import json
import traceback
import pandas as pd
from sqlalchemy import create_engine
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging


CSV_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/data/dataset.csv"
DB_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/db/products_DB.db"
VECTOR_DIR = "/Users/swarajsolanke/Chatbot/E_commerce/vect/vector_store/local"
IMG_DIR = "/Users/swarajsolanke/Chatbot/E_commerce/data/images" 
order_path="/Users/swarajsolanke/Chatbot/E_commerce/data/orders_from_db.csv" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0



os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

URL_RE = re.compile(r"https?://[^\s\"',]+", flags=re.IGNORECASE)

def extract_urls(text):
    """Return list of urls found in text (empty list if none)."""
    if pd.isna(text) or text is None:
        return []
    matches = URL_RE.findall(str(text))
    cleaned = [m.rstrip(').,;\'"') for m in matches]
    return cleaned

def basename_from_url(url: str) -> str:
    """Return sanitized basename extracted from URL (no path/query)."""
    try:
        name = url.split("/")[-1].split("?")[0]
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip()
        return name
    except Exception:
        return ""

def compose_text(row):
    parts = []
    for k in ["category_1","category_2","category_3","title","description"]:
        v = (row.get(k) or "").strip()
        if v:
            parts.append(v)
    return " \n ".join(parts)

def main():
    try:
        print("Loading CSV:", CSV_PATH)
        df = pd.read_csv(CSV_PATH, dtype=str)   
        df.fillna("", inplace=True)

       
        expected_cols = ["category_1","category_2","category_3","title","product_rating","selling_price","description","image_links"]
        missing = set(expected_cols) - set(df.columns)
        if missing:
            print("Warning: missing expected columns:", missing, file=sys.stderr)

        # Extract URLs into a list column
        df["image_urls"] = df["image_links"].apply(extract_urls)

        # Compute image_path from first URL basename if file exists in IMG_DIR.
        # If not found, leave empty string. (No downloads.)
        local_paths = []
        for urls in df["image_urls"].tolist():
            local_path = ""
            if urls:
                b = basename_from_url(urls[0])
                if b:
                    candidate = os.path.join(IMG_DIR, b)
                    if os.path.exists(candidate):
                        local_path = os.path.abspath(candidate)
                    else:
                    
                        try:
                            for f in os.listdir(IMG_DIR):
                                if b in f:
                                    local_path = os.path.abspath(os.path.join(IMG_DIR, f))
                                    break
                        except Exception:
                            pass
            local_paths.append(local_path)
        df["image_path"] = local_paths

        # Convert list column -> JSON string column and then DROP the list column
        df["image_urls_json"] = df["image_urls"].apply(lambda v: json.dumps(v) if isinstance(v, (list, tuple)) else json.dumps([]))
        # Drop the Python-list column BEFORE writing to SQL!
        df = df.drop(columns=["image_urls"])

        # Ensure string columns are plain strings (no lists, dicts)
        df["image_path"] = df["image_path"].fillna("").astype(str)
        df["image_links"] = df["image_links"].fillna("").astype(str)
        df["image_urls_json"] = df["image_urls_json"].astype(str)

        # Save DataFrame to SQLite (now safe: no Python lists)
        print("Saving table to SQLite:", DB_PATH)
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df.to_sql("products", engine, if_exists="replace", index=False)
        print("Saved to DB, rows:", len(df))

       
        texts = []
        metadatas = []
        for _, r in df.iterrows():
            texts.append(compose_text(r))
            try:
                urls_list = json.loads(r.get("image_urls_json", "[]") or "[]")
            except Exception:
                urls_list = []
            meta = {
                "category_1": r.get("category_1",""),
                "category_2": r.get("category_2",""),
                "category_3": r.get("category_3",""),
                "title": r.get("title",""),
                "product_rating": r.get("product_rating",""),
                "selling_price": r.get("selling_price",""),
                "description": r.get("description",""),
                "image_links": r.get("image_links",""),
                "image_urls": urls_list,   # metadata may contain list (FAISS metadata stored separately)
                "image_path": r.get("image_path","")
            }
            metadatas.append(meta)

      
        docs = [Document(page_content=t, metadata=m) for t,m in zip(texts, metadatas)]
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = splitter.split_documents(docs)
        print("Created document chunks:", len(split_docs))

        # Embeddings + FAISS
        print("Loading embedding model:", EMBEDDING_MODEL)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        print("Building FAISS vectorstore...")
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        print("Saving FAISS to:", VECTOR_DIR)
        vectorstore.save_local(VECTOR_DIR)
        print("INGEST DONE")

        if metadatas:
            print("Sample metadata ->", metadatas[0])

        
        try:
            print("\nExample retrieval: query='cricket bat'")
            results = vectorstore.similarity_search_with_score("cricket bat", k=3)
            for i, item in enumerate(results):
                if isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                else:
                    doc = item
                    score = None
                meta = dict(getattr(doc, "metadata", {}) or {})
                print(f"#{i+1} score={score} title={meta.get('title')}")
                print("   image_path:", meta.get("image_path"))
                print("   image_urls:", meta.get("image_urls"))
        except Exception as e:
            print("Retrieval example failed:", e)
            traceback.print_exc()

    except Exception as e:
        print("Fatal error in ingest:", e)
        traceback.print_exc()
        sys.exit(1)
#added the embedding of the order data to product info data and store into the index.faiss and index.pkl file
def compose_order_text(row):
   
    parts = [
        f"Product: {row.get('product_name','')}",
        f"OrderID: {row.get('orderid','')}",
        f"Date: {row.get('orderdate','')}",
        f"Status: {row.get('status','')}",
    ]
    return " | ".join([p for p in parts if p and p.strip()])
def ingest_orders_csv_to_vectorstore(csv_path=order_path,
                                     vector_dir=VECTOR_DIR,
                                     embedding_model=EMBEDDING_MODEL,
                                     chunk_size=CHUNK_SIZE,
                                     chunk_overlap=CHUNK_OVERLAP):
    

 
    print("Loading orders CSV:", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    df.fillna("", inplace=True)

    
    before = len(df)
    df = df[(df.get("product_name", "") != "") & (df.get("orderid", "") != "")]
    after = len(df)
    if before != after:
        print(f"Skipped {before-after} rows with empty product_name or orderid.")

  
    texts = []
    metadatas = []
    for _, r in df.iterrows():
        text = compose_order_text(r)
        print(f"text is added:{text}")
        meta = {
            "orderid": r.get("orderid", ""),
            "product_name": r.get("product_name", ""),
            "orderdate": r.get("orderdate", ""),
            "status": r.get("status", "")
        }
        texts.append(text)
        metadatas.append(meta)
    logging.info("metadata data sucessfully")

   
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    print("Created document chunks from orders:", len(split_docs))

    
    print("Loading embedding model:", embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    
    try:
        index_file = os.path.join(vector_dir, "index.faiss")
        meta_file = os.path.join(vector_dir, "index.pkl")
        if os.path.exists(index_file) and os.path.exists(meta_file):
            print("Existing FAISS index found. Loading and updating it...")
            vectorstore = FAISS.load_local(vector_dir, embeddings,allow_dangerous_deserialization=True)
           
            vectorstore.add_documents(split_docs)
            print(f"vectorstore:{vectorstore}")
        else:
            print("No existing FAISS index found. Creating a new one...")
            vectorstore = FAISS.from_documents(split_docs, embeddings)

        
        print("Saving FAISS vectorstore to:", vector_dir)
        vectorstore.save_local(vector_dir)
        print("Ingest complete. Vectorstore updated.")
    except Exception as e:
        print("Error while ingesting to FAISS vectorstore:", e)
        traceback.print_exc()

ingest_orders_csv_to_vectorstore(csv_path=order_path,
                                     vector_dir=VECTOR_DIR,
                                     embedding_model=EMBEDDING_MODEL,
                                     chunk_size=CHUNK_SIZE,
                                     chunk_overlap=CHUNK_OVERLAP)
if __name__ == "__main__":
    main()
