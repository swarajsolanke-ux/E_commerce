# E_commerce/ingest.py
import pandas as pd
import os
import json
from sqlalchemy import create_engine
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CSV_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/data/product.csv"
DB_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/db/products.db"
VECTOR_PATH = "E_commerce/E_commerce/vector_store/index.faiss" 
USER_DATA = "E_commerce/data/user_data.json"
IMG_DIR = "E_commerce/data/images"


# os.makedirs("E_commerce/vector_store", exist_ok=True)
# os.makedirs("E_commerce/data", exist_ok=True)
# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

#path=E_commerce/E_commerce/E_commerce/vector_store/index.faiss. E_commerce/E_commerce/E_commerce/vector_store/index.faiss

df = pd.read_csv(CSV_PATH)
import re

def extract_filename(path):
    if pd.isna(path) or not path:
        return None
    path = str(path).strip()
    # Remove any prefix like "E_commerce/data/images/"
    path = re.sub(r'^.*[/\\]', '', path)  # Remove everything before last / or \
    path = re.sub(r'^.*data[/\\]images[/\\]', '', path, flags=re.IGNORECASE)
    return path if path else None

# APPLY CLEANING
df["image_path"] = df["image_path"].apply(extract_filename)

# DEBUG: SHOW WHAT'S BEING SAVED
print("\nIMAGE PATHS AFTER CLEANING:")
for i, row in df.head(5).iterrows():
    print(f"  {row['name']:40} → {row['image_path']}")
#df["image_path"] = df["image_path"].apply(lambda x: os.path.basename(str(x)) if pd.notna(x) else None)
print(df["image_path"])

engine = create_engine(f"sqlite:///{DB_PATH}")
df.to_sql("products", engine, if_exists="replace", index=False)

if not os.path.exists(USER_DATA):
    with open(USER_DATA, "w") as f:
        json.dump({
            "users": {"guest": {"wishlist": [], "cart": [], "favorites": [], "past_queries": []}},
            "orders": [],
            "next_order_id": 1000
        }, f, indent=2)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = (df["name"] + " " + df["category"] + " " + df["review"].fillna("")).tolist()
metadatas = df.to_dict("records")
print(f"meatdata sample:{metadatas[0]}")

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = splitter.create_documents(texts, metadatas=metadatas)

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(VECTOR_PATH)

print("INGESTION + USER SYSTEM READY")
print(f"Products: {len(df)}")
print(f"Sample image: {metadatas[0]['image_path']}")