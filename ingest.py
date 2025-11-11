
# E_commerce/ingest.py
import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CSV_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/data/product.csv"
DB_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/db/products.db"
VECTOR_PATH = "E_commerce/vector_store/index.faiss"
IMG_DIR = "/Users/swarajsolanke/Chatbot/E_commerce/data/images"

os.makedirs(VECTOR_PATH, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# CRITICAL: Extract ONLY filename
df["image_path"] = df["image_path"].apply(
    lambda x: os.path.basename(str(x).strip()) if pd.notna(x) else None
)

# Save to DB
from sqlalchemy import create_engine
engine = create_engine(f"sqlite:///{DB_PATH}")
df.to_sql("products", engine, if_exists="replace", index=False)



embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = (df["name"] + " " + df["review"].fillna("")).tolist()
metadatas = df.to_dict("records")

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = splitter.create_documents(texts, metadatas=metadatas)

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(VECTOR_PATH)

print("INGEST DONE")
print("Sample image_path →", metadatas[0]["image_path"])