# import pandas as pd
# import os
# from sqlalchemy import create_engine, Column, Integer, String, Float
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_text_splitters import CharacterTextSplitter



# DB_PATH = '/Users/swarajsolanke/Chatbot/E_commerce/db/products.db'
# VECTOR_PATH = 'E_commerce/vector_store/index.faiss'
# DATA_PATH = '/Users/swarajsolanke/Chatbot/E_commerce/data/product.csv'
# IMAGES_DIR = 'E_commerce/data/images'


# os.makedirs('db', exist_ok=True)
# os.makedirs('vector_store', exist_ok=True)


# df = pd.read_csv(DATA_PATH)


# Base = declarative_base()
# class Product(Base):
#     __tablename__ = 'products'
#     product_id = Column(Integer, primary_key=True)
#     name = Column(String(255))
#     category = Column(String(100))
#     cost = Column(Float)
#     rating = Column(Float)
#     review = Column(String(1000))
#     image_path = Column(String(255))


# engine = create_engine(f'sqlite:///{DB_PATH}')
# Base.metadata.create_all(engine)
# df.to_sql('products', engine, if_exists='replace', index=False)

# print(f"Ingested {len(df)} products to SQLite.")


# embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# texts = (df['name'] + ' ' + df['review']).tolist()
# metadatas = df[['product_id', 'name', 'category', 'cost', 'rating', 'image_path']].to_dict('records')


# splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.create_documents(texts, metadatas=metadatas)

# vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore.save_local(VECTOR_PATH)

# print(f"FAISS index created with {len(texts)} embeddings.")



#added the reviews

# E_commerce/ingest.py
import pandas as pd
import os
from sqlalchemy import create_engine
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Paths
DATA_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/data/product.csv"
DB_PATH = "/Users/swarajsolanke/Chatbot/E_commerce/db/products.db"
VECTOR_PATH = "E_commerce/vector_store/index.faiss"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)

# Load CSV
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} products")


# Save to SQLite
engine = create_engine(f"sqlite:///{DB_PATH}")
df.to_sql("products", engine, if_exists="replace", index=False)
print("Saved to SQLite")

# EMBEDDINGS + METADATA
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Text to embed: name + review
texts = (df["name"] + " " + df["review"].fillna("")).tolist()

# METADATA: ALL COLUMNS including review!
metadatas = df[[
    "product_id", "name", "category", "cost", 
    "rating", "review", "image_path"
]].to_dict("records")

# Splitter (optional)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = splitter.create_documents(texts, metadatas=metadatas)


vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(VECTOR_PATH)
print("FAISS saved with reviews!")
print("Sample metadata:", metadatas[0])  