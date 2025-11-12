import re
import traceback
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException
from fastapi.responses import JSONResponse


vectorstore = globals().get("vectorstore", None)   # replace/ensure this points to your vector store instance
CATALOG = globals().get("CATALOG", None)           # optional: list of product metadata dicts (for exact/fuzzy fallbacks)

PRODUCT_KEYWORDS = {"product", "item", "price", "cost", "rating", "review", "reviews", "how much"}
GREETING_WORDS = {"hello", "hi", "hey", "welcome", "yo"}



def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_convert(value, type_func, default):
    try:
        return type_func(value) if value is not None else default
    except Exception:
        return default


def extract_tokens(s: str) -> set:
    return set(re.findall(r"\b\w+\b", (s or "").lower()))


def find_exact_in_catalog(query: str, catalog: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Case-insensitive normalized exact match on name within provided catalog."""
    if not catalog:
        return None
    target = normalize_text(query)
    for item in catalog:
        meta = item if isinstance(item, dict) else getattr(item, "metadata", {}) or {}
        if normalize_text(meta.get("name", "")) == target:
            return meta
    return None


def is_product_query(query: str) -> bool:
    query_lower = query.lower()
    print(f"query_lower:{query_lower}")
    query_tokens = extract_tokens(query_lower)
    has_keyword = bool(PRODUCT_KEYWORDS.intersection(query_tokens))
    product_patterns = [
        r'\b(price|cost|rating|review)\s+(of|for)\b',
        r'\b(show|find|tell|get|give)\s+.*\b(product|item)\b',
        r'\b(how much|what is)\s+.*\b(price|cost)\b',
        r'\b(what|which)\s+.*\b(category|product)\b',
    ]
    has_pattern = any(re.search(pattern, query_lower) for pattern in product_patterns)
    return has_keyword or has_pattern


def calculate_relevance_score(query: str, metadata: Dict[str, Any]) -> float:
    query_lower = query.lower()
    query_tokens = extract_tokens(query_lower)
    score = 0.0
    max_score = 0.0

    # +2 category match
    max_score += 2.0
    category = str(metadata.get('category', '')).lower()
    if category and category in query_lower:
        score += 2.0

    # +3 name token overlap (pro-rated)
    max_score += 3.0
    name = str(metadata.get('name', '')).lower()
    name_tokens = extract_tokens(name)
    common_tokens = query_tokens.intersection(name_tokens)
    if len(common_tokens) > 0 and name_tokens:
        score += 3.0 * (len(common_tokens) / max(len(name_tokens), 1))

    # +1 for explicit field keywords
    max_score += 1.0
    field_keywords = {'price', 'cost', 'rating', 'review', 'reviews'}
    if query_tokens.intersection(field_keywords):
        score += 1.0
        print(f"score:{score}")

    return score / max_score if max_score > 0 else 0.0


def is_relevant_result(query: str, doc_metadata: Dict[str, Any], vector_score: float = None) -> Tuple[bool, str]:
    """
    Returns (is_relevant, reason).
    - Keeps your original vector-score heuristics.
    - Relevance fallback if vector_score isn't decisive.
    """
    if not is_product_query(query):
        return False, "Query is not product-related"

    relevance = calculate_relevance_score(query, doc_metadata)

    if vector_score is not None:
        try:
            score_val = float(vector_score)
            # NOTE: these thresholds were in your original code. Keep them but tweak as needed.
            if score_val > 0.95:
                return False, f"Vector distance too high: {score_val:.3f}"
            if score_val < 0.4 and relevance > 0.3:
                return True, f"Strong match - vector: {score_val:.3f}, relevance: {relevance:.2f}"
        except Exception:
            pass

    if relevance > 0.4:
        return True, f"Relevance score: {relevance:.2f}"

    return False, f"Low relevance: {relevance:.2f}"



def extract_answer_from_query(query: str, product: Dict[str, Any]) -> str:
    query_lower = query.lower()
    # rating
    if any(word in query_lower for word in ['rating', 'rated', 'star', 'stars']):
        rating = product.get('rating', None)
        return f"{float(rating):.1f}" if rating else "No rating available"
    # price/cost
    if any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap', 'how much', "give me"]):
        cost = product.get('cost', None)
        return f"Rs{float(cost):.2f}" if cost else "Price not available"
    # review
    if any(word in query_lower for word in ['review', 'reviews', 'feedback', 'opinion']):
        review = product.get('review', '')
        return review if review else "No reviews available"
    # name
    if any(word in query_lower for word in ['name', 'called', 'title']):
        return product.get('name', 'Unknown product')
    # category
    if any(word in query_lower for word in ['category', 'type', 'kind']):
        return product.get('category', 'Unknown category')
    # default fallback
    return product.get('name', 'Product information available')



def _doc_metadata(doc) -> Dict[str, Any]:
    if doc is None:
        return {}
    if isinstance(doc, dict):
        return doc.get("metadata", doc)
    m = getattr(doc, "metadata", None)
    if isinstance(m, dict):
        return m
    try:
        return dict(doc.__dict__)
    except Exception:
        return {}


def get_recommendations(main: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """
    Return up to *k* products from the same category (excluding the main product).
    Strategy:
      1) Try vectorstore similarity search for the main product (preferred).
      2) If vectorstore unavailable or returns nothing, fallback to scanning CATALOG (if available)
         and returning top items from the same category (excluding main).
    """
    try:
        cat = normalize_text(main.get("category", "") or "")
        print(f"category to be find:{cat}")
        main_name = normalize_text(main.get("name", "") or "")
        if not main_name:
            return []

        recs: List[Dict[str, Any]] = []
        seen = {main_name}

     
        if vectorstore is not None:
            try:
                # ask for more so we can filter to same category and exclude main
                results = vectorstore.similarity_search_with_score(f"{main_name} {cat}".strip(), k=10)
                print(f"vector results: {results}")
            except Exception as e:
                print(f"[REC] vector search failed: {e}")
                results = []

            for doc, _score in results:
                m = dict(_doc_metadata(doc) or {})
                name = normalize_text(m.get("name", ""))
                prod_cat = normalize_text(m.get("category", ""))
                if not name or name in seen:
                    continue
                # require same category when category is known
                if cat and prod_cat != cat:
                    continue
                rec = {
                    "name": str(m.get("name", "Unknown")),
                    "cost": safe_convert(m.get("cost"), float, 0),
                    "rating": safe_convert(m.get("rating"), float, 0),
                    "review": str(m.get("review", "No review")),
                    "category": str(m.get("category", "Unknown")),
                    "image": str(m.get("image_path")) if m.get("image_path") else None
                }
                recs.append(rec)
                seen.add(name)
                if len(recs) >= k:
                    break

        
        if len(recs) < k and CATALOG:
            for item in CATALOG:
                m = item if isinstance(item, dict) else getattr(item, "metadata", {}) or {}
                name = normalize_text(m.get("name", ""))
                prod_cat = normalize_text(m.get("category", ""))
                if not name or name in seen:
                    continue
                if cat and prod_cat != cat:
                    continue
                recs.append({
                    "name": str(m.get("name", "Unknown")),
                    "cost": safe_convert(m.get("cost"), float, 0),
                    "rating": safe_convert(m.get("rating"), float, 0),
                    "review": str(m.get("review", "No review")),
                    "category": str(m.get("category", "Unknown")),
                    "image": str(m.get("image_path")) if m.get("image_path") else None
                })
                seen.add(name)
                if len(recs) >= k:
                    break

        return recs[:k]
    except Exception as e:
        print(f"[REC] error: {e}")
        return []


@app.post("/query")
def search_products(request: QueryRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        query_lower = query.lower()
        words = extract_tokens(query_lower)
        if words & GREETING_WORDS:
            print(f"GREETING DETECTED → {query!r}")
            return "welcome to ecommerce chatbot , how may i help you?"

        # 1) Ensure this is a product-oriented query
        if not is_product_query(query):
            print("Query is not product-related")
            return JSONResponse({
                "response": "sorry i couldn't provide you answer. Please ask a relevant product query (e.g., price, rating, review).",
                "products": [],
                "recommendations": [],
                "debug": {"relevance": False, "reason": "Not a product query"}
            })

        # 2) Try to find an exact product in CATALOG first (strongest evidence)
        exact_meta = find_exact_in_catalog(query, CATALOG)
        if exact_meta:
            print("Exact product found in CATALOG")
            product = {
                "name": str(exact_meta.get("name", "Unknown")),
                "cost": safe_convert(exact_meta.get("cost"), float, 0),
                "rating": safe_convert(exact_meta.get("rating"), float, 0),
                "review": str(exact_meta.get("review", "No review")),
                "category": str(exact_meta.get("category", "Unknown")),
                "image": str(exact_meta.get("image_path")) if exact_meta.get("image_path") else None
            }
            answer = extract_answer_from_query(query, product)
            recommendations = get_recommendations(product, k=3)
            return JSONResponse({
                "response": answer,
                "products": [product],
                "recommendations": recommendations,
                "debug": {"relevance": True, "reason": "Exact match in catalog"}
            })

        # 3) Ask vectorstore for the top match (if available)
        results_with_scores = []
        if vectorstore is not None:
            try:
                results_with_scores = vectorstore.similarity_search_with_score(query, k=1) or []
            except Exception as e:
                print(f"Vectorstore search error: {e}")
                results_with_scores = []

        # If no vectorstore results or empty, go to "not found" fallback and attempt recommendations
        if not results_with_scores:
            print("No vector store results (or vectorstore unavailable). Attempting recommendation fallbacks.")
            # Build a fake main meta from query to try recommendations via vectorstore (if available)
            fake_main = {"name": query, "category": ""}
            recs = get_recommendations(fake_main, k=3)
            # If still empty and we have a CATALOG, provide top items from same category if category present in query
            if not recs and CATALOG:
                # try to find products whose name or category tokens overlap with query tokens
                qtokens = extract_tokens(query)
                candidates = []
                for item in CATALOG:
                    m = item if isinstance(item, dict) else getattr(item, "metadata", {}) or {}
                    m_name = normalize_text(m.get("name", ""))
                    m_cat = normalize_text(m.get("category", ""))
                    if qtokens.intersection(extract_tokens(m_name)) or qtokens.intersection(extract_tokens(m_cat)):
                        candidates.append(m)
                # take up to k candidates
                for m in candidates[:3]:
                    recs.append({
                        "name": str(m.get("name", "Unknown")),
                        "cost": safe_convert(m.get("cost"), float, 0),
                        "rating": safe_convert(m.get("rating"), float, 0),
                        "review": str(m.get("review", "No review")),
                        "category": str(m.get("category", "Unknown")),
                        "image": str(m.get("image_path")) if m.get("image_path") else None
                    })
            # final user-facing "not available" message + suggestions (if any)
            if not recs:
                return JSONResponse({
                    "response": f"Sorry — '{query}' is not available in the database and we don't have similar products to suggest right now.",
                    "products": [],
                    "recommendations": [],
                    "debug": {"relevance": False, "reason": "No results & no recommendations"}
                })
            else:
                first_name = recs[0].get("name", "a similar product")
                return JSONResponse({
                    "response": f"Product '{query}' is not available in the database. You can try a similar product: {first_name}.",
                    "products": [],
                    "recommendations": recs,
                    "debug": {"relevance": False, "reason": "No exact/vector match — fallback recommendations provided"}
                })

        # 4) We have a top vectorstore result; check relevance as before
        top_doc, vector_score = results_with_scores[0]
        metadata = dict(_doc_metadata(top_doc) or {})
        print(f"Vector score: {vector_score}")
        print(f"Metadata: {metadata}")

        is_relevant, reason = is_relevant_result(query, metadata, vector_score)
        if not is_relevant:
            # Not relevant. But per your requirement: say product not available and recommend similar product(s) if present.
            print(f"Result not relevant: {reason}. Running fallback recommendations.")
            # Attempt recommendations based on the top_doc metadata (category/name) or on catalog
            main_meta = {
                "name": metadata.get("name", "") or query,
                "category": metadata.get("category", "") or ""
            }
            recs = get_recommendations(main_meta, k=3)
            if not recs:
                return JSONResponse({
                    "response": f"Sorry — '{query}' is not available in the database.",
                    "products": [],
                    "recommendations": [],
                    "debug": {"relevance": False, "reason": reason, "vector_score": float(vector_score)}
                })
            else:
                first_name = recs[0].get("name", "a similar product")
                return JSONResponse({
                    "response": f"Product '{query}' is not available in the database. You can try a similar product: {first_name}.",
                    "products": [],
                    "recommendations": recs,
                    "debug": {"relevance": False, "reason": reason, "vector_score": float(vector_score)}
                })

        # 5) If relevant -> prepare product and answer as before
        print(f"Result is relevant: {reason}")
        product = {
            "name": str(metadata.get("name", "Unknown")),
            "cost": safe_convert(metadata.get("cost"), float, 0),
            "rating": safe_convert(metadata.get("rating"), float, 0),
            "review": str(metadata.get("review", "No review")),
            "category": str(metadata.get("category", "Unknown")),
            "image": str(metadata.get("image_path")) if metadata.get("image_path") else None
        }

        answer = extract_answer_from_query(query, product)
        print(f"Answer: {answer}")
        print(f"Product: {product['name']}")

        recommendations = get_recommendations(product, k=3)

        return JSONResponse({
            "response": answer,
            "products": [product],
            "recommendations": recommendations,
            "debug": {
                "relevance": True,
                "reason": reason,
                "vector_score": float(vector_score)
            }
        })

    except Exception as e:
        print(f"\nError processing query:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
print()