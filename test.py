from pydantic import BaseModel
import re
class QueryRequest(BaseModel):
    query: str

VALID_PRODUCT_FIELDS = {
    'product_id', 'name', 'category', 'cost', 'price', 'rating', 
    'review', 'reviews', 'image', 'image_path'
}

VALID_CATEGORIES = {
    'clothing', 'sports', 'electronics', 'home', 'garden', 'books'
}

PRODUCT_KEYWORDS = VALID_PRODUCT_FIELDS.union(VALID_CATEGORIES).union({
    'product', 'item', 'buy', 'purchase', 'show', 'tell', 'find',
    'what', 'which', 'how much', 'expensive', 'cheap'
})

def is_product_query(query: str) -> bool:
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    has_keyword = bool(PRODUCT_KEYWORDS.intersection(query_tokens))
    product_patterns = [
        r'\b(price|cost|rating|review)\s+(of|for)',
        r'\b(show|find|tell|get)\s+.*\b(product|item)',
        r'\b(how much|what is)\s+.*\b(price|cost)',
        r'\b(what|which)\s+.*\b(category|product)',
    ]
    has_pattern = any(re.search(pattern, query_lower) for pattern in product_patterns)
    return has_keyword or has_pattern

def calculate_relevance_score(query: str, metadata: Dict[str, Any]) -> float:
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
    field_keywords = {'price', 'cost', 'rating', 'review', 'reviews'}
    if query_tokens.intersection(field_keywords):
        score += 1.0
        print(f"score:{score}")
    return score / max_score if max_score > 0 else 0.0

def is_relevant_result(query: str, doc_metadata: Dict[str, Any], vector_score: float = None) -> Tuple[bool, str]:
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
    if any(word in query_lower for word in ['name', 'called', 'title']):
        return product.get('name', 'Unknown product')
    if any(word in query_lower for word in ['category', 'type', 'kind']):
        return product.get('category', 'Unknown category')
    return product.get('name', 'Product information available')
def safe_convert(value, type_func, default):
            try:
                return type_func(value) if value is not None else default
            except:
                return default

def get_recommendations(main: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """
    Return up to *k* products from the **same category** (excluding the main product)
    using vector similarity.
    """
    try:
        cat = main.get("category", "").lower()
        print(f"category to be find:{cat}")
        main_name = main.get("name", "").lower()
        if not cat or not main_name:
            return []

       
        results = vectorstore.similarity_search_with_score(
            f"{main_name} {cat}", k=10
        )

        recs = []
        seen = {main_name}
        for doc, _ in results:
            m = dict(getattr(doc, "metadata", {}) or {})
            name = str(m.get("name", "")).lower()
            prod_cat = str(m.get("category", "")).lower()

            if name == main_name or name in seen or prod_cat != cat:
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
        return recs
    except Exception as e:
        print(f"[REC] error: {e}")
        return []

