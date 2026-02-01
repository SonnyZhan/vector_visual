"""
=============================================================================
VECTOR DB VISUALIZER - Backend Server
=============================================================================

This is a FastAPI server that:
1. Stores text items with their vector embeddings in memory
2. Provides REST API endpoints for the React frontend
3. Computes PCA projections to visualize 128D vectors in 3D

Key concepts used:
- FastAPI: Modern Python web framework for building APIs
- Pydantic: Data validation using Python type hints
- NumPy: Numerical computing (vectors, matrices)
- scikit-learn PCA: Dimensionality reduction algorithm

The embedding uses YOUR custom vectorize.py which:
- Tokenizes text into words
- Hashes each token to generate a deterministic random vector
- Sums token vectors and normalizes to create the final embedding
=============================================================================
"""

import sys
import os
from datetime import datetime, timezone
from typing import List, Optional
import numpy as np
from sklearn.decomposition import PCA

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# IMPORT YOUR CUSTOM VECTORIZATION
# =============================================================================
# Add parent directory to Python path so we can import vectorize.py
# vectorize.py is in the project root, but main.py is in /server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorize import text_to_vectors, tokenize


# =============================================================================
# PYDANTIC MODELS - Request/Response Validation
# =============================================================================
# Pydantic models define the shape of data we accept and return.
# FastAPI automatically validates incoming requests against these schemas.

class ItemCreate(BaseModel):
    """
    Schema for creating a new item.
    The 'text' field is required and must be 1-1000 characters.
    """
    text: str = Field(..., min_length=1, max_length=1000)
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Strip whitespace and ensure text isn't empty."""
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        return v


class SearchRequest(BaseModel):
    """
    Schema for semantic search requests.
    - query: The text to search for similar items
    - k: Number of results to return (default 10, max 100)
    """
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=100)


class ProjectionRequest(BaseModel):
    """
    Schema for requesting a 3D projection of all vectors.
    - method: "pca" for PCA projection, "raw" for raw dimension selection
    - dims: For "raw" method, which 3 dimensions to use as [x, y, z]
    """
    method: str = Field(default="pca")
    dims: Optional[List[int]] = Field(default=None)


# =============================================================================
# VECTOR STORE - In-Memory Storage
# =============================================================================
# This class holds all our data in memory. When the server restarts,
# all data is lost. For persistence, you'd use a database like MongoDB.

class VectorStore:
    """
    Simple in-memory storage for text items and their embeddings.
    
    Each item contains:
    - id: Unique identifier (auto-incremented)
    - text: The original text
    - embedding: 128-dimensional vector from vectorize.py
    - tokens: The words extracted from the text
    - createdAt: Timestamp of creation
    """
    
    def __init__(self):
        self.items = []          # List of all stored items
        self.next_id = 1         # Counter for generating unique IDs
        self.pca_cache = None    # Cached PCA results (for performance)
        self.pca_item_count = 0  # Track when cache needs refresh
    
    def add(self, text: str) -> dict:
        """
        Add a new item with its embedding.
        
        Steps:
        1. Generate 128D embedding using your vectorize.py
        2. Create item dict with metadata
        3. Add to storage
        4. Invalidate PCA cache (data changed!)
        """
        # Generate embedding using YOUR vectorize.py function
        # dim=128: Creates a 128-dimensional vector
        # normalize=True: Makes vector length = 1 (unit vector)
        embedding = text_to_vectors(text, dim=128, normalize=True)
        
        item = {
            "id": str(self.next_id),
            "text": text,
            "embedding": embedding.tolist(),  # Convert numpy array to list for JSON
            "tokens": tokenize(text),         # Show which words were used
            "dim": 128,
            "createdAt": datetime.now(timezone.utc).isoformat()
        }
        
        self.items.append(item)
        self.next_id += 1
        
        # Invalidate cache - new data means we need to recompute PCA
        self.pca_cache = None
        
        return item
    
    def get(self, item_id: str) -> Optional[dict]:
        """Get a single item by its ID."""
        for item in self.items:
            if item["id"] == item_id:
                return item
        return None
    
    def get_all(self) -> List[dict]:
        """Get all stored items."""
        return self.items
    
    def search(self, query: str, k: int = 10) -> List[dict]:
        """
        Find the k most similar items to the query using COSINE SIMILARITY.
        
        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (most similar)
        - 0.0 = perpendicular (unrelated)
        - -1.0 = opposite direction (least similar)
        
        Formula: cos(θ) = (A · B) / (|A| × |B|)
        
        Since our vectors are normalized (length = 1), this simplifies to:
        cos(θ) = A · B  (just the dot product!)
        """
        if not self.items:
            return []
        
        # Convert query text to 128D vector
        query_vec = text_to_vectors(query, dim=128, normalize=True)
        
        results = []
        for item in self.items:
            item_vec = np.array(item["embedding"])
            
            # COSINE SIMILARITY via dot product
            # Since both vectors are normalized, dot product = cosine similarity
            score = float(np.dot(query_vec, item_vec))
            
            results.append({
                "id": item["id"],
                "text": item["text"],
                "score": score  # 1.0 = identical, 0.0 = unrelated
            })
        
        # Sort by similarity score, highest first
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:k]  # Return top k results
    
    def project_pca(self) -> dict:
        """
        Project all 128D embeddings down to 3D using PCA.
        
        =================================================================
        WHAT IS PCA? (Principal Component Analysis)
        =================================================================
        
        Problem: We have 128-dimensional vectors, but can only visualize 3D.
        Solution: Find the "best" 3 dimensions that preserve the most information.
        
        PCA works by:
        1. Computing the COVARIANCE MATRIX of all vectors
           - Shows how each dimension relates to others
        
        2. Finding EIGENVECTORS of this matrix
           - These are the "principal components" (PCs)
           - PC1 = direction of maximum variance
           - PC2 = direction of 2nd most variance (perpendicular to PC1)
           - PC3 = direction of 3rd most variance (perpendicular to PC1 & PC2)
        
        3. PROJECT each 128D vector onto these 3 directions
           - Result: 3D coordinates [x, y, z]
        
        The math: new_coords = V @ original_vector
        Where V is a 3×128 matrix of the top 3 eigenvectors
        
        =================================================================
        """
        if not self.items:
            return {"points": [], "pca_info": None}
        
        # =============================================================
        # EDGE CASE: Single item
        # =============================================================
        # PCA needs at least 2 samples to compute variance.
        # With 1 sample, variance = (S^2) / (n-1) = division by zero → NaN
        # Solution: Place the single point at the origin with dummy PCA info
        if len(self.items) == 1:
            item = self.items[0]
            return {
                "points": [{
                    "id": item["id"],
                    "text": item["text"],
                    "xyz": [0.0, 0.0, 0.0],  # Origin
                    "pca_raw": [0.0, 0.0, 0.0]
                }],
                "pca_info": {
                    "n_components": 3,
                    "original_dims": 128,
                    "explained_variance_ratio": [0.0, 0.0, 0.0],
                    "total_variance_explained": 0.0,
                    "top_dims_per_pc": [],
                    "note": "PCA requires at least 2 items to compute variance"
                }
            }
        
        # Check if we can use cached results
        if self.pca_cache is not None and self.pca_item_count == len(self.items):
            return self.pca_cache
        
        # Stack all embeddings into a matrix: (num_items × 128)
        embeddings = np.array([item["embedding"] for item in self.items])
        
        # Determine number of components
        # Can't have more components than items or dimensions
        n_components = min(3, len(self.items), embeddings.shape[1])
        
        # =============================================================
        # FIT PCA
        # =============================================================
        # sklearn's PCA does all the heavy lifting:
        # 1. Centers the data (subtracts mean)
        # 2. Computes covariance matrix
        # 3. Finds eigenvectors/eigenvalues
        # 4. Selects top n_components eigenvectors
        pca = PCA(n_components=n_components)
        
        # fit_transform: Fit PCA to data AND transform in one step
        # Input: (num_items × 128) matrix
        # Output: (num_items × 3) matrix of projected coordinates
        projected = pca.fit_transform(embeddings)
        
        # Save raw values before normalization (for display)
        raw_projected = projected.copy()
        
        # =============================================================
        # NORMALIZE TO [-1, 1] RANGE
        # =============================================================
        # Scale coordinates to fit nicely in the 3D scene
        if len(projected) > 1:
            min_vals = projected.min(axis=0)  # Min for each dimension
            max_vals = projected.max(axis=0)  # Max for each dimension
            ranges = max_vals - min_vals
            ranges[ranges < 1e-10] = 1.0  # Avoid division by zero
            
            # Linear scaling: map [min, max] → [-1, 1]
            projected = 2 * (projected - min_vals) / ranges - 1
        
        # Pad with zeros if we have fewer than 3 components
        if n_components < 3:
            padding = np.zeros((projected.shape[0], 3 - n_components))
            projected = np.hstack([projected, padding])
            raw_projected = np.hstack([raw_projected, padding])
        
        # =============================================================
        # SANITIZE NaN/Inf VALUES
        # =============================================================
        # Edge case: If all vectors are identical, PCA produces NaN.
        # np.nan_to_num converts NaN→0 and Inf→large finite numbers
        projected = np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        raw_projected = np.nan_to_num(raw_projected, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Build list of points for frontend
        points = []
        for i, item in enumerate(self.items):
            points.append({
                "id": item["id"],
                "text": item["text"],
                "xyz": projected[i].tolist(),      # Normalized coords for 3D scene
                "pca_raw": raw_projected[i].tolist()  # Raw PCA values for display
            })
        
        # =============================================================
        # EXTRACT PCA STATISTICS
        # =============================================================
        
        # Explained variance ratio: How much of the original data's
        # variance is captured by each principal component
        # Example: [0.45, 0.25, 0.15] means PC1 captures 45%, PC2 25%, PC3 15%
        # Also sanitize these values in case of edge cases
        explained_variance_raw = pca.explained_variance_ratio_
        explained_variance_raw = np.nan_to_num(explained_variance_raw, nan=0.0, posinf=0.0, neginf=0.0)
        explained_variance = explained_variance_raw.tolist()
        total_variance = sum(explained_variance)
        
        # Find which original dimensions contribute most to each PC
        # pca.components_ is a (3 × 128) matrix where each row is a PC
        # The values show how much each original dimension contributes
        top_dims_per_pc = []
        for i in range(n_components):
            component = pca.components_[i]
            
            # Get indices of top 5 dimensions by absolute value
            top_indices = np.argsort(np.abs(component))[-5:][::-1].tolist()
            top_weights = [float(component[idx]) for idx in top_indices]
            
            top_dims_per_pc.append({
                "indices": top_indices,   # Which dimensions matter most
                "weights": top_weights    # How much they contribute (+/-)
            })
        
        pca_info = {
            "n_components": n_components,
            "original_dims": 128,
            "explained_variance_ratio": explained_variance,
            "total_variance_explained": total_variance,
            "top_dims_per_pc": top_dims_per_pc
        }
        
        result = {"points": points, "pca_info": pca_info}
        
        # Cache the results
        self.pca_cache = result
        self.pca_item_count = len(self.items)
        
        return result
    
    def project_raw(self, dims: List[int]) -> List[dict]:
        """
        Project using raw dimension indices.
        
        Instead of PCA, just pick 3 dimensions from the 128D vector
        and use them directly as [x, y, z].
        
        Example: dims=[0, 50, 100] means:
        - X = embedding[0]
        - Y = embedding[50]
        - Z = embedding[100]
        """
        if not self.items:
            return []
        
        # Extract the 3 chosen dimensions from each embedding
        all_coords = []
        for item in self.items:
            emb = item["embedding"]
            xyz = [
                emb[dims[0]] if dims[0] < len(emb) else 0,
                emb[dims[1]] if dims[1] < len(emb) else 0,
                emb[dims[2]] if dims[2] < len(emb) else 0
            ]
            all_coords.append(xyz)
        
        # Normalize to [-1, 1] for consistent visualization
        all_coords = np.array(all_coords)
        if len(all_coords) > 1:
            min_vals = all_coords.min(axis=0)
            max_vals = all_coords.max(axis=0)
            ranges = max_vals - min_vals
            ranges[ranges < 1e-10] = 1.0
            all_coords = 2 * (all_coords - min_vals) / ranges - 1
        
        points = []
        for i, item in enumerate(self.items):
            points.append({
                "id": item["id"],
                "text": item["text"],
                "xyz": all_coords[i].tolist()
            })
        
        return points


# =============================================================================
# CREATE GLOBAL STORE INSTANCE
# =============================================================================
# Single instance shared across all requests
store = VectorStore()


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="Vector DB Visualizer",
    description="Visualize text embeddings in 3D using custom token hashing",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------------
# Browsers block requests from one origin (e.g., localhost:5173)
# to another origin (e.g., localhost:8000) by default.
# CORS middleware tells the browser "it's okay, allow these origins".

# Get allowed origins from environment variable, or use defaults
# In production, set FRONTEND_URL to your deployed frontend URL
frontend_url = os.environ.get("FRONTEND_URL", "")
allowed_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",   # Alternative port
    "http://127.0.0.1:5173",
    "http://localhost:5174",   # Vite fallback ports
    "http://localhost:5175",
]

# Add production frontend URL if configured
if frontend_url:
    allowed_origins.append(frontend_url)
    # Also allow the www version if applicable
    if frontend_url.startswith("https://"):
        allowed_origins.append(frontend_url.replace("https://", "https://www."))

# For Render.com deployments - allow .onrender.com subdomains
allowed_origins.extend([
    "https://vector-db-frontend.onrender.com",
    "https://vector-db.onrender.com",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# =============================================================================
# API ENDPOINTS
# =============================================================================
# Each endpoint handles a specific type of request from the frontend.
# FastAPI automatically generates OpenAPI docs at /docs

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint - verify the server is running.
    Returns basic status info.
    """
    return {
        "status": "healthy",
        "storage": "in-memory",
        "embedding_method": "token-hash-128d",
        "item_count": len(store.items)
    }


@app.post("/api/items")
async def create_item(item: ItemCreate):
    """
    Create a new item with its embedding.
    
    Request body: { "text": "your text here" }
    
    Response: The created item (without full embedding for smaller payload)
    """
    try:
        new_item = store.add(item.text)
        
        # Return item info (excluding embedding to save bandwidth)
        return {
            "id": new_item["id"],
            "text": new_item["text"],
            "tokens": new_item["tokens"],
            "dim": new_item["dim"],
            "createdAt": new_item["createdAt"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/items")
async def list_items(lite: bool = Query(default=True)):
    """
    Get all stored items.
    
    Query params:
    - lite=true (default): Return only id and text (smaller response)
    - lite=false: Return full items including embeddings
    """
    items = store.get_all()
    
    if lite:
        # Lite mode: just IDs and text for listing
        return {
            "items": [{"id": i["id"], "text": i["text"]} for i in items],
            "count": len(items)
        }
    
    return {"items": items, "count": len(items)}


@app.get("/api/items/{item_id}")
async def get_item(item_id: str):
    """
    Get a single item by ID, including full embedding and stats.
    
    This is called when clicking on a point in the 3D scene.
    """
    item = store.get(item_id)
    
    if not item:
        raise HTTPException(status_code=404, detail=f"Item not found: {item_id}")
    
    # Compute statistics about the embedding vector
    emb = np.array(item["embedding"])
    stats = {
        "norm": float(np.linalg.norm(emb)),  # Vector length (should be ~1.0)
        "mean": float(np.mean(emb)),          # Average value
        "std": float(np.std(emb)),            # Standard deviation
        "min": float(np.min(emb)),            # Minimum value
        "max": float(np.max(emb))             # Maximum value
    }
    
    return {**item, "stats": stats}


@app.post("/api/search")
async def search(request: SearchRequest):
    """
    Search for similar items using cosine similarity.
    
    Request body: { "query": "search text", "k": 10 }
    
    Returns the top k most similar items with their similarity scores.
    """
    try:
        results = store.search(request.query, request.k)
        return {
            "query": request.query,
            "results": results,
            "k": request.k
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/projection")
async def projection(request: ProjectionRequest):
    """
    Compute 3D projection of all embeddings.
    
    Request body:
    - { "method": "pca" } for PCA projection
    - { "method": "raw", "dims": [0, 1, 2] } for raw dimension selection
    
    Returns:
    - points: Array of {id, text, xyz} for 3D rendering
    - pca_info: Statistics about the PCA (only for PCA method)
    """
    try:
        if request.method == "pca":
            result = store.project_pca()
            return {
                "method": "pca",
                "points": result["points"],
                "pca_info": result["pca_info"]
            }
        
        elif request.method == "raw":
            dims = request.dims or [0, 1, 2]
            
            # Validate dimension indices
            if len(dims) != 3:
                raise HTTPException(
                    status_code=400,
                    detail="Raw projection requires exactly 3 dims"
                )
            for d in dims:
                if d < 0 or d >= 128:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Dim {d} out of range [0, 127]"
                    )
            
            points = store.project_raw(dims)
            return {"method": "raw", "dims": dims, "points": points}
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown method: {request.method}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/items")
async def clear_items():
    """
    Clear all items (reset storage).
    Useful for starting fresh without restarting the server.
    """
    global store
    store = VectorStore()
    return {"message": "All items cleared", "count": 0}


# =============================================================================
# RUN SERVER
# =============================================================================
# This block runs when you execute: python main.py

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("VECTOR DB VISUALIZER")
    print("=" * 60)
    print("Using custom token-hash embeddings (128 dimensions)")
    print("In-memory storage - data resets on restart")
    print("")
    print("API Endpoints:")
    print("  POST /api/items      - Add new item")
    print("  GET  /api/items      - List all items")
    print("  GET  /api/items/{id} - Get item details")
    print("  POST /api/search     - Semantic search")
    print("  POST /api/projection - Get 3D projection")
    print("  GET  /api/health     - Health check")
    print("=" * 60)
    
    # Configure uvicorn to handle restarts better
    # reload=False prevents orphaned child processes
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False
    )
    server = uvicorn.Server(config)
    server.run()
