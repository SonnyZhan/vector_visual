# Vector DB Visualizer

A web application for visualizing high-dimensional vector embeddings in 3D space. Uses custom token-hash embeddings (128D), FastAPI for the backend, and React + Three.js for the frontend.

![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square)
![Frontend](https://img.shields.io/badge/Frontend-React%20+%20Three.js-61DAFB?style=flat-square)
![Embeddings](https://img.shields.io/badge/Embeddings-Token%20Hash%20128D-7b61ff?style=flat-square)

## Features

- **3D Visualization**: View vector embeddings projected into 3D space
- **Custom Vectorization**: Uses token-hash embeddings (128 dimensions)
- **PCA Projection**: Reduces 128D → 3D with explained variance info
- **Vector Lines**: Visual arrows from origin showing vector direction
- **Semantic Search**: Find similar vectors using cosine similarity
- **Real-time Updates**: Add new items and see them appear instantly
- **Full Vector Inspection**: View all 128 dimensions of any vector
- **Interactive Controls**: Orbit, pan, and zoom in 3D space

## Architecture

```
vector_db/
├── server/                 # FastAPI backend
│   ├── main.py            # API endpoints + in-memory storage
│   └── requirements.txt
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Scene3D.jsx       # 3D visualization
│   │   │   ├── ControlPanel.jsx  # Left panel controls
│   │   │   └── DetailsPanel.jsx  # Right panel details + PCA info
│   │   ├── hooks/
│   │   │   └── useVectorDB.js    # API hook
│   │   ├── App.jsx
│   │   └── App.css
│   ├── package.json
│   └── vite.config.js
├── vectorize.py           # Custom token-hash vectorization
├── render.yaml            # Render deployment config
└── README.md
```

## Quick Start (Local Development)

### Backend

```bash
cd server

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

Server runs at `http://localhost:8000`

### Frontend

```bash
cd client

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend runs at `http://localhost:5173`

## Deploy to Render (Free)

### Step 1: Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Create repo on GitHub, then push
git remote add origin https://github.com/YOUR_USERNAME/vector-db-visualizer.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and create both services:
   - `vector-db-api` (FastAPI backend)
   - `vector-db-frontend` (React static site)
5. Click **"Apply"**

### Step 3: Update URLs

After deployment, you'll get URLs like:
- Backend: `https://vector-db-api.onrender.com`
- Frontend: `https://vector-db-frontend.onrender.com`

**Update the URLs in `render.yaml`:**

```yaml
# In the backend service:
- key: FRONTEND_URL
  value: https://vector-db-frontend.onrender.com  # Your actual frontend URL

# In the frontend service:
- key: VITE_API_URL
  value: https://vector-db-api.onrender.com  # Your actual backend URL
```

Then push the changes:
```bash
git add render.yaml
git commit -m "Update production URLs"
git push
```

Render will automatically redeploy.

### Alternative: Deploy Services Manually

If you prefer not to use Blueprint:

**Deploy Backend:**
1. New → Web Service
2. Connect repo, set Root Directory: `server`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add env var: `FRONTEND_URL` = your frontend URL

**Deploy Frontend:**
1. New → Static Site
2. Connect repo, set Root Directory: `client`
3. Build Command: `npm install && npm run build`
4. Publish Directory: `dist`
5. Add env var: `VITE_API_URL` = your backend URL

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/items` | Add new item |
| `GET` | `/api/items` | List all items |
| `GET` | `/api/items/{id}` | Get item with full embedding |
| `POST` | `/api/search` | Semantic search |
| `POST` | `/api/projection` | Get 3D projections (PCA or raw) |
| `DELETE` | `/api/items` | Clear all items |

### Example: Add Item

```bash
curl -X POST http://localhost:8000/api/items \
  -H "Content-Type: application/json" \
  -d '{"text": "machine learning"}'
```

### Example: Search

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "k": 5}'
```

## How the Vectorization Works

The custom `vectorize.py` creates 128D embeddings by:

1. **Tokenize**: Split text into words
2. **Hash**: Each token → MD5 hash → seed for random number generator
3. **Generate**: Create deterministic 128D vector from seed
4. **Sum**: Add all token vectors together
5. **Normalize**: L2 normalize to unit length

This means:
- Same word always produces same vector
- Similar words (sharing tokens) have similar vectors
- No ML model needed - pure hashing!

## How PCA Projection Works

PCA (Principal Component Analysis) reduces 128D → 3D:

1. **Center**: Subtract mean from all vectors
2. **Covariance**: Compute how dimensions relate
3. **Eigenvectors**: Find directions of maximum variance
4. **Project**: Transform vectors onto top 3 eigenvectors

The frontend shows:
- Variance captured by each axis (PC1, PC2, PC3)
- Top contributing original dimensions
- Total information retained

## Important Notes

⚠️ **In-memory storage**: Data resets when server restarts! This is by design for simplicity.

⚠️ **Free tier limits**: Render free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds.

⚠️ **PCA needs 2+ items**: With only 1 item, PCA can't compute variance. Add at least 2 items for meaningful visualization.

## Tech Stack

- **Backend**: FastAPI, NumPy, scikit-learn (PCA)
- **Frontend**: React 18, Three.js, @react-three/fiber, @react-three/drei
- **Embeddings**: Custom token-hash (128 dimensions)
- **Hosting**: Render.com (free tier)

## License

MIT License - feel free to use and modify!
