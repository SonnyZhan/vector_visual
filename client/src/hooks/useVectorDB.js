/**
 * Custom hook for Vector DB API interactions
 */
import { useState, useCallback, useEffect, useRef } from 'react';

// API Base URL:
// - In development: Uses Vite proxy ('/api' → localhost:8000)
// - In production: Uses VITE_API_URL environment variable
// The || '/api' fallback ensures local dev works without .env file
const API_BASE = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

/**
 * Hook for managing Vector DB state and API calls
 */
export function useVectorDB() {
  const [items, setItems] = useState([]);
  const [points, setPoints] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState({
    items: false,
    add: false,
    search: false,
    projection: false,
    details: false
  });
  const [error, setError] = useState(null);
  const [projectionMethod, setProjectionMethod] = useState('pca');
  const [rawDims, setRawDims] = useState([0, 1, 2]);
  const [pcaInfo, setPcaInfo] = useState(null);
  
  // Server status for Render free tier cold starts
  // 'cold' = not pinged yet, 'warming' = ping in progress, 'ready' = server responding
  const [serverStatus, setServerStatus] = useState('cold');
  const [serverWakeTime, setServerWakeTime] = useState(null);
  const wakeStartTime = useRef(null);

  /**
   * Wake up the server by pinging the health endpoint
   * On Render free tier, the server sleeps after 15 min of inactivity.
   * This pre-warms it so API calls don't have a 30s delay.
   */
  const wakeUpServer = useCallback(async () => {
    if (serverStatus === 'ready') return; // Already awake
    
    setServerStatus('warming');
    wakeStartTime.current = Date.now();
    
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const elapsed = Date.now() - wakeStartTime.current;
        setServerWakeTime(elapsed);
        setServerStatus('ready');
        console.log(`✅ Server ready in ${elapsed}ms`);
      } else {
        throw new Error('Health check failed');
      }
    } catch (err) {
      console.error('Server wake-up failed:', err);
      // Retry after a short delay
      setTimeout(() => {
        setServerStatus('cold');
      }, 2000);
    }
  }, [serverStatus]);

  // Auto-wake server on mount
  useEffect(() => {
    wakeUpServer();
  }, []);

  /**
   * Fetch all items and compute projection
   */
  const fetchItems = useCallback(async () => {
    setLoading(prev => ({ ...prev, items: true }));
    setError(null);
    
    try {
      // Fetch items
      const itemsRes = await fetch(`${API_BASE}/items?lite=true`);
      if (!itemsRes.ok) throw new Error('Failed to fetch items');
      const itemsData = await itemsRes.json();
      setItems(itemsData.items || []);
      
      // Fetch projection
      await fetchProjection(projectionMethod, rawDims);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching items:', err);
    } finally {
      setLoading(prev => ({ ...prev, items: false }));
    }
  }, [projectionMethod, rawDims]);

  /**
   * Fetch projection from server
   */
  const fetchProjection = useCallback(async (method = 'pca', dims = [0, 1, 2]) => {
    setLoading(prev => ({ ...prev, projection: true }));
    
    try {
      const body = method === 'pca' 
        ? { method: 'pca' }
        : { method: 'raw', dims };
      
      const res = await fetch(`${API_BASE}/projection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      if (!res.ok) throw new Error('Failed to compute projection');
      
      const data = await res.json();
      setPoints(data.points || []);
      setProjectionMethod(method);
      if (method === 'pca' && data.pca_info) {
        setPcaInfo(data.pca_info);
      } else if (method === 'raw') {
        setRawDims(dims);
        setPcaInfo(null);
      }
    } catch (err) {
      setError(err.message);
      console.error('Error fetching projection:', err);
    } finally {
      setLoading(prev => ({ ...prev, projection: false }));
    }
  }, []);

  /**
   * Add a new item
   */
  const addItem = useCallback(async (text) => {
    if (!text.trim()) return;
    
    setLoading(prev => ({ ...prev, add: true }));
    setError(null);
    
    try {
      const res = await fetch(`${API_BASE}/items`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() })
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to add item');
      }
      
      const newItem = await res.json();
      
      // Refresh items and projection
      await fetchItems();
      
      return newItem;
    } catch (err) {
      setError(err.message);
      console.error('Error adding item:', err);
      throw err;
    } finally {
      setLoading(prev => ({ ...prev, add: false }));
    }
  }, [fetchItems]);

  /**
   * Get full item details including embedding
   */
  const getItemDetails = useCallback(async (itemId) => {
    setLoading(prev => ({ ...prev, details: true }));
    
    try {
      const res = await fetch(`${API_BASE}/items/${itemId}`);
      if (!res.ok) throw new Error('Failed to fetch item details');
      
      const item = await res.json();
      setSelectedItem(item);
      return item;
    } catch (err) {
      setError(err.message);
      console.error('Error fetching item details:', err);
    } finally {
      setLoading(prev => ({ ...prev, details: false }));
    }
  }, []);

  /**
   * Perform semantic search
   */
  const search = useCallback(async (query, k = 10) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    setLoading(prev => ({ ...prev, search: true }));
    setError(null);
    
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), k })
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Search failed');
      }
      
      const data = await res.json();
      setSearchResults(data.results || []);
      return data.results;
    } catch (err) {
      setError(err.message);
      console.error('Error searching:', err);
    } finally {
      setLoading(prev => ({ ...prev, search: false }));
    }
  }, []);

  /**
   * Clear search results
   */
  const clearSearch = useCallback(() => {
    setSearchResults([]);
  }, []);

  /**
   * Clear selected item
   */
  const clearSelection = useCallback(() => {
    setSelectedItem(null);
  }, []);

  return {
    // State
    items,
    points,
    selectedItem,
    searchResults,
    loading,
    error,
    projectionMethod,
    rawDims,
    pcaInfo,
    serverStatus,      // 'cold' | 'warming' | 'ready'
    serverWakeTime,    // milliseconds it took to wake up
    
    // Actions
    fetchItems,
    fetchProjection,
    addItem,
    getItemDetails,
    search,
    clearSearch,
    clearSelection,
    setRawDims,
    wakeUpServer
  };
}

export default useVectorDB;
