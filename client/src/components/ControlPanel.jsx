/**
 * Control Panel component - Left sidebar
 * Contains: Add item, Search, Projection controls
 */
import React, { useState, useCallback } from 'react';

export function ControlPanel({
  onAddItem,
  onSearch,
  onProjectionChange,
  projectionMethod,
  rawDims,
  onRawDimsChange,
  loading,
  itemCount,
  error,
  serverStatus,
  serverWakeTime
}) {
  const [newItemText, setNewItemText] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [localDims, setLocalDims] = useState(rawDims);

  // Handle add item
  const handleAddItem = useCallback(async (e) => {
    e.preventDefault();
    if (!newItemText.trim() || loading.add) return;
    
    try {
      await onAddItem(newItemText);
      setNewItemText('');
    } catch (err) {
      // Error is handled by parent
    }
  }, [newItemText, loading.add, onAddItem]);

  // Handle search
  const handleSearch = useCallback(async (e) => {
    e.preventDefault();
    if (loading.search) return;
    await onSearch(searchQuery);
  }, [searchQuery, loading.search, onSearch]);

  // Handle projection change
  const handleProjectionChange = useCallback((method) => {
    if (method === 'raw') {
      onProjectionChange('raw', localDims);
    } else {
      onProjectionChange('pca');
    }
  }, [onProjectionChange, localDims]);

  // Handle dimension change
  const handleDimChange = useCallback((axis, value) => {
    const newDims = [...localDims];
    const idx = ['x', 'y', 'z'].indexOf(axis);
    newDims[idx] = Math.max(0, Math.min(127, parseInt(value) || 0));
    setLocalDims(newDims);
    onRawDimsChange(newDims);
    
    if (projectionMethod === 'raw') {
      onProjectionChange('raw', newDims);
    }
  }, [localDims, projectionMethod, onProjectionChange, onRawDimsChange]);

  return (
    <div className="panel left-panel">
      {/* Header */}
      <div className="panel-header">
        <div className="logo">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" 
                    stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <div className="panel-title">Vector DB</div>
            <div className="panel-subtitle">Visualizer</div>
          </div>
        </div>
      </div>

      <div className="panel-content">
        {/* Add Item Section */}
        <div className="control-section">
          <div className="section-label">Add Vector</div>
          <form onSubmit={handleAddItem}>
            <div className="input-group">
              <input
                type="text"
                className="text-input"
                placeholder="Enter word or phrase..."
                value={newItemText}
                onChange={(e) => setNewItemText(e.target.value)}
                maxLength={1000}
                disabled={loading.add}
              />
            </div>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={!newItemText.trim() || loading.add}
            >
              {loading.add ? (
                <>
                  <span className="loading-spinner"></span>
                  Embedding...
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                  </svg>
                  Add Vector
                </>
              )}
            </button>
          </form>
        </div>

        {/* Search Section */}
        <div className="control-section">
          <div className="section-label">Semantic Search</div>
          <form onSubmit={handleSearch}>
            <div className="input-group">
              <input
                type="text"
                className="text-input"
                placeholder="Search similar vectors..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                maxLength={1000}
                disabled={loading.search}
              />
            </div>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={loading.search}
            >
              {loading.search ? (
                <>
                  <span className="loading-spinner"></span>
                  Searching...
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                  </svg>
                  Search
                </>
              )}
            </button>
          </form>
        </div>

        {/* Projection Section */}
        <div className="control-section">
          <div className="section-label">Projection Method</div>
          <div className="projection-buttons">
            <button
              type="button"
              className={`btn btn-secondary ${projectionMethod === 'pca' ? 'active' : ''}`}
              onClick={() => handleProjectionChange('pca')}
              disabled={loading.projection}
            >
              PCA
            </button>
            <button
              type="button"
              className={`btn btn-secondary ${projectionMethod === 'raw' ? 'active' : ''}`}
              onClick={() => handleProjectionChange('raw')}
              disabled={loading.projection}
            >
              Raw Dims
            </button>
          </div>
          
          {/* Raw Dimensions Selector */}
          {projectionMethod === 'raw' && (
            <div className="dims-selector">
              <div className="dim-input">
                <label>X Dim</label>
                <input
                  type="number"
                  min="0"
                  max="127"
                  value={localDims[0]}
                  onChange={(e) => handleDimChange('x', e.target.value)}
                />
              </div>
              <div className="dim-input">
                <label>Y Dim</label>
                <input
                  type="number"
                  min="0"
                  max="127"
                  value={localDims[1]}
                  onChange={(e) => handleDimChange('y', e.target.value)}
                />
              </div>
              <div className="dim-input">
                <label>Z Dim</label>
                <input
                  type="number"
                  min="0"
                  max="127"
                  value={localDims[2]}
                  onChange={(e) => handleDimChange('z', e.target.value)}
                />
              </div>
            </div>
          )}
        </div>

        {/* Error display */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* Stats */}
        <div className="stats-bar">
          <div className="stat-item">
            <div className="stat-value">{itemCount}</div>
            <div className="stat-label">Vectors</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">128</div>
            <div className="stat-label">Dimensions</div>
          </div>
        </div>

        {/* Server Status - shows when warming up on Render free tier */}
        <div className={`server-status ${serverStatus}`}>
          <div className="status-indicator">
            {serverStatus === 'warming' && (
              <>
                <span className="loading-spinner"></span>
                <span>Waking up server...</span>
              </>
            )}
            {serverStatus === 'ready' && (
              <>
                <span className="status-dot ready"></span>
                <span>Server ready{serverWakeTime > 1000 ? ` (${(serverWakeTime / 1000).toFixed(1)}s)` : ''}</span>
              </>
            )}
            {serverStatus === 'cold' && (
              <>
                <span className="status-dot cold"></span>
                <span>Connecting...</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;
