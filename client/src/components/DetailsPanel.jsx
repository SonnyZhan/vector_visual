/**
 * Details Panel component - Right sidebar
 * Shows selected item details, search results, and PCA explanation
 */
import React, { useMemo } from 'react';

export function DetailsPanel({
  selectedItem,
  searchResults,
  loading,
  onResultClick,
  onClearSearch,
  pcaInfo,
  projectionMethod,
  points
}) {
  // Format date
  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Compute vector norm
  const computeNorm = (embedding) => {
    if (!embedding) return 0;
    const sumSquares = embedding.reduce((sum, val) => sum + val * val, 0);
    return Math.sqrt(sumSquares);
  };

  // Get the selected point's PCA raw values
  const selectedPointPca = useMemo(() => {
    if (!selectedItem || !points) return null;
    const point = points.find(p => p.id === selectedItem.id);
    return point?.pca_raw || null;
  }, [selectedItem, points]);

  return (
    <div className="panel right-panel">
      {/* Header */}
      <div className="panel-header">
        <div className="panel-title">Inspector</div>
        <div className="panel-subtitle">
          {selectedItem ? 'Vector Details' : 'Select a point'}
        </div>
      </div>

      <div className="panel-content">
        {/* Loading state */}
        {loading.details && (
          <div className="loading">
            <span className="loading-spinner"></span>
            Loading details...
          </div>
        )}

        {/* Selected Item Details */}
        {selectedItem && !loading.details && (
          <div className="selected-item">
            <div className="item-text">"{selectedItem.text}"</div>
            
            {/* Tokens display */}
            {selectedItem.tokens && (
              <div className="tokens-display">
                {selectedItem.tokens.map((token, idx) => (
                  <span key={idx} className="token-badge">{token}</span>
                ))}
              </div>
            )}
            
            {/* Metadata grid */}
            <div className="item-meta">
              <div className="meta-item">
                <div className="meta-label">Dimensions</div>
                <div className="meta-value">{selectedItem.dim || 128}</div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Norm</div>
                <div className="meta-value">
                  {selectedItem.stats?.norm?.toFixed(4) || computeNorm(selectedItem.embedding).toFixed(4)}
                </div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Mean</div>
                <div className="meta-value">
                  {selectedItem.stats?.mean?.toFixed(4) || '—'}
                </div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Std Dev</div>
                <div className="meta-value">
                  {selectedItem.stats?.std?.toFixed(4) || '—'}
                </div>
              </div>
              <div className="meta-item" style={{ gridColumn: 'span 2' }}>
                <div className="meta-label">Created</div>
                <div className="meta-value">{formatDate(selectedItem.createdAt)}</div>
              </div>
            </div>

            {/* PCA Explanation Section */}
            {projectionMethod === 'pca' && pcaInfo && (
              <div className="pca-section">
                <div className="section-header">
                  <span className="section-icon">📐</span>
                  <span className="section-title">PCA Projection</span>
                </div>
                
                <div className="pca-explanation">
                  <p className="explanation-text">
                    <strong>How 128D → 3D works:</strong> PCA (Principal Component Analysis) 
                    finds the directions of maximum variance in your data. It projects 
                    each 128-dimensional vector onto 3 new axes (PC1, PC2, PC3) that 
                    capture the most important patterns.
                  </p>
                  
                  <div className="pca-formula">
                    <div className="formula-title">The Math:</div>
                    <code className="formula">
                      xyz = V<sub>3×128</sub> · embedding<sub>128×1</sub>
                    </code>
                    <p className="formula-note">
                      Where V contains the top 3 eigenvectors of the covariance matrix
                    </p>
                  </div>
                </div>

                {/* Variance Explained */}
                <div className="variance-section">
                  <div className="variance-title">Variance Captured</div>
                  <div className="variance-bars">
                    {pcaInfo.explained_variance_ratio.map((ratio, idx) => (
                      <div key={idx} className="variance-bar-container">
                        <div className="variance-label">PC{idx + 1}</div>
                        <div className="variance-bar-bg">
                          <div 
                            className="variance-bar-fill"
                            style={{ 
                              width: `${ratio * 100}%`,
                              background: idx === 0 ? '#ff6b8a' : idx === 1 ? '#00ff9d' : '#00d9ff'
                            }}
                          />
                        </div>
                        <div className="variance-percent">{(ratio * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                  <div className="total-variance">
                    Total: <strong>{(pcaInfo.total_variance_explained * 100).toFixed(1)}%</strong> of original variance
                  </div>
                </div>

                {/* This Point's PCA Values */}
                {selectedPointPca && (
                  <div className="point-pca-values">
                    <div className="pca-values-title">This Vector's Position:</div>
                    <div className="pca-coords">
                      <div className="pca-coord">
                        <span className="coord-label" style={{ color: '#ff6b8a' }}>X (PC1):</span>
                        <span className="coord-value">{selectedPointPca[0]?.toFixed(4)}</span>
                      </div>
                      <div className="pca-coord">
                        <span className="coord-label" style={{ color: '#00ff9d' }}>Y (PC2):</span>
                        <span className="coord-value">{selectedPointPca[1]?.toFixed(4)}</span>
                      </div>
                      <div className="pca-coord">
                        <span className="coord-label" style={{ color: '#00d9ff' }}>Z (PC3):</span>
                        <span className="coord-value">{selectedPointPca[2]?.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Top Contributing Dimensions */}
                {pcaInfo.top_dims_per_pc && (
                  <div className="top-dims-section">
                    <div className="top-dims-title">Top Contributing Dimensions:</div>
                    {pcaInfo.top_dims_per_pc.map((pc, pcIdx) => (
                      <div key={pcIdx} className="pc-dims">
                        <span className="pc-label" style={{ 
                          color: pcIdx === 0 ? '#ff6b8a' : pcIdx === 1 ? '#00ff9d' : '#00d9ff' 
                        }}>
                          PC{pcIdx + 1}:
                        </span>
                        <span className="dim-list">
                          {pc.indices.slice(0, 3).map((dimIdx, i) => (
                            <span key={i} className="dim-badge">
                              d{dimIdx} ({pc.weights[i] > 0 ? '+' : ''}{pc.weights[i].toFixed(2)})
                            </span>
                          ))}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Raw Dims Explanation */}
            {projectionMethod === 'raw' && (
              <div className="pca-section">
                <div className="section-header">
                  <span className="section-icon">📊</span>
                  <span className="section-title">Raw Dimensions</span>
                </div>
                <p className="explanation-text">
                  Directly using 3 dimensions from the 128D vector as X, Y, Z coordinates.
                  No transformation applied — you're viewing the actual embedding values.
                </p>
              </div>
            )}

            {/* Vector Display */}
            {selectedItem.embedding && (
              <div className="vector-section">
                <div className="vector-header">
                  <span className="vector-title">Full 128D Embedding</span>
                  <span className="vector-dim">{selectedItem.embedding.length} dims</span>
                </div>
                <div className="vector-display">
                  {selectedItem.embedding.map((val, idx) => (
                    <span 
                      key={idx} 
                      className={`vector-value ${val >= 0 ? 'positive' : 'negative'}`}
                      title={`Dim ${idx}: ${val}`}
                    >
                      {val.toFixed(4)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Empty state when nothing selected */}
        {!selectedItem && !loading.details && searchResults.length === 0 && (
          <div className="empty-state">
            <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M12 16v-4M12 8h.01"></path>
            </svg>
            <div className="empty-title">No selection</div>
            <div className="empty-description">
              Click on a point in the 3D scene to view its vector details 
              and see how PCA projects it from 128D to 3D.
            </div>
          </div>
        )}

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="search-results">
            <div className="search-results-header">
              <span className="search-results-title">Search Results</span>
              <span className="result-count">{searchResults.length} matches</span>
            </div>
            
            <button 
              className="btn btn-secondary" 
              onClick={onClearSearch}
              style={{ marginBottom: '12px', width: '100%', padding: '8px' }}
            >
              Clear Results
            </button>
            
            {searchResults.map((result, idx) => (
              <div 
                key={result.id}
                className="result-item"
                onClick={() => onResultClick(result.id)}
              >
                <div className="result-rank">{idx + 1}</div>
                <div className="result-content">
                  <div className="result-text">{result.text}</div>
                  <div className="result-score">
                    Score: {(result.score * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Loading search */}
        {loading.search && (
          <div className="loading">
            <span className="loading-spinner"></span>
            Searching vectors...
          </div>
        )}
      </div>
    </div>
  );
}

export default DetailsPanel;
