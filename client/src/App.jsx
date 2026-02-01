/**
 * Vector DB Visualizer - Main App Component
 * 
 * A web application for visualizing vector embeddings stored in MongoDB Atlas.
 * Features 3D visualization, semantic search, and multiple projection modes.
 */
import React, { useEffect, useCallback, useMemo } from 'react';
import { Scene3D } from './components/Scene3D';
import { ControlPanel } from './components/ControlPanel';
import { DetailsPanel } from './components/DetailsPanel';
import { useVectorDB } from './hooks/useVectorDB';

function App() {
  const {
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
    serverStatus,
    serverWakeTime,
    
    // Actions
    fetchItems,
    fetchProjection,
    addItem,
    getItemDetails,
    search,
    clearSearch,
    setRawDims
  } = useVectorDB();

  // Initial data fetch
  useEffect(() => {
    fetchItems();
  }, []);

  // Handle point click - fetch full details
  const handlePointClick = useCallback((itemId) => {
    getItemDetails(itemId);
  }, [getItemDetails]);

  // Handle search result click
  const handleResultClick = useCallback((itemId) => {
    getItemDetails(itemId);
  }, [getItemDetails]);

  // Handle add item
  const handleAddItem = useCallback(async (text) => {
    return await addItem(text);
  }, [addItem]);

  // Handle search
  const handleSearch = useCallback(async (query) => {
    await search(query, 10);
  }, [search]);

  // Handle projection change
  const handleProjectionChange = useCallback((method, dims) => {
    fetchProjection(method, dims || rawDims);
  }, [fetchProjection, rawDims]);

  // Get highlighted IDs from search results
  const highlightedIds = useMemo(() => {
    return searchResults.map(r => r.id);
  }, [searchResults]);

  // Get selected item ID
  const selectedId = selectedItem?.id || null;

  return (
    <div className="app">
      {/* Left Panel - Controls */}
      <ControlPanel
        onAddItem={handleAddItem}
        onSearch={handleSearch}
        onProjectionChange={handleProjectionChange}
        projectionMethod={projectionMethod}
        rawDims={rawDims}
        onRawDimsChange={setRawDims}
        loading={loading}
        itemCount={items.length}
        error={error}
        serverStatus={serverStatus}
        serverWakeTime={serverWakeTime}
      />

      {/* Center - 3D Scene */}
      <Scene3D
        points={points}
        highlightedIds={highlightedIds}
        selectedId={selectedId}
        onPointClick={handlePointClick}
        projectionMethod={projectionMethod}
        itemCount={items.length}
      />

      {/* Right Panel - Details */}
      <DetailsPanel
        selectedItem={selectedItem}
        searchResults={searchResults}
        loading={loading}
        onResultClick={handleResultClick}
        onClearSearch={clearSearch}
        pcaInfo={pcaInfo}
        projectionMethod={projectionMethod}
        points={points}
      />
    </div>
  );
}

export default App;
