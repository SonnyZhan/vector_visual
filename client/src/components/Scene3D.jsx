/**
 * 3D Scene component using React Three Fiber
 * Renders vector embeddings as points in 3D space with lines from origin
 */
import React, { useRef, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Html, Line } from '@react-three/drei';
import * as THREE from 'three';

// ============================================================================
// Vector Line Component - Line from origin to point with arrowhead
// ============================================================================

function VectorLine({ position, color, opacity = 0.6 }) {
  const points = useMemo(() => [
    [0, 0, 0],
    position
  ], [position]);

  // Calculate arrow direction and position
  const arrowProps = useMemo(() => {
    const vec = new THREE.Vector3(...position);
    const length = vec.length();
    if (length < 0.01) return null;
    
    // Position arrow at 90% along the line
    const arrowPos = vec.clone().multiplyScalar(0.92);
    
    // Calculate rotation to point arrow in the right direction
    const direction = vec.clone().normalize();
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    
    return {
      position: [arrowPos.x, arrowPos.y, arrowPos.z],
      quaternion,
      scale: Math.min(0.06, length * 0.08)
    };
  }, [position]);

  return (
    <group>
      {/* The line */}
      <Line
        points={points}
        color={color}
        lineWidth={1.5}
        opacity={opacity}
        transparent
      />
      
      {/* Arrowhead cone */}
      {arrowProps && (
        <mesh 
          position={arrowProps.position}
          quaternion={arrowProps.quaternion}
        >
          <coneGeometry args={[arrowProps.scale * 0.5, arrowProps.scale * 1.5, 8]} />
          <meshBasicMaterial color={color} opacity={opacity} transparent />
        </mesh>
      )}
    </group>
  );
}

// ============================================================================
// Point Component - Individual vector point with line from origin
// ============================================================================

function Point({ position, text, id, isHighlighted, isSelected, rank, onClick, onHover }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  // Animation for highlighted points
  useFrame((state) => {
    if (meshRef.current) {
      if (isHighlighted || isSelected) {
        meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.2);
      } else {
        meshRef.current.scale.setScalar(hovered ? 1.3 : 1);
      }
    }
  });
  
  // Determine color based on state
  const color = useMemo(() => {
    if (isSelected) return '#ff6b8a';
    if (isHighlighted) return '#00ff9d';
    if (hovered) return '#00d9ff';
    return '#7b61ff';
  }, [isSelected, isHighlighted, hovered]);
  
  const size = useMemo(() => {
    if (isSelected) return 0.08;
    if (isHighlighted) return 0.06 + (10 - Math.min(rank || 10, 10)) * 0.005;
    return 0.04;
  }, [isSelected, isHighlighted, rank]);

  // Line opacity based on state
  const lineOpacity = useMemo(() => {
    if (isSelected) return 0.9;
    if (isHighlighted) return 0.8;
    if (hovered) return 0.7;
    return 0.35;
  }, [isSelected, isHighlighted, hovered]);

  return (
    <group>
      {/* Vector line from origin */}
      <VectorLine 
        position={position} 
        color={color} 
        opacity={lineOpacity}
      />
      
      {/* Point sphere */}
      <group position={position}>
        <mesh
          ref={meshRef}
          onClick={(e) => {
            e.stopPropagation();
            onClick(id);
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHovered(true);
            onHover(id, text, e.point);
          }}
          onPointerOut={() => {
            setHovered(false);
            onHover(null);
          }}
        >
          <sphereGeometry args={[size, 16, 16]} />
          <meshStandardMaterial 
            color={color} 
            emissive={color}
            emissiveIntensity={isHighlighted || isSelected ? 0.8 : hovered ? 0.4 : 0.2}
          />
        </mesh>
        
        {/* Glow effect for highlighted/selected */}
        {(isHighlighted || isSelected) && (
          <mesh>
            <sphereGeometry args={[size * 2, 16, 16]} />
            <meshBasicMaterial 
              color={color} 
              transparent 
              opacity={0.15}
            />
          </mesh>
        )}
        
        {/* Label - show on hover or if selected */}
        {(hovered || isSelected) && (
          <Text
            position={[0, size + 0.1, 0]}
            fontSize={0.06}
            color="#e6e8eb"
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.004}
            outlineColor="#0a0e14"
          >
            {text.length > 20 ? text.substring(0, 20) + '...' : text}
          </Text>
        )}
        
        {/* Rank badge for search results */}
        {isHighlighted && rank && rank <= 3 && (
          <Html position={[size + 0.05, size + 0.05, 0]} center>
            <div style={{
              background: 'linear-gradient(135deg, #00d9ff, #7b61ff)',
              color: '#0a0e14',
              fontSize: '10px',
              fontWeight: 'bold',
              padding: '2px 6px',
              borderRadius: '10px',
              whiteSpace: 'nowrap'
            }}>
              #{rank}
            </div>
          </Html>
        )}
      </group>
    </group>
  );
}

// ============================================================================
// Points Cloud - All points rendered together
// ============================================================================

function PointsCloud({ points, highlightedIds, selectedId, onPointClick, onPointHover }) {
  // Create lookup for highlighted items with their ranks
  const highlightedMap = useMemo(() => {
    const map = new Map();
    highlightedIds.forEach((id, index) => {
      map.set(id, index + 1);
    });
    return map;
  }, [highlightedIds]);

  return (
    <group>
      {points.map((point) => (
        <Point
          key={point.id}
          id={point.id}
          position={point.xyz}
          text={point.text}
          isHighlighted={highlightedMap.has(point.id)}
          isSelected={point.id === selectedId}
          rank={highlightedMap.get(point.id)}
          onClick={onPointClick}
          onHover={onPointHover}
        />
      ))}
    </group>
  );
}

// ============================================================================
// Origin Marker - Small sphere at origin
// ============================================================================

function OriginMarker() {
  return (
    <mesh position={[0, 0, 0]}>
      <sphereGeometry args={[0.03, 16, 16]} />
      <meshBasicMaterial color="#ffffff" opacity={0.8} transparent />
    </mesh>
  );
}

// ============================================================================
// Grid and Axes helpers
// ============================================================================

function SceneHelpers() {
  return (
    <>
      {/* Grid on XZ plane through origin */}
      <gridHelper args={[4, 20, '#1a222d', '#151b24']} position={[0, 0, 0]} />
      
      {/* Origin marker */}
      <OriginMarker />
      
      {/* Axes from origin */}
      <group>
        {/* X axis - red */}
        <Line
          points={[[0, 0, 0], [2, 0, 0]]}
          color="#ff6b8a"
          lineWidth={2}
        />
        <Text position={[2.15, 0, 0]} fontSize={0.1} color="#ff6b8a">X</Text>
        
        {/* Y axis - green */}
        <Line
          points={[[0, 0, 0], [0, 2, 0]]}
          color="#00ff9d"
          lineWidth={2}
        />
        <Text position={[0, 2.15, 0]} fontSize={0.1} color="#00ff9d">Y</Text>
        
        {/* Z axis - blue */}
        <Line
          points={[[0, 0, 0], [0, 0, 2]]}
          color="#00d9ff"
          lineWidth={2}
        />
        <Text position={[0, 0, 2.15]} fontSize={0.1} color="#00d9ff">Z</Text>
      </group>
    </>
  );
}

// ============================================================================
// Camera Controls wrapper
// ============================================================================

function CameraController() {
  const { camera } = useThree();
  
  // Set initial camera position
  useMemo(() => {
    camera.position.set(2.5, 2, 2.5);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return (
    <OrbitControls
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      minDistance={0.5}
      maxDistance={10}
      dampingFactor={0.05}
      enableDamping={true}
      target={[0, 0, 0]}
    />
  );
}

// ============================================================================
// Main Scene3D Component
// ============================================================================

export function Scene3D({ 
  points = [], 
  highlightedIds = [],
  selectedId = null,
  onPointClick,
  projectionMethod = 'pca',
  itemCount = 0
}) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const handlePointHover = useCallback((id, text, position) => {
    if (id) {
      setHoveredPoint({ id, text, position });
    } else {
      setHoveredPoint(null);
    }
  }, []);

  return (
    <div className="scene-container" style={{ width: '100%', height: '100%' }}>
      {/* Status overlay */}
      <div className="scene-overlay">
        <span className="dot"></span>
        <span>{itemCount} vectors</span>
        <span style={{ color: 'var(--text-muted)' }}>•</span>
        <span>{projectionMethod.toUpperCase()}</span>
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ fov: 50 }}
        style={{ background: '#0a0e14' }}
        gl={{ antialias: true }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} color="#7b61ff" />
        
        {/* Scene helpers */}
        <SceneHelpers />
        
        {/* Points with vector lines */}
        <PointsCloud
          points={points}
          highlightedIds={highlightedIds}
          selectedId={selectedId}
          onPointClick={onPointClick}
          onPointHover={handlePointHover}
        />
        
        {/* Camera controls */}
        <CameraController />
        
        {/* Background */}
        <color attach="background" args={['#0a0e14']} />
        <fog attach="fog" args={['#0a0e14', 4, 10]} />
      </Canvas>

      {/* Hover tooltip (rendered outside canvas) */}
      {hoveredPoint && (
        <div 
          className="tooltip"
          style={{
            left: '50%',
            bottom: '20px',
            transform: 'translateX(-50%)'
          }}
        >
          {hoveredPoint.text}
        </div>
      )}
    </div>
  );
}

export default Scene3D;
