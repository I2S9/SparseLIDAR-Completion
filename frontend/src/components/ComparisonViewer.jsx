/**
 * Comparison viewer for Before/After demonstration
 * Shows: Partial cloud, Poisson reconstruction, Deep Learning reconstruction
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { loadPLY, loadPLYFromFile } from '../utils/loaders.js';

const VIEW_MODES = {
  PARTIAL: 'partial',
  POISSON: 'poisson',
  DEEP_LEARNING: 'deep_learning',
  SIDE_BY_SIDE: 'side_by_side'
};

export default function ComparisonViewer() {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const [viewMode, setViewMode] = useState(VIEW_MODES.PARTIAL);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loadedClouds, setLoadedClouds] = useState({
    partial: null,
    poisson: null,
    deepLearning: null
  });
  const [normals, setNormals] = useState({
    partial: null,
    poisson: null,
    deepLearning: null
  });
  const [showNormals, setShowNormals] = useState(false);
  const normalsGroupRef = useRef(null);

  useEffect(() => {
    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 2);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Orbital controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 0.5;
    controls.maxDistance = 10;
    controlsRef.current = controls;

    // Add grid helper
    const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Add axes helper
    const axesHelper = new THREE.AxesHelper(0.5);
    scene.add(axesHelper);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Create normals visualization
  const createNormalsGroup = (points, normalsArray, color) => {
    if (!normalsArray || normalsArray.length === 0) return null;
    
    const group = new THREE.Group();
    const positions = points.geometry.attributes.position.array;
    const numPoints = positions.length / 3;
    
    // Sample points to avoid too many arrows (every 10th point)
    const sampleRate = Math.max(1, Math.floor(numPoints / 500));
    const normalLength = 0.05;
    
    for (let i = 0; i < numPoints; i += sampleRate) {
      const idx = i * 3;
      const x = positions[idx];
      const y = positions[idx + 1];
      const z = positions[idx + 2];
      
      const nx = normalsArray[idx];
      const ny = normalsArray[idx + 1];
      const nz = normalsArray[idx + 2];
      
      // Skip if normal is zero
      if (nx === 0 && ny === 0 && nz === 0) continue;
      
      const origin = new THREE.Vector3(x, y, z);
      const direction = new THREE.Vector3(nx, ny, nz).normalize();
      
      const arrow = new THREE.ArrowHelper(
        direction,
        origin,
        normalLength,
        color,
        normalLength * 0.3,
        normalLength * 0.2
      );
      
      group.add(arrow);
    }
    
    return group;
  };

  // Update normals visualization when showNormals or viewMode changes
  useEffect(() => {
    if (!sceneRef.current) return;
    
    // Remove existing normals group
    if (normalsGroupRef.current) {
      sceneRef.current.remove(normalsGroupRef.current);
      normalsGroupRef.current = null;
    }
    
    // Add normals if enabled
    if (showNormals) {
      const group = new THREE.Group();
      let hasNormals = false;
      
      if (viewMode === VIEW_MODES.PARTIAL && loadedClouds.partial && normals.partial) {
        const normalsGroup = createNormalsGroup(loadedClouds.partial, normals.partial, 0xff6b6b);
        if (normalsGroup) {
          group.add(normalsGroup);
          hasNormals = true;
        }
      } else if (viewMode === VIEW_MODES.POISSON && loadedClouds.poisson && normals.poisson) {
        const normalsGroup = createNormalsGroup(loadedClouds.poisson, normals.poisson, 0x4ecdc4);
        if (normalsGroup) {
          group.add(normalsGroup);
          hasNormals = true;
        }
      } else if (viewMode === VIEW_MODES.DEEP_LEARNING && loadedClouds.deepLearning && normals.deepLearning) {
        const normalsGroup = createNormalsGroup(loadedClouds.deepLearning, normals.deepLearning, 0x95e1d3);
        if (normalsGroup) {
          group.add(normalsGroup);
          hasNormals = true;
        }
      } else if (viewMode === VIEW_MODES.SIDE_BY_SIDE) {
        if (loadedClouds.partial && normals.partial) {
          const normalsGroup = createNormalsGroup(loadedClouds.partial, normals.partial, 0xff6b6b);
          if (normalsGroup) {
            normalsGroup.position.x = -0.5;
            group.add(normalsGroup);
            hasNormals = true;
          }
        }
        if (loadedClouds.poisson && normals.poisson) {
          const normalsGroup = createNormalsGroup(loadedClouds.poisson, normals.poisson, 0x4ecdc4);
          if (normalsGroup) {
            normalsGroup.position.x = 0;
            group.add(normalsGroup);
            hasNormals = true;
          }
        }
        if (loadedClouds.deepLearning && normals.deepLearning) {
          const normalsGroup = createNormalsGroup(loadedClouds.deepLearning, normals.deepLearning, 0x95e1d3);
          if (normalsGroup) {
            normalsGroup.position.x = 0.5;
            group.add(normalsGroup);
            hasNormals = true;
          }
        }
      }
      
      if (hasNormals) {
        normalsGroupRef.current = group;
        sceneRef.current.add(group);
      }
    }
  }, [showNormals, viewMode, loadedClouds, normals]);

  // Update scene when view mode or loaded clouds change
  useEffect(() => {
    if (!sceneRef.current) return;

    // Remove all point clouds
    const objectsToRemove = [];
    sceneRef.current.traverse((child) => {
      if (child instanceof THREE.Points) {
        objectsToRemove.push(child);
      }
    });
    objectsToRemove.forEach(obj => {
      sceneRef.current.remove(obj);
      obj.geometry.dispose();
      obj.material.dispose();
    });

    // Add clouds based on view mode
    if (viewMode === VIEW_MODES.PARTIAL && loadedClouds.partial) {
      const partial = loadedClouds.partial.clone();
      partial.material = new THREE.PointsMaterial({
        color: 0xff6b6b,
        size: 0.01,
        sizeAttenuation: true
      });
      sceneRef.current.add(partial);
    } else if (viewMode === VIEW_MODES.POISSON && loadedClouds.poisson) {
      const poisson = loadedClouds.poisson.clone();
      poisson.material = new THREE.PointsMaterial({
        color: 0x4ecdc4,
        size: 0.01,
        sizeAttenuation: true
      });
      sceneRef.current.add(poisson);
    } else if (viewMode === VIEW_MODES.DEEP_LEARNING && loadedClouds.deepLearning) {
      const dl = loadedClouds.deepLearning.clone();
      dl.material = new THREE.PointsMaterial({
        color: 0x95e1d3,
        size: 0.01,
        sizeAttenuation: true
      });
      sceneRef.current.add(dl);
    } else if (viewMode === VIEW_MODES.SIDE_BY_SIDE) {
      // Side by side view
      if (loadedClouds.partial) {
        const partial = loadedClouds.partial.clone();
        partial.position.x = -0.5;
        partial.material = new THREE.PointsMaterial({
          color: 0xff6b6b,
          size: 0.008,
          sizeAttenuation: true
        });
        sceneRef.current.add(partial);
      }
      if (loadedClouds.poisson) {
        const poisson = loadedClouds.poisson.clone();
        poisson.position.x = 0;
        poisson.material = new THREE.PointsMaterial({
          color: 0x4ecdc4,
          size: 0.008,
          sizeAttenuation: true
        });
        sceneRef.current.add(poisson);
      }
      if (loadedClouds.deepLearning) {
        const dl = loadedClouds.deepLearning.clone();
        dl.position.x = 0.5;
        dl.material = new THREE.PointsMaterial({
          color: 0x95e1d3,
          size: 0.008,
          sizeAttenuation: true
        });
        sceneRef.current.add(dl);
      }
    }
  }, [viewMode, loadedClouds]);

  const loadPointCloud = async (file, type) => {
    setIsLoading(true);
    setError(null);

    try {
      const { points, normals: normalsArray } = await loadPLYFromFile(file);

      // Center and scale
      const box = new THREE.Box3().setFromObject(points);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      
      points.geometry.translate(-center.x, -center.y, -center.z);
      points.geometry.scale(1 / maxDim, 1 / maxDim, 1 / maxDim);

      // Store loaded cloud
      setLoadedClouds(prev => ({
        ...prev,
        [type]: points
      }));

      // Store normals if present
      setNormals(prev => ({
        ...prev,
        [type]: normalsArray
      }));

      // Reset camera
      cameraRef.current.position.set(0, 0, 2);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();

      console.log(`Loaded ${type}: ${points.geometry.attributes.position.count} points${normalsArray ? ' (with normals)' : ''}`);
    } catch (err) {
      console.error('Error loading point cloud:', err);
      setError(`Failed to load ${type}: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (event, type) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.ply')) {
      loadPointCloud(file, type);
    } else {
      setError('Please select a .ply file');
    }
  };

  const loadDefaultFiles = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Try to load from public/exports directory (served by Vite)
      const files = {
        partial: '/exports/input_partial.ply',
        poisson: '/exports/poisson_reconstruction.ply',
        deepLearning: '/exports/output_predicted.ply'
      };

      // Load files sequentially
      const newLoadedClouds = { ...loadedClouds };
      const newNormals = { ...normals };
      
      for (const [type, path] of Object.entries(files)) {
        try {
          const { points, normals: normalsArray } = await loadPLY(path);
          
          // Center and scale
          const box = new THREE.Box3().setFromObject(points);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          
          points.geometry.translate(-center.x, -center.y, -center.z);
          points.geometry.scale(1 / maxDim, 1 / maxDim, 1 / maxDim);

          newLoadedClouds[type] = points;
          newNormals[type] = normalsArray;
        } catch (err) {
          console.warn(`Could not load ${type} from ${path}:`, err);
        }
      }

      // Update state with all loaded clouds and normals
      setLoadedClouds(newLoadedClouds);
      setNormals(newNormals);

      // Set default view if any cloud was loaded
      const hasAnyCloud = Object.values(newLoadedClouds).some(cloud => cloud !== null);
      if (hasAnyCloud) {
        setViewMode(VIEW_MODES.SIDE_BY_SIDE);
      } else {
        setError('Could not load demo files. Please use file upload buttons.');
      }
    } catch (err) {
      setError(`Failed to load default files: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      {/* Three.js canvas */}
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* UI Overlay */}
      <div
        style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          zIndex: 1000,
          background: 'rgba(0, 0, 0, 0.85)',
          padding: '25px',
          borderRadius: '10px',
          color: 'white',
          fontFamily: 'Arial, sans-serif',
          minWidth: '300px',
          maxWidth: '400px'
        }}
      >
        <h1 style={{ margin: '0 0 10px 0', fontSize: '24px', fontWeight: 'bold', color: '#4ecdc4' }}>
          SparseLIDAR Completion
        </h1>
        <h2 style={{ margin: '0 0 20px 0', fontSize: '16px', opacity: 0.8, fontStyle: 'italic' }}>
          Before / After Comparison Demo
        </h2>
        <p style={{ margin: '0 0 20px 0', fontSize: '12px', opacity: 0.7, lineHeight: '1.5' }}>
          Interactive 3D visualization comparing partial input, Poisson reconstruction, and Deep Learning completion.
        </p>

        {/* Load default files button */}
        <button
          onClick={loadDefaultFiles}
          disabled={isLoading}
          style={{
            width: '100%',
            padding: '12px',
            marginBottom: '20px',
            background: isLoading ? '#666' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 'bold'
          }}
        >
          {isLoading ? 'Loading...' : 'Load Demo Files'}
        </button>

        {/* File loaders */}
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '14px', marginBottom: '10px', opacity: 0.9 }}>Load Files:</h3>
          
          <div style={{ marginBottom: '10px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px', opacity: 0.8 }}>
              Partial Input (Red)
            </label>
            <label
              style={{
                display: 'inline-block',
                padding: '8px 15px',
                background: '#ff6b6b',
                color: 'white',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              Choose File
              <input
                type="file"
                accept=".ply"
                onChange={(e) => handleFileSelect(e, 'partial')}
                disabled={isLoading}
                style={{ display: 'none' }}
              />
            </label>
          </div>

          <div style={{ marginBottom: '10px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px', opacity: 0.8 }}>
              Poisson (Cyan)
            </label>
            <label
              style={{
                display: 'inline-block',
                padding: '8px 15px',
                background: '#4ecdc4',
                color: 'white',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              Choose File
              <input
                type="file"
                accept=".ply"
                onChange={(e) => handleFileSelect(e, 'poisson')}
                disabled={isLoading}
                style={{ display: 'none' }}
              />
            </label>
          </div>

          <div style={{ marginBottom: '10px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px', opacity: 0.8 }}>
              Deep Learning (Green)
            </label>
            <label
              style={{
                display: 'inline-block',
                padding: '8px 15px',
                background: '#95e1d3',
                color: 'white',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              Choose File
              <input
                type="file"
                accept=".ply"
                onChange={(e) => handleFileSelect(e, 'deepLearning')}
                disabled={isLoading}
                style={{ display: 'none' }}
              />
            </label>
          </div>
        </div>

        {/* Show normals button */}
        <div style={{ marginBottom: '20px' }}>
          <button
            onClick={() => setShowNormals(!showNormals)}
            style={{
              width: '100%',
              padding: '10px',
              background: showNormals ? '#6c5ce7' : '#333',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 'bold'
            }}
          >
            {showNormals ? 'Hide Normals' : 'Show Normals'}
          </button>
        </div>

        {/* View mode selector */}
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '14px', marginBottom: '10px', opacity: 0.9 }}>View Mode:</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={() => setViewMode(VIEW_MODES.PARTIAL)}
              style={{
                padding: '10px',
                background: viewMode === VIEW_MODES.PARTIAL ? '#ff6b6b' : '#333',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              Partial Only
            </button>
            <button
              onClick={() => setViewMode(VIEW_MODES.POISSON)}
              style={{
                padding: '10px',
                background: viewMode === VIEW_MODES.POISSON ? '#4ecdc4' : '#333',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              Poisson Only
            </button>
            <button
              onClick={() => setViewMode(VIEW_MODES.DEEP_LEARNING)}
              style={{
                padding: '10px',
                background: viewMode === VIEW_MODES.DEEP_LEARNING ? '#95e1d3' : '#333',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              Deep Learning Only
            </button>
            <button
              onClick={() => setViewMode(VIEW_MODES.SIDE_BY_SIDE)}
              style={{
                padding: '10px',
                background: viewMode === VIEW_MODES.SIDE_BY_SIDE ? '#6c5ce7' : '#333',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 'bold'
              }}
            >
              Side by Side (All)
            </button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div
            style={{
              padding: '10px',
              background: 'rgba(255, 0, 0, 0.3)',
              borderRadius: '4px',
              fontSize: '12px',
              marginBottom: '15px'
            }}
          >
            {error}
          </div>
        )}

        {/* Instructions */}
        <div style={{ fontSize: '11px', opacity: 0.7, lineHeight: '1.6' }}>
          <p style={{ margin: '5px 0' }}><strong>Controls:</strong></p>
          <p style={{ margin: '3px 0' }}>• Left-click + drag: Rotate</p>
          <p style={{ margin: '3px 0' }}>• Right-click + drag: Pan</p>
          <p style={{ margin: '3px 0' }}>• Scroll: Zoom</p>
        </div>

        {/* Status */}
        <div style={{ marginTop: '15px', fontSize: '11px', opacity: 0.6 }}>
          <p style={{ margin: '3px 0' }}>
            Partial: {loadedClouds.partial ? '✓' : '✗'}
          </p>
          <p style={{ margin: '3px 0' }}>
            Poisson: {loadedClouds.poisson ? '✓' : '✗'}
          </p>
          <p style={{ margin: '3px 0' }}>
            Deep Learning: {loadedClouds.deepLearning ? '✓' : '✗'}
          </p>
        </div>
      </div>
    </div>
  );
}

