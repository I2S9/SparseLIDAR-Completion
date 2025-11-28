/**
 * Comparison viewer for Before/After demonstration
 * Shows: Partial cloud, Poisson reconstruction, Deep Learning reconstruction
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { loadPLY, loadPLYFromFile, loadPLYMesh } from '../utils/loaders.js';

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
  const [pointSize, setPointSize] = useState(0.008); // Default: middle of 0.005-0.012 range
  const [showPoissonMesh, setShowPoissonMesh] = useState(false);
  const poissonMeshRef = useRef(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState(null);

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

  // Load demo files automatically on mount
  useEffect(() => {
    if (sceneRef.current) {
      loadDefaultFiles();
    }
  }, [sceneRef.current]);

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

  // Update point sizes when pointSize changes
  useEffect(() => {
    if (!sceneRef.current) return;
    
    sceneRef.current.traverse((child) => {
      if (child instanceof THREE.Points && child.material instanceof THREE.PointsMaterial) {
        if (viewMode === VIEW_MODES.SIDE_BY_SIDE) {
          child.material.size = pointSize * 0.8;
        } else {
          child.material.size = pointSize;
        }
        child.material.needsUpdate = true;
      }
    });
  }, [pointSize, viewMode]);

  // Update Poisson mesh wireframe visibility
  useEffect(() => {
    if (!sceneRef.current) return;
    
    // Remove all existing Poisson meshes from scene
    const meshesToRemove = [];
    sceneRef.current.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material && child.material.wireframe) {
        meshesToRemove.push(child);
      }
    });
    meshesToRemove.forEach(mesh => {
      sceneRef.current.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    });
    
    // Add mesh if enabled and mesh is loaded
    if (showPoissonMesh && poissonMeshRef.current) {
      // Only show in Poisson or Side by Side modes
      if (viewMode === VIEW_MODES.POISSON || viewMode === VIEW_MODES.SIDE_BY_SIDE) {
        const mesh = poissonMeshRef.current.clone();
        
        if (viewMode === VIEW_MODES.SIDE_BY_SIDE) {
          mesh.position.x = 0; // Center position for Poisson in side-by-side
        }
        
        sceneRef.current.add(mesh);
      }
    }
  }, [showPoissonMesh, viewMode]);

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
        size: pointSize,
        sizeAttenuation: true
      });
      sceneRef.current.add(partial);
    } else if (viewMode === VIEW_MODES.POISSON && loadedClouds.poisson) {
      const poisson = loadedClouds.poisson.clone();
      poisson.material = new THREE.PointsMaterial({
        color: 0x4ecdc4,
        size: pointSize,
        sizeAttenuation: true
      });
      sceneRef.current.add(poisson);
    } else if (viewMode === VIEW_MODES.DEEP_LEARNING && loadedClouds.deepLearning) {
      const dl = loadedClouds.deepLearning.clone();
      dl.material = new THREE.PointsMaterial({
        color: 0x95e1d3,
        size: pointSize,
        sizeAttenuation: true
      });
      sceneRef.current.add(dl);
    } else if (viewMode === VIEW_MODES.SIDE_BY_SIDE) {
      // Side by side view - slightly smaller for better visibility
      const sideBySideSize = pointSize * 0.8;
      if (loadedClouds.partial) {
        const partial = loadedClouds.partial.clone();
        partial.position.x = -0.5;
        partial.material = new THREE.PointsMaterial({
          color: 0xff6b6b,
          size: sideBySideSize,
          sizeAttenuation: true
        });
        sceneRef.current.add(partial);
      }
      if (loadedClouds.poisson) {
        const poisson = loadedClouds.poisson.clone();
        poisson.position.x = 0;
        poisson.material = new THREE.PointsMaterial({
          color: 0x4ecdc4,
          size: sideBySideSize,
          sizeAttenuation: true
        });
        sceneRef.current.add(poisson);
      }
      if (loadedClouds.deepLearning) {
        const dl = loadedClouds.deepLearning.clone();
        dl.position.x = 0.5;
        dl.material = new THREE.PointsMaterial({
          color: 0x95e1d3,
          size: sideBySideSize,
          sizeAttenuation: true
        });
        sceneRef.current.add(dl);
      }
    }
  }, [viewMode, loadedClouds, pointSize]);

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

      // Try to load Poisson mesh
      try {
        const mesh = await loadPLYMesh('/exports/poisson_mesh.ply');
        
        // Center and scale mesh (same as point clouds)
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        
        mesh.geometry.translate(-center.x, -center.y, -center.z);
        mesh.geometry.scale(1 / maxDim, 1 / maxDim, 1 / maxDim);
        
        poissonMeshRef.current = mesh;
        console.log('Poisson mesh loaded successfully');
      } catch (err) {
        console.warn('Could not load Poisson mesh:', err);
      }

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
          minWidth: '320px',
          maxWidth: '420px',
          maxHeight: 'calc(100vh - 40px)',
          overflowY: 'auto',
          overflowX: 'hidden'
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
            background: isLoading ? '#666' : '#333',
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
          
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <div style={{ flex: '1', minWidth: '100px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '11px', opacity: 0.8 }}>
                Partial (Red)
              </label>
              <label
                style={{
                  display: 'block',
                  padding: '8px 12px',
                  background: '#ff6b6b',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  textAlign: 'center'
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

            <div style={{ flex: '1', minWidth: '100px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '11px', opacity: 0.8 }}>
                Poisson (Cyan)
              </label>
              <label
                style={{
                  display: 'block',
                  padding: '8px 12px',
                  background: '#4ecdc4',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  textAlign: 'center'
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

            <div style={{ flex: '1', minWidth: '100px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '11px', opacity: 0.8 }}>
                DL (Green)
              </label>
              <label
                style={{
                  display: 'block',
                  padding: '8px 12px',
                  background: '#95e1d3',
                  color: 'white',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  textAlign: 'center'
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

        {/* Show Poisson mesh wireframe button */}
        <div style={{ marginBottom: '20px' }}>
          <button
            onClick={() => setShowPoissonMesh(!showPoissonMesh)}
            style={{
              width: '100%',
              padding: '10px',
              background: showPoissonMesh ? '#4ecdc4' : '#333',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 'bold'
            }}
          >
            {showPoissonMesh ? 'Hide Poisson Mesh' : 'Show Poisson Mesh (Wireframe)'}
          </button>
        </div>

        {/* Compare Metrics button */}
        <div style={{ marginBottom: '20px' }}>
          <button
            onClick={async () => {
              if (!showMetrics) {
                try {
                  const response = await fetch('/metrics.json');
                  
                  // Check if response is actually JSON
                  const contentType = response.headers.get('content-type');
                  if (!contentType || !contentType.includes('application/json')) {
                    throw new Error('Response is not JSON. File may not exist. Run: python backend/notebooks/generate_metrics.py');
                  }
                  
                  if (response.ok) {
                    const text = await response.text();
                    
                    // Check if response is empty or HTML
                    if (!text || text.trim().startsWith('<')) {
                      throw new Error('metrics.json not found. Run: python backend/notebooks/generate_metrics.py');
                    }
                    
                    try {
                      const data = JSON.parse(text);
                      setMetrics(data);
                      setShowMetrics(true);
                      setError(null);
                    } catch (parseErr) {
                      throw new Error(`Invalid JSON format: ${parseErr.message}. Regenerate metrics.json`);
                    }
                  } else {
                    throw new Error(`HTTP ${response.status}: Could not load metrics.json. Run: python backend/notebooks/generate_metrics.py`);
                  }
                } catch (err) {
                  setError(`Failed to load metrics: ${err.message}`);
                  setShowMetrics(false);
                }
              } else {
                setShowMetrics(false);
              }
            }}
            style={{
              width: '100%',
              padding: '10px',
              background: showMetrics ? '#6c5ce7' : '#333',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 'bold'
            }}
          >
            {showMetrics ? 'Hide Metrics' : 'Compare Metrics'}
          </button>
        </div>

        {/* Metrics display */}
        {showMetrics && metrics && (
          <div style={{ 
            marginBottom: '20px', 
            padding: '15px', 
            background: 'rgba(255, 255, 255, 0.1)', 
            borderRadius: '6px',
            fontSize: '12px'
          }}>
            <h3 style={{ fontSize: '14px', marginBottom: '10px', opacity: 0.9 }}>
              Evaluation Metrics
            </h3>
            {Object.entries(metrics).map(([method, data]) => {
              const methodName = method === 'partial' ? 'Partial' : 
                                method === 'poisson' ? 'Poisson' : 
                                method === 'deep' ? 'Deep Learning' : method;
              
              // Find best values for highlighting
              const allCds = Object.values(metrics).map(m => m.cd).filter(v => v !== null);
              const allFscores = Object.values(metrics).map(m => m.fscore).filter(v => v !== null);
              const bestCd = Math.min(...allCds);
              const bestFscore = Math.max(...allFscores);
              
              const isBestCd = data.cd !== null && data.cd === bestCd;
              const isBestFscore = data.fscore !== null && data.fscore === bestFscore;
              
              return (
                <div key={method} style={{ marginBottom: '10px', padding: '8px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '5px', color: method === 'partial' ? '#ff6b6b' : method === 'poisson' ? '#4ecdc4' : '#95e1d3' }}>
                    {methodName}
                  </div>
                  <div style={{ fontSize: '11px', lineHeight: '1.6' }}>
                    <div>
                      CD: <span style={{ color: isBestCd ? '#4CAF50' : 'inherit', fontWeight: isBestCd ? 'bold' : 'normal' }}>
                        {data.cd !== null ? data.cd.toFixed(6) : 'N/A'}
                      </span>
                      {isBestCd && <span style={{ color: '#4CAF50', marginLeft: '5px' }}>✓ Best</span>}
                    </div>
                    <div>
                      F-score: <span style={{ color: isBestFscore ? '#4CAF50' : 'inherit', fontWeight: isBestFscore ? 'bold' : 'normal' }}>
                        {data.fscore !== null ? data.fscore.toFixed(4) : 'N/A'}
                      </span>
                      {isBestFscore && <span style={{ color: '#4CAF50', marginLeft: '5px' }}>✓ Best</span>}
                    </div>
                  </div>
                </div>
              );
            })}
            <div style={{ fontSize: '10px', opacity: 0.7, marginTop: '10px', fontStyle: 'italic' }}>
              Lower CD is better • Higher F-score is better
            </div>
          </div>
        )}

        {/* Point size slider */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', fontSize: '14px', marginBottom: '10px', opacity: 0.9 }}>
            Point Size: {pointSize.toFixed(3)}
          </label>
          <input
            type="range"
            min="0.005"
            max="0.012"
            step="0.0005"
            value={pointSize}
            onChange={(e) => setPointSize(parseFloat(e.target.value))}
            style={{
              width: '100%',
              height: '6px',
              borderRadius: '3px',
              background: '#333',
              outline: 'none',
              cursor: 'pointer'
            }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', opacity: 0.6, marginTop: '5px' }}>
            <span>0.005</span>
            <span>0.012</span>
          </div>
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
      </div>
    </div>
  );
}

