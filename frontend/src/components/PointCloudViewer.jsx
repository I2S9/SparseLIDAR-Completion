/**
 * Three.js viewer for PLY/GLB cloud files
 * Displays point clouds with orbital controls and file import
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { loadPLY, loadPLYFromFile } from '../utils/loaders.js';

export default function PointCloudViewer() {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const pointsRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
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

  const loadPointCloud = async (file) => {
    setIsLoading(true);
    setError(null);

    try {
      // Remove previous point cloud
      if (pointsRef.current && sceneRef.current) {
        sceneRef.current.remove(pointsRef.current);
        pointsRef.current.geometry.dispose();
        pointsRef.current.material.dispose();
        pointsRef.current = null;
      }

      // Load new point cloud
      const points = await loadPLYFromFile(file);

      // Center and scale point cloud
      const box = new THREE.Box3().setFromObject(points);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      
      points.geometry.translate(-center.x, -center.y, -center.z);
      points.geometry.scale(1 / maxDim, 1 / maxDim, 1 / maxDim);

      // Update material
      points.material = new THREE.PointsMaterial({
        color: 0x00ff00,
        size: 0.01,
        sizeAttenuation: true
      });

      // Add to scene
      sceneRef.current.add(points);
      pointsRef.current = points;

      // Reset camera position
      cameraRef.current.position.set(0, 0, 2);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();

      console.log(`Loaded point cloud: ${points.geometry.attributes.position.count} points`);
    } catch (err) {
      console.error('Error loading point cloud:', err);
      setError(`Failed to load file: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.name.endsWith('.ply')) {
        loadPointCloud(file);
      } else {
        setError('Please select a .ply file');
      }
    }
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      {/* Three.js canvas container */}
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* UI Overlay */}
      <div
        style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          zIndex: 1000,
          background: 'rgba(0, 0, 0, 0.7)',
          padding: '20px',
          borderRadius: '8px',
          color: 'white',
          fontFamily: 'Arial, sans-serif'
        }}
      >
        <h2 style={{ margin: '0 0 15px 0', fontSize: '18px' }}>Point Cloud Viewer</h2>
        
        {/* File input button */}
        <label
          style={{
            display: 'inline-block',
            padding: '10px 20px',
            background: '#4CAF50',
            color: 'white',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold'
          }}
        >
          {isLoading ? 'Loading...' : 'Import PLY File'}
          <input
            type="file"
            accept=".ply"
            onChange={handleFileSelect}
            disabled={isLoading}
            style={{ display: 'none' }}
          />
        </label>

        {/* Error message */}
        {error && (
          <div
            style={{
              marginTop: '10px',
              padding: '10px',
              background: 'rgba(255, 0, 0, 0.3)',
              borderRadius: '4px',
              fontSize: '12px'
            }}
          >
            {error}
          </div>
        )}

        {/* Instructions */}
        <div style={{ marginTop: '15px', fontSize: '12px', opacity: 0.8 }}>
          <p style={{ margin: '5px 0' }}>• Click and drag to rotate</p>
          <p style={{ margin: '5px 0' }}>• Right-click and drag to pan</p>
          <p style={{ margin: '5px 0' }}>• Scroll to zoom</p>
        </div>
      </div>
    </div>
  );
}
