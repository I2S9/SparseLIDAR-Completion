/**
 * Helpers to load PLY or GLB files in Three.js
 */

import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

/**
 * Load a PLY file and return a Three.js Points object with normals
 * @param {string} url - URL or path to PLY file
 * @returns {Promise<{points: THREE.Points, normals: Float32Array|null}>} Promise resolving to Points object and normals
 */
export async function loadPLY(url) {
  return new Promise((resolve, reject) => {
    const loader = new PLYLoader();
    loader.load(
      url,
      (geometry) => {
        // Create material for points
        const material = new THREE.PointsMaterial({
          color: 0x00ff00,
          size: 0.01,
          sizeAttenuation: true
        });
        
        // Create points object
        const points = new THREE.Points(geometry, material);
        
        // Extract normals if present
        let normals = null;
        if (geometry.attributes.normal) {
          normals = geometry.attributes.normal.array;
        }
        
        resolve({ points, normals });
      },
      (progress) => {
        // Progress callback
        console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
      },
      (error) => {
        reject(error);
      }
    );
  });
}

/**
 * Load a PLY mesh file (with faces) and return a Three.js Mesh object
 * @param {string} url - URL or path to PLY file
 * @returns {Promise<THREE.Mesh>} Promise resolving to Mesh object
 */
export async function loadPLYMesh(url) {
  return new Promise((resolve, reject) => {
    const loader = new PLYLoader();
    loader.load(
      url,
      (geometry) => {
        // Check if geometry has faces (index attribute)
        if (geometry.index && geometry.index.count > 0) {
          // It's a mesh with faces
          const material = new THREE.MeshBasicMaterial({
            wireframe: true,
            color: 0x4ecdc4
          });
          const mesh = new THREE.Mesh(geometry, material);
          resolve(mesh);
        } else {
          reject(new Error('PLY file does not contain mesh faces'));
        }
      },
      (progress) => {
        console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
      },
      (error) => {
        reject(error);
      }
    );
  });
}

/**
 * Load a GLB/GLTF file
 * @param {string} url - URL or path to GLB/GLTF file
 * @returns {Promise<THREE.Group>} Promise resolving to Group object
 */
export async function loadGLB(url) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        resolve(gltf.scene);
      },
      (progress) => {
        console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
      },
      (error) => {
        reject(error);
      }
    );
  });
}

/**
 * Load a PLY file from File object
 * @param {File} file - File object
 * @returns {Promise<{points: THREE.Points, normals: Float32Array|null}>} Promise resolving to Points object and normals
 */
export async function loadPLYFromFile(file) {
  return new Promise((resolve, reject) => {
    const loader = new PLYLoader();
    const reader = new FileReader();
    
    reader.onload = (event) => {
      try {
        const geometry = loader.parse(event.target.result);
        const material = new THREE.PointsMaterial({
          color: 0x00ff00,
          size: 0.01,
          sizeAttenuation: true
        });
        const points = new THREE.Points(geometry, material);
        
        // Extract normals if present
        let normals = null;
        if (geometry.attributes.normal) {
          normals = geometry.attributes.normal.array;
        }
        
        resolve({ points, normals });
      } catch (error) {
        reject(error);
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsArrayBuffer(file);
  });
}
