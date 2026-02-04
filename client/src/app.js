import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';

// Scene 
const scene = new THREE.Scene();

// Camera 
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.3, 2.5); 

// Renderer 
const container = document.getElementById('vrm-canvas-container');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

// 4. Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const spotLight = new THREE.SpotLight(0xffffff, 1.5);
spotLight.position.set(5, 5, 5);
scene.add(spotLight);

// 5. Load VRM
let currentVrm = null;
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.load(
    './models/Vesi_00.vrm',
    (gltf) => {
        const vrm = gltf.userData.vrm;
        scene.add(vrm.scene);
        currentVrm = vrm;

        vrm.scene.rotation.y = Math.PI;

        document.getElementById('loading-screen').classList.add('opacity-0');
        setTimeout(() => document.getElementById('loading-screen').remove(), 500);
        
        console.log("VRM Load Complete:", vrm);
    },
    (progress) => console.log('Loading...', (progress.loaded / progress.total * 100) + '%'),
    (error) => console.error('Error loading VRM:', error)
);

// Window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// 7. Renderer loop
const clock = new THREE.Clock();
function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    
    
    if (currentVrm) {
        currentVrm.update(deltaTime);
    }
    
    renderer.render(scene, camera);
}
animate();