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

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const spotLight = new THREE.SpotLight(0xffffff, 1.5);
spotLight.position.set(5, 5, 5);
scene.add(spotLight);

const listener = new THREE.AudioListener();
camera.add(listener);
const vesiSound = new THREE.Audio(listener);
const analyser = new THREE.AudioAnalyser(vesiSound, 256); // Brain to 'read' the sound

// Load VRM
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

        const ls = document.getElementById('loading-screen');
        if (ls) {
            ls.classList.add('opacity-0');
            setTimeout(() => {
                if (ls.parentNode) ls.remove();
            }, 500);
        }
        
        console.log("VRM Load Complete");
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

// TODO fix this
const clock = new THREE.Clock();
function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    
    if (currentVrm) {
        const volume = analyser.getAverageFrequency();
        const mouthOpen = Math.min(volume / 40, 1.0);

        // animation
        currentVrm.expressionManager.setValue('aa', mouthOpen);
        
        currentVrm.update(deltaTime);
    }
    
    renderer.render(scene, camera);
}
animate();

// Chat
async function sendMessage(event) {
    if (event) event.preventDefault();

    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    input.value = ''; 

    // 127 to local
    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        console.log("Data received:", data);

        if (typeof vesiSound !== 'undefined' && data.audio_url) {
            const audioLoader = new THREE.AudioLoader();
            audioLoader.load(data.audio_url, (buffer) => {
                if (vesiSound.isPlaying) vesiSound.stop();
                vesiSound.setBuffer(buffer);
                vesiSound.play();
            });
        } else {
            console.warn("vesiSound is not defined or no audio_url received");
        }

    } catch (err) {
        console.error("Fetch Error:", err);
    }
}

// 4. Link to UI
const chatForm = document.getElementById('chat-form');

chatForm.addEventListener('submit', function(e) {
    e.preventDefault();
    e.stopImmediatePropagation();
    console.log("Reload blocked. Calling sendMessage...");
    sendMessage(e);
    return false;
});

// debug

// console.log("=== SCRIPT LOADED ===");
// console.log("Form element:", document.getElementById('chat-form'));
// console.log("Send button:", document.getElementById('send-btn'));
// console.log("Chat input:", document.getElementById('chat-input'));

// const form = document.getElementById('chat-form');
// if (form) {
//     console.log("Attaching listener to form...");
//     form.addEventListener('submit', function(e) {
//         console.log("🚨 FORM SUBMIT EVENT FIRED!");
//         e.preventDefault();
//         e.stopPropagation();
//         sendMessage(e);
//         return false;
//     }, {capture:true});
//     console.log("Listener attached successfully");
// } else {
//     console.error("❌ FORM NOT FOUND!");
// }