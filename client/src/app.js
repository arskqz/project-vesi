import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';

// Scene 
const scene = new THREE.Scene();

// Camera (set to about chest height)
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.0, 0.8); 

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

// Key light (main light)
const keyLight = new THREE.DirectionalLight(0xffffff, 0.2);
keyLight.position.set(2, 3, 3);
scene.add(keyLight);

// Fill light (soft filler)
const fillLight = new THREE.DirectionalLight(0xaaccff, 0.2);
fillLight.position.set(-2, 2, 2);
scene.add(fillLight);

// Rim light (behind)
const rimLight = new THREE.DirectionalLight(0xff88cc, 0.3); 
rimLight.position.set(0, 2, -3);
scene.add(rimLight);

const listener = new THREE.AudioListener();
camera.add(listener);
const vesiSound = new THREE.Audio(listener);
const analyser = new THREE.AudioAnalyser(vesiSound, 256);

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

         if (vrm.humanoid) {
            vrm.humanoid.resetNormalizedPose();
            
            const leftUpperArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
            const leftLowerArm = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
            const leftHand = vrm.humanoid.getNormalizedBoneNode('leftHand');
            
            if (leftUpperArm) {
                leftUpperArm.rotation.x = 0.6;   // Rotate forward toward body
                leftUpperArm.rotation.y = 0.1;
                leftUpperArm.rotation.z = 1.1;   // Bring arm down
            }
            if (leftLowerArm) {
                leftLowerArm.rotation.x = -0.75; // bend elboe outward
                leftLowerArm.rotation.y = 0;     // Bend elbow inward
                leftLowerArm.rotation.z = 1;     // Additional bend
            }
            if (leftHand) {
                leftHand.rotation.z = 0.2;
                leftHand.rotation.x = 0;         
            }
            
            const rightUpperArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
            const rightLowerArm = vrm.humanoid.getNormalizedBoneNode('rightLowerArm');
            const rightHand = vrm.humanoid.getNormalizedBoneNode('rightHand');
            
            if (rightUpperArm) {
                rightUpperArm.rotation.x = 0.6;    // Rotate forward toward body
                rightUpperArm.rotation.y = -0.1;
                rightUpperArm.rotation.z = -1.1;   // Bring arm down
            }
            if (rightLowerArm) {
                rightLowerArm.rotation.x = -0.75;  // Bend elbow outward
                rightLowerArm.rotation.y = 0;      // Bend elbow inward
                rightLowerArm.rotation.z = -1;     // Additional bend
            }
            if (rightHand) {
                rightHand.rotation.z = -0.2;
                rightHand.rotation.x = -2;         
            }
        }

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

const clock = new THREE.Clock();
let idleTime = 0;

function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    idleTime += deltaTime;
    
    if (currentVrm) {
        // Lip flap anim
        const volume = analyser.getAverageFrequency();
        const mouthOpen = Math.min(volume / 40, 1.0);
        currentVrm.expressionManager.setValue('aa', mouthOpen);
        
        // full idle anim
        
        // chest / body move up and down
        const breathCycle = Math.sin(idleTime * 1.5) * 0.015; 
        currentVrm.scene.position.y = breathCycle;
        
        // Sway left to right
        const swayCycle = Math.sin(idleTime * 0.4) * 0.02; 
        currentVrm.scene.rotation.z = swayCycle;
        
        // Head tilt
        const headBone = currentVrm.humanoid.getNormalizedBoneNode('head');
        if (headBone) {
            const headTilt = Math.sin(idleTime * 0.6) * 0.05;
            headBone.rotation.z = headTilt;
        }
        
        // Random blink
        if (Math.random() < 0.002) {
            currentVrm.expressionManager.setValue('blink', 1.0);
            setTimeout(() => {
                currentVrm.expressionManager.setValue('blink', 0);
            }, 150);
        }
        
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
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        console.log("Data received:", data);

        updateMoodBar(data.mood);

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

// vesi_mood_scrore
function updateMoodBar(moodScore) {
    const moodBar = document.getElementById('mood-bar');
    const moodValue = document.getElementById('mood-value');
    
    moodBar.style.width = moodScore + '%';
    moodValue.textContent = moodScore + '%';
}

// Link to UI
const chatForm = document.getElementById('chat-form');

chatForm.addEventListener('submit', function(e) {
    e.preventDefault();
    e.stopImmediatePropagation();
    console.log("Calling sendMessage...");
    sendMessage(e);
    return false;
});

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

document.getElementById('mic-btn').addEventListener('mousedown', async function() {
    if (isRecording) return;
    
    const micBtn = this;
    isRecording = true;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];
            
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            
            try {
                const response = await fetch('http://127.0.0.1:8000/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                const transcribedText = data.text;
                
                document.getElementById('chat-input').value = transcribedText;
                
                // Auto send timeout ms
                setTimeout(() => {
                    sendMessage();
                }, 1000);
                
            } catch (err) {
                console.error("Transcription error:", err);
            }
            
            isRecording = false;
        };
        
        mediaRecorder.start();
        micBtn.classList.add('text-red-500');
        console.log("Recording...");
        
    } catch (err) {
        console.error("Mic access error:", err);
        isRecording = false;
    }
});

document.getElementById('mic-btn').addEventListener('mouseup', function() {
    if (!isRecording || !mediaRecorder) return;
    
    const micBtn = this;
    
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    micBtn.classList.remove('text-red-500');
    console.log("Recording stopped, transcribing...");
});

// fat finger protection
document.getElementById('mic-btn').addEventListener('mouseleave', function() {
    if (!isRecording || !mediaRecorder) return;
    
    const micBtn = this;
    
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    micBtn.classList.remove('text-red-500');
    console.log("Recording stopped, transcribing...");
});
