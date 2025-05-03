import { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// Component for 3D Point Cloud visualization
function ThreeDReconstruction() {
    const [disparityImage, setDisparityImage] = useState(null);
    const [leftEpipolarImage, setLeftEpipolarImage] = useState(null);
    const [rightEpipolarImage, setRightEpipolarImage] = useState(null);
    const [points3D, setPoints3D] = useState([]);
    const [numDisparities, setNumDisparities] = useState(64);
    const [method, setMethod] = useState('StereoBM');
    const leftInputRef = useRef(null);
    const rightInputRef = useRef(null);
    
    // Handle file upload and reconstruction
    const handleUpload = async () => {
        if (!leftInputRef.current.files[0] || !rightInputRef.current.files[0]) {
            alert('Please upload both left and right images!');
            return;
        }

        const formData = new FormData();
        formData.append('left_image', leftInputRef.current.files[0]);
        formData.append('right_image', rightInputRef.current.files[0]);
        formData.append('num_disparities', numDisparities);
        formData.append('method', method);  // Gửi phương pháp cho backend

        try {
          const response = await fetch('http://localhost:5000/3dconstruction', {
            method: 'POST',
            body: formData,
          });
            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }
            setDisparityImage(`data:image/png;base64,${data.disparity}`);
            setLeftEpipolarImage(`data:image/png;base64,${data.left_epipolar}`);
            setRightEpipolarImage(`data:image/png;base64,${data.right_epipolar}`);
            setPoints3D(data.points_3d);
            console.log('Points 3D:', data.points_3d);
            console.log('Number of points:', data.points_3d.length);
        } catch (error) {
            console.error('Error reconstructing:', error);
            alert('Failed to reconstruct');
        }
    };

    // Improved PointCloud component with small discrete points
    const PointCloud = ({ points }) => {
        const pointsRef = useRef();
        
        useEffect(() => {
            if (pointsRef.current && points && points.length > 0) {
                console.log("Rendering colored point cloud with", points.length, "points");
                
                // Create positions and colors arrays
                const positions = new Float32Array(points.length * 3);
                const colors = new Float32Array(points.length * 3);
                
                // Populate arrays with position and color data
                for (let i = 0; i < points.length; i++) {
                    // Position (first 3 values are x,y,z)
                    positions[i * 3] = points[i][0];     // x
                    positions[i * 3 + 1] = points[i][1]; // y
                    positions[i * 3 + 2] = points[i][2]; // z
                    
                    // Color (next 3 values are r,g,b)
                    colors[i * 3] = points[i][3] / 255.0;     // r
                    colors[i * 3 + 1] = points[i][4] / 255.0; // g
                    colors[i * 3 + 2] = points[i][5] / 255.0; // b
                }
                
                // Update geometry
                pointsRef.current.geometry.setAttribute(
                    'position', 
                    new THREE.BufferAttribute(positions, 3)
                );
                
                pointsRef.current.geometry.setAttribute(
                    'color',
                    new THREE.BufferAttribute(colors, 3)
                );
                
                pointsRef.current.geometry.computeBoundingSphere();
            }
        }, [points]);
        
        return (
            <points ref={pointsRef}>
                <bufferGeometry>
                    <bufferAttribute
                        attach="attributes-position"
                        count={points.length}
                        array={new Float32Array(points.length * 3)}
                        itemSize={3}
                    />
                    <bufferAttribute
                        attach="attributes-color"
                        count={points.length}
                        array={new Float32Array(points.length * 3)}
                        itemSize={3}
                    />
                </bufferGeometry>
                <pointsMaterial
                    size={0.05}
                    sizeAttenuation={true}
                    vertexColors={true}
                    transparent={false}
                />
            </points>
        );
    };

    return (
        <div className="bg-blue-50 p-4 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4 text-indigo-800">Part B: 3D Reconstruction</h2>
            
            {/* Input controls */}
            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700">Left Image:</label>
                    <input type="file" accept="image/*,.pgm" ref={leftInputRef} className="mb-2" />
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700">Right Image:</label>
                    <input type="file" accept="image/*,.pgm" ref={rightInputRef} className="mb-2" />
                </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700">Number of Disparities:</label>
                    <select 
                        value={numDisparities} 
                        onChange={(e) => setNumDisparities(parseInt(e.target.value))}
                        className="block w-full mt-1 rounded-md border-gray-300 shadow-sm"
                    >
                        <option value="16">16</option>
                        <option value="32">32</option>
                        <option value="64">64</option>
                        <option value="128">128</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-700">Method:</label>
                    <select 
                        value={method} 
                        onChange={(e) => setMethod(e.target.value)}
                        className="block w-full mt-1 rounded-md border-gray-300 shadow-sm"
                    >
                        <option value="StereoBM">StereoBM</option>
                        <option value="StereoSGBM">StereoSGBM</option>
                    </select>
                </div>
            </div>
            
            <button 
                onClick={handleUpload}
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
            >
                Reconstruct
            </button>
            
            <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Disparity Map</h3>
                    {disparityImage && <img src={disparityImage} alt="Disparity" className="max-w-full h-auto" />}
                </div>
                <div>
                    <h3 className="text-lg font-medium text-gray-700">3D Point Cloud</h3>
                    {points3D.length > 0 ? (
                        <Canvas
                            camera={{ position: [10, 5, 10], fov: 50 }}
                            style={{ height: '400px', backgroundColor: '#222' }}
                        >
                            <ambientLight intensity={0.8} />
                            <pointLight position={[10, 10, 10]} intensity={1} />
                            <PointCloud points={points3D} />
                            <OrbitControls 
                                enableDamping 
                                dampingFactor={0.25}
                                rotateSpeed={0.5}
                                zoomSpeed={0.8}
                            />
                            <gridHelper args={[20, 20, 0x444444, 0x222222]} />
                            <axesHelper args={[5]} />
                        </Canvas>
                    ) : (
                        <p className="text-red-500">No 3D points to render</p>
                    )}
                </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Left Image with Epipolar Lines</h3>
                    {leftEpipolarImage && <img src={leftEpipolarImage} alt="Left Epipolar" className="max-w-full h-auto" />}
                </div>
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Right Image with Epipolar Lines</h3>
                    {rightEpipolarImage && <img src={rightEpipolarImage} alt="Right Epipolar" className="max-w-full h-auto" />}
                </div>
            </div>
        </div>
    );
}

export default ThreeDReconstruction;