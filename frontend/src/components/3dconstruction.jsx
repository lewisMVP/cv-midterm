import { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { Points, PointMaterial, OrbitControls } from '@react-three/drei';
import * as THREE from 'three'; // Make sure Three.js is imported

// Component xử lý Part B: 3D Reconstruction
function ThreeDReconstruction() {
    // State lưu trữ kết quả từ backend
    const [disparityImage, setDisparityImage] = useState(null);
    const [leftEpipolarImage, setLeftEpipolarImage] = useState(null);
    const [rightEpipolarImage, setRightEpipolarImage] = useState(null);
    const [points3D, setPoints3D] = useState([]);
    const [numDisparities, setNumDisparities] = useState(64);
    const [method, setMethod] = useState('StereoBM');  // Thêm state để chọn phương pháp
    const leftInputRef = useRef(null);
    const rightInputRef = useRef(null);

    // Add this function to properly format points for Three.js
    const PointCloud = ({ points }) => {
        const pointsRef = useRef();
        
        useEffect(() => {
            if (pointsRef.current && points.length > 0) {
                // Log for debugging
                console.log("Updating point cloud with", points.length, "points");
                
                // Create a flat Float32Array from the points array
                const positions = new Float32Array(points.length * 3);
                for (let i = 0; i < points.length; i++) {
                    positions[i * 3] = points[i][0];     // x
                    positions[i * 3 + 1] = points[i][1]; // y
                    positions[i * 3 + 2] = points[i][2]; // z
                }
                
                // Update the buffer with the new positions
                pointsRef.current.geometry.setAttribute(
                    'position', 
                    new THREE.BufferAttribute(positions, 3)
                );
                
                // Update the geometry
                pointsRef.current.geometry.computeBoundingSphere();
            }
        }, [points]);
        
        return (
            <points ref={pointsRef}>
                <bufferGeometry>
                    <bufferAttribute
                        attach="attributes-position"
                        count={points.length}
                        array={new Float32Array(points.flat())}
                        itemSize={3}
                    />
                </bufferGeometry>
                <pointsMaterial 
                    size={0.5} 
                    color="green" 
                    sizeAttenuation={true} 
                    transparent={true}
                    alphaTest={0.5}
                />
            </points>
        );
    };

    // Xử lý khi người dùng upload ảnh và gọi API reconstruct
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

    // Debug output
    useEffect(() => {
      if (points3D && points3D.length > 0) {
        console.log("Points for rendering:", points3D.length);
        console.log("Sample points:", points3D.slice(0, 5));
      }
    }, [points3D]);

    return (
        <div className="bg-blue-50 p-4 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4 text-indigo-800">Part B: 3D Reconstruction</h2>
            <input type="file" accept="image/*,.pgm" ref={leftInputRef} className="mb-2" />
            <input type="file" accept="image/*,.pgm" ref={rightInputRef} className="mb-2" />
            <div className="mb-4">
                <label className="mr-2">Num Disparities:</label>
                <select value={numDisparities} onChange={(e) => setNumDisparities(e.target.value)}>
                    <option value="16">16</option>
                    <option value="32">32</option>
                    <option value="64">64</option>
                    <option value="128">128</option>
                </select>
            </div>
            <div className="mb-4">
                <label className="mr-2">Method:</label>
                <select value={method} onChange={(e) => setMethod(e.target.value)}>
                    <option value="StereoBM">StereoBM</option>
                    <option value="StereoSGBM">StereoSGBM</option>
                </select>
            </div>
            <button onClick={handleUpload}>Reconstruct</button>
            <div className="grid grid-cols-2 gap-2 mt-4">
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Disparity Map</h3>
                    {disparityImage && <img src={disparityImage} alt="Disparity" className="max-w-full h-auto" />}
                </div>
                <div>
                    <h3 className="text-lg font-medium text-gray-700">3D Point Cloud</h3>
                    {points3D.length > 0 ? (
                        <Canvas
                            camera={{ position: [0, 0, 50], far: 10000 }}
                            style={{ height: '400px', backgroundColor: '#111' }}
                        >
                            <ambientLight intensity={0.5} />
                            <pointLight position={[10, 10, 10]} />
                            <PointCloud points={points3D} />
                            <OrbitControls 
                                enableDamping 
                                dampingFactor={0.25}
                                rotateSpeed={0.5}
                                zoomSpeed={0.8}
                            />
                        </Canvas>
                    ) : (
                        <p className="text-red-500">No 3D points to render. The disparity map may contain too much noise, or the images need better rectification.</p>
                    )}
                </div>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
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