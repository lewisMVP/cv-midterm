import { useState, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { Points, PointMaterial, OrbitControls } from '@react-three/drei';

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
                        <Canvas camera={{ position: [0, 0, 50], fov: 75 }} style={{ height: '300px' }}>
                            <Points>
                                <bufferGeometry>
                                    <bufferAttribute
                                        attach="attributes-position"
                                        array={new Float32Array(points3D.flat())}
                                        itemSize={3}
                                        count={points3D.length}
                                    />
                                </bufferGeometry>
                                <PointMaterial size={0.05} color="green" />
                            </Points>
                            <ambientLight intensity={0.5} />
                            <OrbitControls />
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