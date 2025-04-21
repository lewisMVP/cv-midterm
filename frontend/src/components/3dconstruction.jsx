import { useState, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { Points, PointMaterial, OrbitControls } from '@react-three/drei';

function ThreeDReconstruction() {
  const [disparityImage, setDisparityImage] = useState(null);
  const [epipolarImage, setEpipolarImage] = useState(null);
  const [points3D, setPoints3D] = useState([]);
  const leftInputRef = useRef(null);
  const rightInputRef = useRef(null);

  const handleUpload = async () => {
    if (!leftInputRef.current.files[0] || !rightInputRef.current.files[0]) {
      alert('Please upload both left and right images!');
      return;
    }

    const formData = new FormData();
    formData.append('left_image', leftInputRef.current.files[0]);
    formData.append('right_image', rightInputRef.current.files[0]);

    try {
      const response = await fetch('https://cv-midterm.onrender.com', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      setDisparityImage(`data:image/png;base64,${data.disparity}`);
      setEpipolarImage(`data:image/png;base64,${data.left_epipolar}`);
      setPoints3D(data.points_3d);
    } catch (error) {
      console.error('Error reconstructing:', error);
      alert('Failed to reconstruct');
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Part B: 3D Reconstruction</h2>
      <input type="file" accept="image/*" ref={leftInputRef} className="mb-2" />
      <input type="file" accept="image/*" ref={rightInputRef} className="mb-2" />
      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Reconstruct
      </button>
      <div className="grid grid-cols-2 gap-2 mt-4">
        <div>
          <h3 className="text-lg font-medium">Disparity Map</h3>
          {disparityImage && <img src={disparityImage} alt="Disparity" className="max-w-full h-auto" />}
        </div>
        <div>
          <h3 className="text-lg font-medium">3D Point Cloud</h3>
          {points3D.length > 0 && (
            <Canvas camera={{ position: [0, 0, 5] }} style={{ height: '300px' }}>
              <Points>
                <bufferGeometry>
                  <bufferAttribute
                    attach="attributes-position"
                    array={new Float32Array(points3D.flat())}
                    itemSize={3}
                    count={points3D.length}
                  />
                </bufferGeometry>
                <PointMaterial size={0.01} color="green" />
              </Points>
              <ambientLight />
              <OrbitControls />
            </Canvas>
          )}
        </div>
      </div>
      <div className="mt-4">
        <h3 className="text-lg font-medium">Left Image with Epipolar Lines</h3>
        {epipolarImage && <img src={epipolarImage} alt="Epipolar" className="max-w-full h-auto" />}
      </div>
    </div>
  );
}

export default ThreeDReconstruction;