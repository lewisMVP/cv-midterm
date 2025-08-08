import { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Stats } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Layers, Settings, Maximize2, Download, Play, Pause, RotateCcw, Info, Eye, Cpu } from 'lucide-react';
import * as THREE from 'three';

function ThreeDReconstruction() {
    const [disparityImage, setDisparityImage] = useState(null);
    const [leftEpipolarImage, setLeftEpipolarImage] = useState(null);
    const [rightEpipolarImage, setRightEpipolarImage] = useState(null);
    const [points3D, setPoints3D] = useState([]);
    const [numDisparities, setNumDisparities] = useState(64);
    const [method, setMethod] = useState('StereoBM');
    const [isProcessing, setIsProcessing] = useState(false);
    const [autoRotate, setAutoRotate] = useState(false);
    const [showGrid, setShowGrid] = useState(true);
    const [showStats, setShowStats] = useState(false);
    const [viewMode, setViewMode] = useState('disparity'); // 'disparity', '3d', 'epipolar'
    const leftInputRef = useRef(null);
    const rightInputRef = useRef(null);
    const [leftPreview, setLeftPreview] = useState(null);
    const [rightPreview, setRightPreview] = useState(null);

    const handleLeftImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setLeftPreview(URL.createObjectURL(file));
        }
    };

    const handleRightImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setRightPreview(URL.createObjectURL(file));
        }
    };

    const handleUpload = async () => {
        if (!leftInputRef.current.files[0] || !rightInputRef.current.files[0]) {
            alert('Please upload both left and right images!');
            return;
        }

        setIsProcessing(true);
        const formData = new FormData();
        formData.append('left_image', leftInputRef.current.files[0]);
        formData.append('right_image', rightInputRef.current.files[0]);
        formData.append('num_disparities', numDisparities);
        formData.append('method', method);

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
            setViewMode('disparity');
        } catch (error) {
            console.error('Error reconstructing:', error);
            alert('Failed to reconstruct');
        } finally {
            setIsProcessing(false);
        }
    };

    const PointCloud = ({ points }) => {
        const pointsRef = useRef();
        
        useEffect(() => {
            if (pointsRef.current && points && points.length > 0) {
                const positions = new Float32Array(points.length * 3);
                const colors = new Float32Array(points.length * 3);
                
                for (let i = 0; i < points.length; i++) {
                    positions[i * 3] = points[i][0];
                    positions[i * 3 + 1] = points[i][1];
                    positions[i * 3 + 2] = points[i][2];
                    
                    colors[i * 3] = points[i][3] / 255.0;
                    colors[i * 3 + 1] = points[i][4] / 255.0;
                    colors[i * 3 + 2] = points[i][5] / 255.0;
                }
                
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

    const viewModes = [
        { id: 'disparity', label: 'Disparity Map', icon: Layers },
        { id: '3d', label: '3D Point Cloud', icon: Maximize2 },
        { id: 'epipolar', label: 'Epipolar Lines', icon: Eye },
    ];

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-8"
        >
            {/* Header Card */}
            <div className="card-apple p-8">
                <div className="flex items-start justify-between mb-6">
                    <div>
                        <h2 className="text-title-2 font-semibold text-apple-gray-900 mb-2">
                            3D Reconstruction
                        </h2>
                        <p className="text-body text-apple-gray-600">
                            Generate 3D point clouds from stereo image pairs using traditional stereo matching algorithms
                        </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="badge-apple badge-apple-blue">
                            <Layers className="w-3 h-3" />
                            Stereo Vision
                        </span>
                    </div>
                </div>

                {/* Image Upload Section */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                        <label className="block text-callout font-medium text-apple-gray-700 mb-3">
                            Left Image
                        </label>
                        <div 
                            className="upload-area h-48 cursor-pointer group"
                            onClick={() => leftInputRef.current?.click()}
                        >
                            <input
                                type="file"
                                accept="image/*,.pgm"
                                ref={leftInputRef}
                                onChange={handleLeftImageUpload}
                                className="hidden"
                            />
                            {leftPreview ? (
                                <img src={leftPreview} alt="Left" className="w-full h-full object-cover rounded-apple-sm" />
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full">
                                    <Upload className="w-8 h-8 text-apple-blue mb-2 group-hover:scale-110 transition-transform" />
                                    <p className="text-callout text-apple-gray-600">Upload Left Image</p>
                                </div>
                            )}
                        </div>
                    </div>

                    <div>
                        <label className="block text-callout font-medium text-apple-gray-700 mb-3">
                            Right Image
                        </label>
                        <div 
                            className="upload-area h-48 cursor-pointer group"
                            onClick={() => rightInputRef.current?.click()}
                        >
                            <input
                                type="file"
                                accept="image/*,.pgm"
                                ref={rightInputRef}
                                onChange={handleRightImageUpload}
                                className="hidden"
                            />
                            {rightPreview ? (
                                <img src={rightPreview} alt="Right" className="w-full h-full object-cover rounded-apple-sm" />
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full">
                                    <Upload className="w-8 h-8 text-apple-blue mb-2 group-hover:scale-110 transition-transform" />
                                    <p className="text-callout text-apple-gray-600">Upload Right Image</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Settings Section */}
                <div className="bg-apple-gray-50 rounded-apple p-6 mb-6">
                    <div className="flex items-center gap-2 mb-4">
                        <Settings className="w-4 h-4 text-apple-gray-600" />
                        <h3 className="text-callout font-medium text-apple-gray-700">
                            Reconstruction Settings
                        </h3>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="block text-footnote text-apple-gray-600 mb-2">
                                Number of Disparities
                            </label>
                            <select 
                                value={numDisparities} 
                                onChange={(e) => setNumDisparities(parseInt(e.target.value))}
                                className="input-apple"
                            >
                                <option value="16">16 (Fast, Low Quality)</option>
                                <option value="32">32 (Balanced)</option>
                                <option value="64">64 (Good Quality)</option>
                                <option value="128">128 (Best Quality)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-footnote text-apple-gray-600 mb-2">
                                Stereo Method
                            </label>
                            <select 
                                value={method} 
                                onChange={(e) => setMethod(e.target.value)}
                                className="input-apple"
                            >
                                <option value="StereoBM">StereoBM (Block Matching)</option>
                                <option value="StereoSGBM">StereoSGBM (Semi-Global)</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Action Button */}
                <motion.button
                    onClick={handleUpload}
                    disabled={isProcessing || !leftPreview || !rightPreview}
                    className="btn-apple w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    {isProcessing ? (
                        <>
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                            >
                                <Cpu className="w-4 h-4" />
                            </motion.div>
                            Processing Stereo Matching...
                        </>
                    ) : (
                        <>
                            <Layers className="w-4 h-4" />
                            Start 3D Reconstruction
                        </>
                    )}
                </motion.button>
            </div>

            {/* Results Section */}
            <AnimatePresence>
                {disparityImage && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="card-apple p-8"
                    >
                        {/* View Mode Tabs */}
                        <div className="flex justify-center mb-8">
                            <div className="tab-nav inline-flex">
                                {viewModes.map((mode) => {
                                    const Icon = mode.icon;
                                    return (
                                        <motion.button
                                            key={mode.id}
                                            onClick={() => setViewMode(mode.id)}
                                            className={`tab-button ${viewMode === mode.id ? 'active' : ''}`}
                                            whileHover={{ scale: 1.02 }}
                                            whileTap={{ scale: 0.98 }}
                                        >
                                            <span className="flex items-center gap-2">
                                                <Icon className="w-4 h-4" />
                                                {mode.label}
                                            </span>
                                        </motion.button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Content based on view mode */}
                        <AnimatePresence mode="wait">
                            {/* Disparity Map View */}
                            {viewMode === 'disparity' && (
                                <motion.div
                                    key="disparity"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="space-y-6"
                                >
                                    <div className="text-center">
                                        <h3 className="text-title-3 font-semibold mb-2">Disparity Map</h3>
                                        <p className="text-callout text-apple-gray-600 mb-6">
                                            Depth representation where brighter regions are closer to the camera
                                        </p>
                                        <div className="relative inline-block">
                                            <img 
                                                src={disparityImage} 
                                                alt="Disparity Map" 
                                                className="max-w-full h-auto rounded-apple shadow-apple-lg"
                                            />
                                            <div className="absolute top-4 right-4">
                                                <span className="badge-apple bg-white/90 backdrop-blur text-apple-gray-700">
                                                    <Info className="w-3 h-3" />
                                                    {method} | {numDisparities} disparities
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Disparity Info Cards */}
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        <div className="metric-card">
                                            <div className="flex items-center gap-2 mb-2">
                                                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                                                    <Layers className="w-4 h-4 text-white" />
                                                </div>
                                                <span className="text-callout font-medium">Method</span>
                                            </div>
                                            <p className="text-title-3 font-semibold text-apple-gray-900">{method}</p>
                                        </div>
                                        <div className="metric-card">
                                            <div className="flex items-center gap-2 mb-2">
                                                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                                                    <Settings className="w-4 h-4 text-white" />
                                                </div>
                                                <span className="text-callout font-medium">Disparities</span>
                                            </div>
                                            <p className="text-title-3 font-semibold text-apple-gray-900">{numDisparities}</p>
                                        </div>
                                        <div className="metric-card">
                                            <div className="flex items-center gap-2 mb-2">
                                                <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
                                                    <Cpu className="w-4 h-4 text-white" />
                                                </div>
                                                <span className="text-callout font-medium">Points</span>
                                            </div>
                                            <p className="text-title-3 font-semibold text-apple-gray-900">{points3D.length.toLocaleString()}</p>
                                        </div>
                                    </div>
                                </motion.div>
                            )}

                            {/* 3D Point Cloud View */}
                            {viewMode === '3d' && (
                                <motion.div
                                    key="3d"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                >
                                    <div className="text-center mb-6">
                                        <h3 className="text-title-3 font-semibold mb-2">3D Point Cloud</h3>
                                        <p className="text-callout text-apple-gray-600">
                                            Interactive 3D visualization with {points3D.length.toLocaleString()} points
                                        </p>
                                    </div>

                                    {/* 3D Controls */}
                                    <div className="flex justify-center gap-2 mb-4">
                                        <motion.button
                                            onClick={() => setAutoRotate(!autoRotate)}
                                            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                                                autoRotate ? 'bg-apple-blue text-white' : 'bg-apple-gray-100 text-apple-gray-700'
                                            }`}
                                            whileHover={{ scale: 1.05 }}
                                            whileTap={{ scale: 0.95 }}
                                        >
                                            {autoRotate ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                                        </motion.button>
                                        <motion.button
                                            onClick={() => setShowGrid(!showGrid)}
                                            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                                                showGrid ? 'bg-apple-blue text-white' : 'bg-apple-gray-100 text-apple-gray-700'
                                            }`}
                                            whileHover={{ scale: 1.05 }}
                                            whileTap={{ scale: 0.95 }}
                                        >
                                            Grid
                                        </motion.button>
                                        <motion.button
                                            onClick={() => setShowStats(!showStats)}
                                            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                                                showStats ? 'bg-apple-blue text-white' : 'bg-apple-gray-100 text-apple-gray-700'
                                            }`}
                                            whileHover={{ scale: 1.05 }}
                                            whileTap={{ scale: 0.95 }}
                                        >
                                            Stats
                                        </motion.button>
                                    </div>

                                    {/* 3D Viewer */}
                                    {points3D.length > 0 ? (
                                        <div className="viewer-3d">
                                            <div className="viewer-3d-inner">
                                                <Canvas
                                                    camera={{ position: [10, 5, 10], fov: 50 }}
                                                    style={{ background: 'linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%)' }}
                                                >
                                                    <ambientLight intensity={0.8} />
                                                    <pointLight position={[10, 10, 10]} intensity={1} />
                                                    <PointCloud points={points3D} />
                                                    <OrbitControls 
                                                        enableDamping 
                                                        dampingFactor={0.25}
                                                        rotateSpeed={0.5}
                                                        zoomSpeed={0.8}
                                                        autoRotate={autoRotate}
                                                        autoRotateSpeed={1}
                                                    />
                                                    {showGrid && (
                                                        <Grid 
                                                            args={[20, 20]} 
                                                            cellSize={1}
                                                            cellThickness={1}
                                                            cellColor={'#6f6f6f'}
                                                            sectionSize={5}
                                                            sectionThickness={1.5}
                                                            sectionColor={'#9d9d9d'}
                                                            fadeDistance={50}
                                                            fadeStrength={1}
                                                            followCamera={false}
                                                        />
                                                    )}
                                                    {showStats && <Stats />}
                                                </Canvas>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="flex items-center justify-center h-96 bg-apple-gray-100 rounded-apple">
                                            <p className="text-apple-gray-600">No 3D points generated</p>
                                        </div>
                                    )}
                                </motion.div>
                            )}

                            {/* Epipolar Lines View */}
                            {viewMode === 'epipolar' && (
                                <motion.div
                                    key="epipolar"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="space-y-6"
                                >
                                    <div className="text-center mb-6">
                                        <h3 className="text-title-3 font-semibold mb-2">Epipolar Geometry</h3>
                                        <p className="text-callout text-apple-gray-600">
                                            Visualization of corresponding epipolar lines between stereo pairs
                                        </p>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="space-y-3">
                                            <div className="flex items-center gap-2">
                                                <div className="w-3 h-3 bg-apple-blue rounded-full"></div>
                                                <span className="text-callout font-medium">Left Image Epipolar Lines</span>
                                            </div>
                                            {leftEpipolarImage && (
                                                <img 
                                                    src={leftEpipolarImage} 
                                                    alt="Left Epipolar" 
                                                    className="w-full h-auto rounded-apple shadow-apple"
                                                />
                                            )}
                                        </div>
                                        <div className="space-y-3">
                                            <div className="flex items-center gap-2">
                                                <div className="w-3 h-3 bg-apple-purple rounded-full"></div>
                                                <span className="text-callout font-medium">Right Image Epipolar Lines</span>
                                            </div>
                                            {rightEpipolarImage && (
                                                <img 
                                                    src={rightEpipolarImage} 
                                                    alt="Right Epipolar" 
                                                    className="w-full h-auto rounded-apple shadow-apple"
                                                />
                                            )}
                                        </div>
                                    </div>

                                    {/* Epipolar Info */}
                                    <div className="bg-apple-gray-50 rounded-apple p-6">
                                        <div className="flex items-start gap-3">
                                            <Info className="w-5 h-5 text-apple-blue mt-0.5" />
                                            <div>
                                                <h4 className="text-callout font-medium mb-2">About Epipolar Geometry</h4>
                                                <p className="text-footnote text-apple-gray-600">
                                                    The colored lines show the epipolar constraints between corresponding feature points. 
                                                    Points in one image must lie along the corresponding epipolar line in the other image, 
                                                    which helps validate the stereo calibration and matching quality.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}

export default ThreeDReconstruction;
