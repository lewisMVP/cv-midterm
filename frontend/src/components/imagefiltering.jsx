import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Sparkles, Clock, BarChart3, Eye, Download, X, Info } from 'lucide-react';

function ImageFiltering() {
    const [originalImage, setOriginalImage] = useState(null);
    const [noisyImage, setNoisyImage] = useState(null);
    const [filteredImages, setFilteredImages] = useState({});
    const [psnrValues, setPsnrValues] = useState({});
    const [edgePreservation, setEdgePreservation] = useState({});
    const [ssimValues, setSsimValues] = useState({});
    const [computationTimes, setComputationTimes] = useState({});
    const [isProcessing, setIsProcessing] = useState(false);
    const [selectedFilter, setSelectedFilter] = useState(null);
    const fileInputRef = useRef(null);

    const handleImageUpload = async (e) => {
        const file = e.target.files[0];
        if (file) {
            setOriginalImage(URL.createObjectURL(file));
            setNoisyImage(null);
            setFilteredImages({});
            setPsnrValues({});
            setEdgePreservation({});
            setSelectedFilter(null);
        }
    };

    const applyFilters = async () => {
        if (!originalImage) {
            alert('Please upload an image first!');
            return;
        }

        setIsProcessing(true);
        const formData = new FormData();
        const file = fileInputRef.current.files[0];
        formData.append('image', file);

        try {
            const response = await fetch('http://localhost:5000/filter', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }
            setOriginalImage(`data:image/png;base64,${data.original}`);
            setNoisyImage(`data:image/png;base64,${data.grayscale}`);
            setFilteredImages(data.filtered);
            setPsnrValues(data.psnr || {});
            setSsimValues(data.ssim || {});
            setComputationTimes(data.computation_time || {});
            setEdgePreservation(data.edge_preservation || {});
        } catch (error) {
            console.error('Error applying filters:', error);
            alert('Failed to apply filters');
        } finally {
            setIsProcessing(false);
        }
    };

    const filterInfo = {
        mean: { 
            name: 'Mean Filter', 
            icon: '◉', 
            color: 'from-blue-500 to-cyan-500',
            description: 'Simple averaging filter for noise reduction'
        },
        gaussian: { 
            name: 'Gaussian Filter', 
            icon: '◈', 
            color: 'from-purple-500 to-pink-500',
            description: 'Weighted averaging with Gaussian kernel'
        },
        median: { 
            name: 'Median Filter', 
            icon: '◊', 
            color: 'from-green-500 to-emerald-500',
            description: 'Non-linear filter excellent for salt & pepper noise'
        },
        laplacian: { 
            name: 'Laplacian Sharpening', 
            icon: '◆', 
            color: 'from-orange-500 to-red-500',
            description: 'Edge enhancement and sharpening filter'
        }
    };

    const formatMetric = (value, suffix = '') => {
        if (typeof value === 'number') {
            return value.toFixed(2) + suffix;
        }
        return value + suffix;
    };

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
                            Image Filtering
                        </h2>
                        <p className="text-body text-apple-gray-600">
                            Apply traditional image filtering techniques to reduce noise and enhance quality
                        </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="badge-apple badge-apple-blue">
                            <Sparkles className="w-3 h-3" />
                            Traditional CV
                        </span>
                    </div>
                </div>

                {/* Upload Area */}
                <div className="upload-area group" onClick={() => fileInputRef.current?.click()}>
            <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleImageUpload}
                        className="hidden"
                    />
                    <div className="flex flex-col items-center justify-center py-8">
                        <motion.div
                            animate={{ y: [0, -10, 0] }}
                            transition={{ duration: 2, repeat: Infinity }}
                        >
                            <Upload className="w-12 h-12 text-apple-blue mb-4" />
                        </motion.div>
                        <p className="text-title-3 font-medium text-apple-gray-900 mb-2">
                            Drop your image here
                        </p>
                        <p className="text-callout text-apple-gray-600">
                            or click to browse
                        </p>
                        <p className="text-footnote text-apple-gray-500 mt-2">
                            Supports JPG, PNG, WebP up to 10MB
                        </p>
                    </div>
                </div>

                {originalImage && (
                    <motion.button
                    onClick={applyFilters}
                        disabled={isProcessing}
                        className="btn-apple w-full mt-6 flex items-center justify-center gap-2"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        {isProcessing ? (
                            <>
                                <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                >
                                    <Sparkles className="w-4 h-4" />
                                </motion.div>
                                Processing...
                            </>
                        ) : (
                            <>
                                <Sparkles className="w-4 h-4" />
                    Apply Filters
                            </>
                        )}
                    </motion.button>
                )}
            </div>
            
            {/* Original and Grayscale Images */}
            <AnimatePresence>
                {originalImage && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="grid grid-cols-1 md:grid-cols-2 gap-6"
                    >
                        <div className="card-apple p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-title-3 font-medium">Original Image</h3>
                                <span className="badge-apple badge-apple-green">
                                    <Eye className="w-3 h-3" />
                                    Source
                                </span>
                </div>
                            <div className="image-preview">
                                <img src={originalImage} alt="Original" />
                </div>
            </div>
            
                        {noisyImage && (
                            <div className="card-apple p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-title-3 font-medium">Grayscale Image</h3>
                                    <span className="badge-apple bg-gray-100 text-gray-700">
                                        Preprocessed
                                    </span>
                                </div>
                                <div className="image-preview">
                                    <img src={noisyImage} alt="Grayscale" />
                            </div>
                        </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Filtered Results */}
            {Object.keys(filteredImages).length > 0 && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-title-2 font-semibold">Filter Results</h3>
                        <div className="flex gap-2">
                            {Object.keys(filteredImages).map((filterType) => (
                                <motion.button
                                    key={filterType}
                                    onClick={() => setSelectedFilter(filterType === selectedFilter ? null : filterType)}
                                    className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                                        selectedFilter === filterType
                                            ? 'bg-apple-blue text-white'
                                            : 'bg-apple-gray-100 text-apple-gray-700 hover:bg-apple-gray-200'
                                    }`}
                                    whileHover={{ scale: 1.05 }}
                                    whileTap={{ scale: 0.95 }}
                                >
                                    {filterInfo[filterType]?.icon} {filterInfo[filterType]?.name}
                                </motion.button>
                    ))}
                </div>
            </div>
            
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {Object.entries(filteredImages).map(([filterType, imgSrc], index) => (
                            <motion.div
                                key={filterType}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className={`card-apple p-6 hover-glow transition-all duration-300 ${
                                    selectedFilter === filterType ? 'ring-2 ring-apple-blue' : ''
                                }`}
                            >
                                <div className="flex items-start justify-between mb-4">
                                    <div>
                                        <h4 className="text-title-3 font-medium flex items-center gap-2">
                                            <span className={`text-xl bg-gradient-to-r ${filterInfo[filterType]?.color} bg-clip-text text-transparent`}>
                                                {filterInfo[filterType]?.icon}
                                            </span>
                                            {filterInfo[filterType]?.name}
                                        </h4>
                                        <p className="text-footnote text-apple-gray-600 mt-1">
                                            {filterInfo[filterType]?.description}
                                        </p>
                                    </div>
                                    <button className="text-apple-gray-400 hover:text-apple-blue transition-colors">
                                        <Download className="w-4 h-4" />
                                    </button>
                                </div>

                                <div className="image-preview mb-4">
                                    <img src={`data:image/png;base64,${imgSrc}`} alt={filterType} />
                                </div>

                                {/* Metrics Grid */}
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="metric-card">
                                        <div className="flex items-center gap-2 mb-1">
                                            <BarChart3 className="w-3 h-3 text-apple-blue" />
                                            <span className="text-caption text-apple-gray-600">PSNR</span>
                                        </div>
                                        <p className="text-callout font-semibold text-apple-gray-900">
                                            {formatMetric(psnrValues[filterType], ' dB')}
                                        </p>
                                    </div>
                                    <div className="metric-card">
                                        <div className="flex items-center gap-2 mb-1">
                                            <Eye className="w-3 h-3 text-apple-purple" />
                                            <span className="text-caption text-apple-gray-600">SSIM</span>
                                        </div>
                                        <p className="text-callout font-semibold text-apple-gray-900">
                                            {formatMetric(ssimValues[filterType])}
                                        </p>
                                    </div>
                                    <div className="metric-card">
                                        <div className="flex items-center gap-2 mb-1">
                                            <Clock className="w-3 h-3 text-apple-green" />
                                            <span className="text-caption text-apple-gray-600">Time</span>
                                        </div>
                                        <p className="text-callout font-semibold text-apple-gray-900">
                                            {formatMetric(computationTimes[filterType] * 1000, ' ms')}
                                        </p>
                                    </div>
                                    <div className="metric-card">
                                        <div className="flex items-center gap-2 mb-1">
                                            <Info className="w-3 h-3 text-apple-pink" />
                                            <span className="text-caption text-apple-gray-600">Edge</span>
                                        </div>
                                        <p className="text-callout font-semibold text-apple-gray-900">
                                            {formatMetric(edgePreservation[filterType] || 0)}
                                        </p>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    {/* Comparison Table */}
            {Object.keys(psnrValues).length > 0 && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="card-apple p-6 mt-8"
                        >
                            <h3 className="text-title-3 font-semibold mb-6">Performance Comparison</h3>
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="border-b border-apple-gray-200">
                                            <th className="text-left py-3 px-4 text-caption font-medium text-apple-gray-600 uppercase tracking-wider">
                                                Filter
                                            </th>
                                            <th className="text-center py-3 px-4 text-caption font-medium text-apple-gray-600 uppercase tracking-wider">
                                                PSNR ↑
                                            </th>
                                            <th className="text-center py-3 px-4 text-caption font-medium text-apple-gray-600 uppercase tracking-wider">
                                                SSIM ↑
                                            </th>
                                            <th className="text-center py-3 px-4 text-caption font-medium text-apple-gray-600 uppercase tracking-wider">
                                                Edge ↑
                                            </th>
                                            <th className="text-center py-3 px-4 text-caption font-medium text-apple-gray-600 uppercase tracking-wider">
                                                Time ↓
                                            </th>
                                </tr>
                            </thead>
                                    <tbody>
                                        {Object.keys(psnrValues).map((filter, index) => {
                                            const isOptimal = Math.max(...Object.values(psnrValues)) === psnrValues[filter];
                                            return (
                                                <tr key={filter} className="border-b border-apple-gray-100 hover:bg-apple-gray-50 transition-colors">
                                                    <td className="py-3 px-4">
                                                        <div className="flex items-center gap-2">
                                                            <span className={`text-lg bg-gradient-to-r ${filterInfo[filter]?.color} bg-clip-text text-transparent`}>
                                                                {filterInfo[filter]?.icon}
                                                            </span>
                                                            <span className="text-callout font-medium">
                                                                {filterInfo[filter]?.name}
                                                            </span>
                                                            {isOptimal && (
                                                                <span className="badge-apple badge-apple-green text-xs">
                                                                    Best
                                                                </span>
                                                            )}
                                                        </div>
                                                    </td>
                                                    <td className="py-3 px-4 text-center">
                                                        <span className={`text-callout font-semibold ${isOptimal ? 'text-apple-green' : 'text-apple-gray-900'}`}>
                                                            {formatMetric(psnrValues[filter])}
                                                        </span>
                                                    </td>
                                                    <td className="py-3 px-4 text-center">
                                                        <span className="text-callout">
                                                            {formatMetric(ssimValues[filter])}
                                                        </span>
                                                    </td>
                                                    <td className="py-3 px-4 text-center">
                                                        <span className="text-callout">
                                                            {formatMetric(edgePreservation[filter] || 0)}
                                                        </span>
                                                    </td>
                                                    <td className="py-3 px-4 text-center">
                                                        <span className="text-callout">
                                                            {formatMetric(computationTimes[filter] * 1000)}ms
                                                        </span>
                                                    </td>
                                    </tr>
                                            );
                                        })}
                            </tbody>
                        </table>
                    </div>
                        </motion.div>
                    )}
                </motion.div>
            )}
        </motion.div>
    );
}

export default ImageFiltering;