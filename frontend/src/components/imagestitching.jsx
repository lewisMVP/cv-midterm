import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Grid3x3, Link2, Maximize, Download, X, Info, Eye, ImagePlus, Sparkles } from 'lucide-react';

function ImageStitching() {
    const [matchesImage, setMatchesImage] = useState(null);
    const [panorama, setPanorama] = useState(null);
    const [inliers, setInliers] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [uploadedImages, setUploadedImages] = useState([]);
    const [viewMode, setViewMode] = useState('panorama'); // 'panorama', 'matches'
    const fileInputRef = useRef(null);

    const handleImageUpload = (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            const newImages = [];
            for (let i = 0; i < files.length; i++) {
                newImages.push({
                    file: files[i],
                    preview: URL.createObjectURL(files[i]),
                    name: files[i].name
                });
            }
            setUploadedImages(newImages);
            // Reset results when new images are uploaded
            setMatchesImage(null);
            setPanorama(null);
            setInliers(null);
        }
    };

    const removeImage = (index) => {
        const newImages = [...uploadedImages];
        URL.revokeObjectURL(newImages[index].preview);
        newImages.splice(index, 1);
        setUploadedImages(newImages);
    };

    const handleUpload = async () => {
        const files = fileInputRef.current.files;
        if (files.length < 4) {
            alert('Please upload at least four images!');
            return;
        }

        setIsProcessing(true);
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append(`image_${i}`, files[i]);
        }

        try {
            const response = await fetch('http://localhost:5000/stitch', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }
            setMatchesImage(`data:image/png;base64,${data.matches}`);
            setPanorama(`data:image/png;base64,${data.panorama}`);
            setInliers(data.inliers);
            setViewMode('panorama');
        } catch (error) {
            console.error('Error stitching:', error);
            alert('Failed to stitch images');
        } finally {
            setIsProcessing(false);
        }
    };

    const getInlierQuality = (count) => {
        if (count > 50) return { label: 'Excellent', color: 'text-green-600', bg: 'bg-green-100' };
        if (count > 30) return { label: 'Good', color: 'text-blue-600', bg: 'bg-blue-100' };
        if (count > 15) return { label: 'Fair', color: 'text-yellow-600', bg: 'bg-yellow-100' };
        return { label: 'Poor', color: 'text-red-600', bg: 'bg-red-100' };
    };

    const viewModes = [
        { id: 'panorama', label: 'Panorama', icon: Maximize },
        { id: 'matches', label: 'Feature Matches', icon: Link2 },
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
                            Image Stitching
                        </h2>
                        <p className="text-body text-apple-gray-600">
                            Create panoramic images by detecting features and stitching multiple photos seamlessly
                        </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="badge-apple badge-apple-blue">
                            <Grid3x3 className="w-3 h-3" />
                            Panorama
                        </span>
                    </div>
                </div>

                {/* Upload Area */}
                <div 
                    className="upload-area group cursor-pointer relative"
                    onClick={() => fileInputRef.current?.click()}
                >
                    <input
                        type="file"
                        accept="image/*"
                        multiple
                        ref={fileInputRef}
                        onChange={handleImageUpload}
                        className="hidden"
                    />
                    
                    {uploadedImages.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-12">
                            <motion.div
                                animate={{ y: [0, -10, 0] }}
                                transition={{ duration: 2, repeat: Infinity }}
                            >
                                <ImagePlus className="w-12 h-12 text-apple-blue mb-4" />
                            </motion.div>
                            <p className="text-title-3 font-medium text-apple-gray-900 mb-2">
                                Drop your images here
                            </p>
                            <p className="text-callout text-apple-gray-600">
                                Select at least 4 images for panorama creation
                            </p>
                            <p className="text-footnote text-apple-gray-500 mt-2">
                                Supports JPG, PNG, WebP • Multiple selection enabled
                            </p>
                        </div>
                    ) : (
                        <div className="py-6">
                            <div className="flex items-center justify-between mb-4">
                                <p className="text-callout font-medium text-apple-gray-700">
                                    {uploadedImages.length} images selected
                                </p>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        fileInputRef.current?.click();
                                    }}
                                    className="text-apple-blue hover:text-apple-blue-hover text-sm font-medium"
                                >
                                    Change Images
                                </button>
                            </div>
                            
                            {/* Image Preview Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {uploadedImages.map((img, index) => (
                                    <motion.div
                                        key={index}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        transition={{ delay: index * 0.05 }}
                                        className="relative group"
                                    >
                                        <img
                                            src={img.preview}
                                            alt={`Upload ${index + 1}`}
                                            className="w-full h-24 object-cover rounded-apple-xs"
                                        />
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                removeImage(index);
                                            }}
                                            className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                                        >
                                            <X className="w-3 h-3" />
                                        </button>
                                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-2 rounded-b-apple-xs">
                                            <p className="text-white text-xs truncate">{img.name}</p>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Info Box */}
                {uploadedImages.length > 0 && uploadedImages.length < 4 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-apple p-4 mt-4">
                        <div className="flex items-start gap-3">
                            <Info className="w-5 h-5 text-yellow-600 mt-0.5" />
                            <div>
                                <p className="text-callout font-medium text-yellow-900">
                                    {4 - uploadedImages.length} more image{4 - uploadedImages.length > 1 ? 's' : ''} needed
                                </p>
                                <p className="text-footnote text-yellow-700 mt-1">
                                    A minimum of 4 images is required to create a panorama. For best results, use overlapping images taken from the same position.
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Action Button */}
                {uploadedImages.length >= 4 && (
                    <motion.button
                        onClick={handleUpload}
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
                                Creating Panorama...
                            </>
                        ) : (
                            <>
                                <Grid3x3 className="w-4 h-4" />
                                Stitch Images
                            </>
                        )}
                    </motion.button>
                )}
            </div>

            {/* Results Section */}
            <AnimatePresence>
                {panorama && (
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

                        <AnimatePresence mode="wait">
                            {/* Panorama View */}
                            {viewMode === 'panorama' && (
                                <motion.div
                                    key="panorama"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="space-y-6"
                                >
                                    <div className="text-center">
                                        <h3 className="text-title-3 font-semibold mb-2">Generated Panorama</h3>
                                        <p className="text-callout text-apple-gray-600 mb-6">
                                            Successfully stitched {uploadedImages.length} images into a seamless panorama
                                        </p>
                                    </div>

                                    <div className="relative group">
                                        <img 
                                            src={panorama} 
                                            alt="Panorama" 
                                            className="w-full h-auto rounded-apple shadow-apple-lg"
                                        />
                                        <button className="absolute top-4 right-4 p-2 bg-white/90 backdrop-blur rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-white">
                                            <Download className="w-5 h-5 text-apple-gray-700" />
                                        </button>
                                    </div>

                                    {/* Statistics */}
                                    {inliers && inliers.length > 0 && (
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            <div className="metric-card">
                                                <div className="flex items-center gap-2 mb-2">
                                                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                                                        <ImagePlus className="w-4 h-4 text-white" />
                                                    </div>
                                                    <span className="text-callout font-medium">Images Used</span>
                                                </div>
                                                <p className="text-title-3 font-semibold text-apple-gray-900">
                                                    {uploadedImages.length}
                                                </p>
                                            </div>
                                            <div className="metric-card">
                                                <div className="flex items-center gap-2 mb-2">
                                                    <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                                                        <Link2 className="w-4 h-4 text-white" />
                                                    </div>
                                                    <span className="text-callout font-medium">Total Matches</span>
                                                </div>
                                                <p className="text-title-3 font-semibold text-apple-gray-900">
                                                    {inliers.reduce((a, b) => a + b, 0)}
                                                </p>
                                            </div>
                                            <div className="metric-card">
                                                <div className="flex items-center gap-2 mb-2">
                                                    <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
                                                        <Eye className="w-4 h-4 text-white" />
                                                    </div>
                                                    <span className="text-callout font-medium">Avg Quality</span>
                                                </div>
                                                <p className="text-title-3 font-semibold text-apple-gray-900">
                                                    {getInlierQuality(Math.round(inliers.reduce((a, b) => a + b, 0) / inliers.length)).label}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            )}

                            {/* Feature Matches View */}
                            {viewMode === 'matches' && (
                                <motion.div
                                    key="matches"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="space-y-6"
                                >
                                    <div className="text-center">
                                        <h3 className="text-title-3 font-semibold mb-2">Feature Matching</h3>
                                        <p className="text-callout text-apple-gray-600 mb-6">
                                            Keypoint correspondences between consecutive image pairs
                                        </p>
                                    </div>

                                    {matchesImage && (
                                        <div className="space-y-6">
                                            <img 
                                                src={matchesImage} 
                                                alt="Feature Matches" 
                                                className="w-full h-auto rounded-apple shadow-apple"
                                            />

                                            {/* Inliers Detail */}
                                            {inliers && inliers.length > 0 && (
                                                <div className="bg-apple-gray-50 rounded-apple p-6">
                                                    <h4 className="text-callout font-medium mb-4">Match Quality Analysis</h4>
                                                    <div className="space-y-3">
                                                        {inliers.map((count, idx) => {
                                                            const quality = getInlierQuality(count);
                                                            return (
                                                                <div key={idx} className="flex items-center justify-between">
                                                                    <div className="flex items-center gap-3">
                                                                        <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center text-sm font-medium">
                                                                            {idx + 1}-{idx + 2}
                                                                        </div>
                                                                        <span className="text-callout">
                                                                            Image {idx + 1} ↔ Image {idx + 2}
                                                                        </span>
                                                                    </div>
                                                                    <div className="flex items-center gap-3">
                                                                        <span className="text-callout font-semibold">
                                                                            {count} inliers
                                                                        </span>
                                                                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${quality.bg} ${quality.color}`}>
                                                                            {quality.label}
                                                                        </span>
                                                                    </div>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Info Box */}
                                            <div className="bg-blue-50 border border-blue-200 rounded-apple p-6">
                                                <div className="flex items-start gap-3">
                                                    <Info className="w-5 h-5 text-blue-600 mt-0.5" />
                                                    <div>
                                                        <h4 className="text-callout font-medium text-blue-900 mb-2">
                                                            About Feature Matching
                                                        </h4>
                                                        <p className="text-footnote text-blue-700">
                                                            The lines connect corresponding keypoints detected in adjacent images. 
                                                            More inliers (correctly matched points) indicate better overlap and 
                                                            alignment quality between images, resulting in a smoother panorama.
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}

export default ImageStitching;