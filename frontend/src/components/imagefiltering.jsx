import { useState, useRef } from 'react';

function ImageFiltering() {
    const [originalImage, setOriginalImage] = useState(null);
    const [noisyImage, setNoisyImage] = useState(null);
    const [filteredImages, setFilteredImages] = useState({});
    const [psnrValues, setPsnrValues] = useState({});
    const [edgePreservation, setEdgePreservation] = useState({});
    const [ssimValues, setSsimValues] = useState({});
    const [computationTimes, setComputationTimes] = useState({});
    const fileInputRef = useRef(null);

    const handleImageUpload = async (e) => {
        const file = e.target.files[0];
        if (file) {
            setOriginalImage(URL.createObjectURL(file));
            setNoisyImage(null);
            setFilteredImages({});
            setPsnrValues({});
            setEdgePreservation({});
        }
    };

    const applyFilters = async () => {
        if (!originalImage) {
            alert('Please upload an image first!');
            return;
        }

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
            setNoisyImage(`data:image/png;base64,${data.noisy}`);
            setFilteredImages(data.filtered);
            setPsnrValues(data.psnr || {});
            setSsimValues(data.ssim || {});
            setComputationTimes(data.computation_time || {});
            setEdgePreservation(data.edge_preservation || {});
        } catch (error) {
            console.error('Error applying filters:', error);
            alert('Failed to apply filters');
        }
    };

    return (
        <div className="bg-blue-50 p-4 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4 text-indigo-800">Part A: Image Filtering</h2>
            <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleImageUpload}
                className="mb-4"
            />
            <div>
                <button 
                    onClick={applyFilters}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                >
                    Apply Filters
                </button>
            </div>
            
            {/* Original and Noisy Images */}
            <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Original Image</h3>
                    {originalImage && <img src={originalImage} alt="Original" className="max-w-full h-auto border" />}
                </div>
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Noisy Image</h3>
                    {noisyImage && <img src={noisyImage} alt="Noisy" className="max-w-full h-auto border" />}
                </div>
            </div>
            
            {/* Filtered Images */}
            <div className="mt-6">
                <h3 className="text-xl font-medium text-gray-800">Filtered Images</h3>
                <div className="grid grid-cols-2 md:grid-cols-2 gap-4 mt-2">
                    {Object.entries(filteredImages).map(([filterType, imgSrc]) => (
                        <div key={filterType} className="border p-2 rounded">
                            <p className="text-gray-600 font-medium">{formatFilterName(filterType)}</p>
                            <img src={`data:image/png;base64,${imgSrc}`} alt={filterType} className="max-w-full h-auto" />
                            <div className="mt-2 text-sm">
                                <p className="text-gray-600">PSNR: {psnrValues[filterType]?.toFixed(2)} dB</p>
                                <p className="text-gray-600">SSIM: {ssimValues[filterType]?.toFixed(3)}</p>
                                <p className="text-gray-600">Time: {(computationTimes[filterType] * 1000).toFixed(2)} ms</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
            
            {/* Filter Comparison */}
            {Object.keys(psnrValues).length > 0 && (
                <div className="mt-6">
                    <h3 className="text-xl font-medium text-gray-800">Filter Comparison</h3>
                    <div className="overflow-x-auto mt-2">
                        <table className="min-w-full border divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-4 py-2 text-left text-sm font-medium text-gray-500 uppercase tracking-wider">Filter</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium text-gray-500 uppercase tracking-wider">PSNR (dB)</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium text-gray-500 uppercase tracking-wider">SSIM</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium text-gray-500 uppercase tracking-wider">Edge Preservation</th>
                                    <th className="px-4 py-2 text-left text-sm font-medium text-gray-500 uppercase tracking-wider">Time (ms)</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {Object.keys(psnrValues).map((filter) => (
                                    <tr key={filter}>
                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">{formatFilterName(filter)}</td>
                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">{psnrValues[filter]?.toFixed(2)}</td>
                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">{ssimValues[filter]?.toFixed(3)}</td>
                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">{(edgePreservation[filter] || 0).toFixed(3)}</td>
                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">{(computationTimes[filter] * 1000).toFixed(2)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}

// Helper function to format filter names
function formatFilterName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

export default ImageFiltering;