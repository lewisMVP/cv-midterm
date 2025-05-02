import { useState, useRef } from 'react';

// Component xử lý Part A: Image Filtering
function ImageFiltering() {
    // State lưu trữ các ảnh và giá trị PSNR
    const [originalImage, setOriginalImage] = useState(null);
    const [noisyImage, setNoisyImage] = useState(null);
    const [filteredImages, setFilteredImages] = useState({});
    const [psnrValues, setPsnrValues] = useState({});
    const fileInputRef = useRef(null);

    // Xử lý khi người dùng upload ảnh
    const handleImageUpload = async (e) => {
        const file = e.target.files[0];
        if (file) {
            setOriginalImage(URL.createObjectURL(file));
            setNoisyImage(null);
            setFilteredImages({});
            setPsnrValues({});
        }
    };

    // Áp dụng tất cả bộ lọc và lấy kết quả từ backend
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
            setPsnrValues(data.psnr);
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
                    Apply all Filters
                </button>
            </div>
            <div className="grid grid-cols-3 gap-2 mt-4">
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Original Image</h3>
                    {originalImage && <img src={originalImage} alt="Original" className="max-w-full h-auto" />}
                </div>
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Noisy Image</h3>
                    {noisyImage && <img src={noisyImage} alt="Noisy" className="max-w-full h-auto" />}
                </div>
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Filtered Images</h3>
                    {Object.entries(filteredImages).map(([filterType, imgSrc]) => (
                        <div key={filterType} className="mt-2">
                            <p className="text-gray-600">{filterType.charAt(0).toUpperCase() + filterType.slice(1)} Filter</p>
                            <img src={`data:image/png;base64,${imgSrc}`} alt={filterType} className="max-w-full h-auto" />
                            <p className="text-gray-600">PSNR: {psnrValues[filterType]?.toFixed(2)} dB</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default ImageFiltering;