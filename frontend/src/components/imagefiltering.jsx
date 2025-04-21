import { useState, useRef } from 'react';

function ImageFiltering() {
  const [originalImage, setOriginalImage] = useState(null);
  const [filteredImage, setFilteredImage] = useState(null);
  const [psnr, setPsnr] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setOriginalImage(URL.createObjectURL(file));
      setFilteredImage(null);
      setPsnr(null);
    }
  };

  const applyFilter = async (filterType) => {
    if (!originalImage) {
      alert('Please upload an image first!');
      return;
    }

    const formData = new FormData();
    const file = fileInputRef.current.files[0];
    formData.append('image', file);
    formData.append('filter_type', filterType);

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
      setFilteredImage(`data:image/png;base64,${data.filtered}`);
      setPsnr(data.psnr);
    } catch (error) {
      console.error('Error applying filter:', error);
      alert('Failed to apply filter');
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Part A: Image Filtering</h2>
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleImageUpload}
        className="mb-4"
      />
      <div className="flex space-x-2 mb-4">
        <button
          onClick={() => applyFilter('mean')}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Mean Filter
        </button>
        <button
          onClick={() => applyFilter('gaussian')}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Gaussian Filter
        </button>
        <button
          onClick={() => applyFilter('median')}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Median Filter
        </button>
        <button
          onClick={() => applyFilter('laplacian')}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Laplacian Sharpening
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <h3 className="text-lg font-medium">Original Image</h3>
          {originalImage && <img src={originalImage} alt="Original" className="max-w-full h-auto" />}
        </div>
        <div>
          <h3 className="text-lg font-medium">Filtered Image</h3>
          {filteredImage && <img src={filteredImage} alt="Filtered" className="max-w-full h-auto" />}
          {psnr && <p className="mt-2">PSNR: {psnr.toFixed(2)} dB</p>}
        </div>
      </div>
    </div>
  );
}

export default ImageFiltering;