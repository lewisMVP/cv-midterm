import { useState, useRef } from 'react';

function ImageStitching() {
  const [matchesImage, setMatchesImage] = useState(null);
  const [panorama, setPanorama] = useState(null);
  const [inliers, setInliers] = useState(null);
  const fileInputRef = useRef(null);

  const handleUpload = async () => {
    const files = fileInputRef.current.files;
    if (files.length < 2) {
      alert('Please upload at least two images!');
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append(`image_${i}`, files[i]);
    }

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
      setMatchesImage(`data:image/png;base64,${data.matches}`);
      setPanorama(`data:image/png;base64,${data.panorama}`);
      setInliers(data.inliers);
    } catch (error) {
      console.error('Error stitching:', error);
      alert('Failed to stitch images');
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Part C: Image Stitching</h2>
      <input
        type="file"
        accept="image/*"
        multiple
        ref={fileInputRef}
        className="mb-4"
      />
      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Stitch Images
      </button>
      <div className="grid grid-cols-2 gap-2 mt-4">
        <div>
          <h3 className="text-lg font-medium">Matched Keypoints</h3>
          {matchesImage && <img src={matchesImage} alt="Matches" className="max-w-full h-auto" />}
          {inliers && <p className="mt-2">Inliers: {inliers}</p>}
        </div>
        <div>
          <h3 className="text-lg font-medium">Panorama</h3>
          {panorama && <img src={panorama} alt="Panorama" className="max-w-full h-auto" />}
        </div>
      </div>
    </div>
  );
}

export default ImageStitching;
