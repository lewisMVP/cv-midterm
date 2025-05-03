import { useState, useRef } from 'react';

// Component xử lý Part C: Image Stitching
function ImageStitching() {
    // State lưu trữ kết quả từ backend
    const [matchesImage, setMatchesImage] = useState(null);
    const [panorama, setPanorama] = useState(null);
    const [inliers, setInliers] = useState(null);
    const fileInputRef = useRef(null);

    // Xử lý khi người dùng upload ảnh và gọi API stitch
    const handleUpload = async () => {
        const files = fileInputRef.current.files;
        if (files.length < 4) {
            alert('Please upload at least four images!');
            return;
        }

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
        } catch (error) {
            console.error('Error stitching:', error);
            alert('Failed to stitch images');
        }
    };

    return (
        <div className="bg-blue-50 p-4 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4 text-indigo-800">Part C: Image Stitching</h2>
            <input
                type="file"
                accept="image/*"
                multiple
                ref={fileInputRef}
                className="mb-4"
            />
             <div>
                <button 
                    onClick={handleUpload}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                >
                    Stitch Images
                </button>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Matched Keypoints Between Images</h3>
                    {matchesImage && <img src={matchesImage} alt="Matches" className="max-w-full h-auto" />}
                    {inliers && inliers.length > 0 && (
                        <div className="mt-2">
                            <p className="text-gray-600">Number of Inliers:</p>
                            <ul className="list-disc ml-6">
                                {inliers.map((count, idx) => (
                                    <li key={idx} className="text-gray-600">
                                        Image {idx+1}-{idx+2}: {count} inliers
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
                <div>
                    <h3 className="text-lg font-medium text-gray-700">Panorama</h3>
                    {panorama && <img src={panorama} alt="Panorama" className="max-w-full h-auto" />}
                </div>
            </div>
        </div>
    );
}

export default ImageStitching;