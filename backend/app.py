from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

def image_to_base64(img):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.route('/filter', methods=['POST'])
def apply_filter():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    filter_type = request.form.get('filter_type', 'mean')
    
    # Read image
    img = np.array(Image.open(file))
    if img.shape[2] == 4:  # Convert RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Apply filter
    if filter_type == 'mean':
        filtered = cv2.blur(img, (5, 5))
    elif filter_type == 'gaussian':
        filtered = cv2.GaussianBlur(img, (5, 5), 0)
    elif filter_type == 'median':
        filtered = cv2.medianBlur(img, 5)
    elif filter_type == 'laplacian':
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        filtered = cv2.convertScaleAbs(laplacian)
    else:
        return jsonify({'error': 'Invalid filter type'}), 400
    
    # Convert images to base64
    original_base64 = image_to_base64(img)
    filtered_base64 = image_to_base64(filtered)
    
    # Compute PSNR for comparison
    mse = np.mean((img.astype(float) - filtered.astype(float)) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse > 0 else float('inf')
    
    return jsonify({
        'original': original_base64,
        'filtered': filtered_base64,
        'psnr': psnr
    })

@app.route('/3dconstruction', methods=['POST'])
def reconstruct():
    if 'left_image' not in request.files or 'right_image' not in request.files:
        return jsonify({'error': 'Two images required'}), 400
    
    left_file = request.files['left_image']
    right_file = request.files['right_image']
    
    # Read images
    left_img = cv2.imdecode(np.frombuffer(left_file.read(), np.uint8), cv2.IMREAD_COLOR)
    right_img = cv2.imdecode(np.frombuffer(right_file.read(), np.uint8), cv2.IMREAD_COLOR)
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 3D reconstruction (assume Q matrix)
    Q = np.float32([[1, 0, 0, -320], [0, 1, 0, -240], [0, 0, 0, 1000], [0, 0, -1, 0]])
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = (disparity > disparity.min()) & (np.isfinite(points_3d).all(axis=2))
    points_3d = points_3d[mask].reshape(-1, 3).tolist()
    
    # Compute fundamental matrix and epipolar lines
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    # Draw epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    left_epi = left_img.copy()
    for line in lines1[:10]:
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = left_img.shape[1], int(-(line[2] + line[0] * left_img.shape[1]) / line[1])
        cv2.line(left_epi, (x0, y0), (x1, y1), (0, 255, 0), 1)
    
    return jsonify({
        'disparity': image_to_base64(disp_vis),
        'points_3d': points_3d[:1000],  # Limit for performance
        'left_epipolar': image_to_base64(left_epi)
    })

@app.route('/stitch', methods=['POST'])
def stitch():
    if len(request.files) < 2:
        return jsonify({'error': 'At least two images required'}), 400
    
    images = []
    for key in request.files:
        img = cv2.imdecode(np.frombuffer(request.files[key].read(), np.uint8), cv2.IMREAD_COLOR)
        images.append(img)
    
    # Stitch first two images for simplicity
    img1, img2 = images[:2]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect and match features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Estimate homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = np.sum(mask)
    
    # Warp and blend
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    result[0:h2, 0:w2] = img2  # Simple blending
    
    return jsonify({
        'matches': image_to_base64(match_img),
        'panorama': image_to_base64(result),
        'inliers': int(inliers)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
