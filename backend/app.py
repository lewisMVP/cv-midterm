from flask import Flask, request, jsonify
import cv2
from PIL import Image
import io
import numpy as np
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://lewisMVP.github.io"]}})

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# Part A: Image Filtering
@app.route('/filter', methods=['POST'])
def filter_image():
    # Kiểm tra input
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img = np.array(Image.open(file))
    
    # Chuyển đổi ảnh sang định dạng phù hợp (BGR)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Tạo ảnh nhiễu (Gaussian noise)
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    
    # Áp dụng tất cả bộ lọc
    filters = {
        'mean': cv2.blur(noisy_img, (5, 5)),
        'gaussian': cv2.GaussianBlur(noisy_img, (5, 5), 0),
        'median': cv2.medianBlur(noisy_img, 5),
        'laplacian': cv2.convertScaleAbs(cv2.Laplacian(noisy_img, cv2.CV_64F))
    }
    
    # Tính PSNR cho từng bộ lọc để so sánh hiệu quả giảm nhiễu
    psnr_values = {}
    for key, filtered in filters.items():
        mse = np.mean((img - filtered) ** 2)
        if mse == 0:
            psnr_values[key] = 100
        else:
            max_pixel = 255.0
            psnr_values[key] = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return jsonify({
        'original': image_to_base64(img),
        'noisy': image_to_base64(noisy_img),
        'filtered': {key: image_to_base64(img) for key, img in filters.items()},
        'psnr': psnr_values
    })

# Part B: 3D Reconstruction
@app.route('/3dconstruction', methods=['POST'])
def reconstruct_3d():
    # Kiểm tra input
    if 'left_image' not in request.files or 'right_image' not in request.files:
        return jsonify({'error': 'Both left and right images are required'}), 400
    
    left_file = request.files['left_image']
    right_file = request.files['right_image']
    num_disparities = int(request.form.get('num_disparities', 64))  # Lấy tham số từ frontend
    method = request.form.get('method', 'StereoBM')  # Lấy phương pháp (StereoBM hoặc StereoSGBM)

    # Đọc ảnh bằng OpenCV để hỗ trợ PGM
    left_img = cv2.imdecode(np.frombuffer(left_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imdecode(np.frombuffer(right_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        return jsonify({'error': 'Failed to read images. Ensure they are valid image files (JPEG, PNG, or PGM)'}), 400
    
    # Kiểm tra kích thước ảnh
    if left_img.shape != right_img.shape:
        return jsonify({'error': 'Left and right images must have the same dimensions'}), 400

    # Kiểm tra kích thước hợp lệ
    h, w = left_img.shape
    if h <= 0 or w <= 0:
        return jsonify({'error': 'Invalid image dimensions'}), 400

    # Chuyển ảnh grayscale thành BGR để hiển thị
    left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    left_gray = left_img
    right_gray = right_img
    
    # Hiệu chỉnh ảnh (rectification) - Giả định thông số camera đơn giản
    focal_length = 600.0
    baseline = 80.0
    camera_matrix = np.array([[focal_length, 0, w/2],
                              [0, focal_length, h/2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    T = np.array([baseline, 0, 0], dtype=np.float32)
    
    # Debug: Kiểm tra kiểu dữ liệu và kích thước của các ma trận
    print(f"camera_matrix dtype: {camera_matrix.dtype}, shape: {camera_matrix.shape}")
    print(f"dist_coeffs dtype: {dist_coeffs.dtype}, shape: {dist_coeffs.shape}")
    print(f"R dtype: {R.dtype}, shape: {R.shape}")
    print(f"T dtype: {T.dtype}, shape: {T.shape}")
    
    # Fix type mismatch in stereoRectify
    try:
        # Make sure all matrices have the same data type
        camera_matrix = np.array([[focal_length, 0, w/2],
                                [0, focal_length, h/2],
                                [0, 0, 1]], dtype=np.float64)  # Change to float64
        dist_coeffs = np.zeros(5, dtype=np.float64)  # Change to float64
        R = np.eye(3, dtype=np.float64)  # Change to float64
        T = np.array([baseline, 0, 0], dtype=np.float64)  # Change to float64
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            camera_matrix, dist_coeffs, camera_matrix, dist_coeffs,
            (w, h), R, T, alpha=0
        )
    except cv2.error as e:
        print(f"Stereo rectification failed: {str(e)}")
        # Fallback with consistent data types
        Q = np.float32([[1, 0, 0, -0.5 * w],
                      [0, -1, 0, 0.5 * h],
                      [0, 0, 0, focal_length],
                      [0, 0, -1/baseline, 0]])
    else:
        # Tạo bản đồ ánh xạ để hiệu chỉnh ảnh
        map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)
        
        # Hiệu chỉnh ảnh
        left_rectified = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)
    
    # Tính bản đồ disparity
    if method == 'StereoSGBM':
        stereo = cv2.StereoSGBM_create(
            minDisparity=-16,
            numDisparities=num_disparities,
            blockSize=3,
            P1=8 * 3 * 3 ** 2,
            P2=32 * 3 * 3 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=150,
            speckleRange=64
        )
    else:
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=5)
    
    disparity = stereo.compute(left_rectified, right_rectified)
    disparity = cv2.filterSpeckles(disparity, 0, 4000, num_disparities)[0]
    print(f"Disparity min: {disparity.min()}, max: {disparity.max()}, mean: {disparity.mean()}")
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Tái tạo điểm 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    points_3d = points_3d.reshape(-1, 3)
    print(f"Total points before filtering: {points_3d.shape[0]}")

    # Filter out invalid points
    points_3d = points_3d[np.isfinite(points_3d).all(axis=1)]
    print(f"Points after removing non-finite values: {points_3d.shape[0]}")

    # Use percentile-based filtering to preserve structure
    z_vals = points_3d[:, 2]
    x_vals = points_3d[:, 0]
    y_vals = points_3d[:, 1]

    # Remove extreme outliers (1st and 99th percentiles)
    z_min, z_max = np.percentile(z_vals, [1, 99])
    x_min, x_max = np.percentile(x_vals, [1, 99])
    y_min, y_max = np.percentile(y_vals, [1, 99])

    # Apply filters
    mask = (z_vals > z_min) & (z_vals < z_max)
    mask &= (x_vals > x_min) & (x_vals < x_max)
    mask &= (y_vals > y_min) & (y_vals < y_max)
    points_3d = points_3d[mask]
    print(f"Points after percentile filtering: {points_3d.shape[0]}")

    # Apply better scale factor depending on the range of values
    z_range = np.max(np.abs(points_3d[:, 2]))
    scale_factor = 10.0 / (z_range + 1e-6)  # Adaptive scaling
    points_3d = points_3d * scale_factor
    print(f"Applied scale factor: {scale_factor}")

    # Ensure points are within reasonable bounds for visualization
    max_distance = 20.0  # Keep within reasonable bounds for Three.js
    points_3d = points_3d[np.sum(points_3d**2, axis=1) < max_distance**2]
    print(f"Points after distance filtering: {points_3d.shape[0]}")

    # Downsample if necessary to maintain performance
    if len(points_3d) > 10000:
        # Use stratified sampling based on depth to maintain structure
        n_bins = 20
        z_bins = np.linspace(points_3d[:, 2].min(), points_3d[:, 2].max(), n_bins)
        sampled_points = []
        
        for i in range(n_bins-1):
            bin_mask = (points_3d[:, 2] >= z_bins[i]) & (points_3d[:, 2] < z_bins[i+1])
            bin_points = points_3d[bin_mask]
            
            if len(bin_points) > 0:
                # Sample proportionally to bin size with minimum of 10 points per bin
                bin_samples = max(10, int(10000 * len(bin_points) / len(points_3d)))
                indices = np.random.choice(len(bin_points), min(bin_samples, len(bin_points)), replace=False)
                sampled_points.append(bin_points[indices])
        
        if sampled_points:
            points_3d = np.vstack(sampled_points)
            
        # If stratified sampling failed, fall back to random sampling
        if len(points_3d) > 10000:
            indices = np.random.choice(len(points_3d), 10000, replace=False)
            points_3d = points_3d[indices]

    print(f"Final point cloud size: {points_3d.shape[0]} points")

    # Send as list
    points_3d = points_3d.tolist()
    
    # Tính ma trận cơ bản và đường epipolar
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:10]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    # Vẽ đường epipolar trên ảnh trái
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    left_epipolar = left_img_bgr.copy()
    for line, pt in zip(lines1, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [left_img.shape[1], -(line[2] + line[0] * left_img.shape[1]) / line[1]])
        cv2.line(left_epipolar, (x0, y0), (x1, y1), color, 1)
        cv2.circle(left_epipolar, tuple(map(int, pt[0])), 5, color, -1)
    
    # Vẽ đường epipolar trên ảnh phải
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    right_epipolar = right_img_bgr.copy()
    for line, pt in zip(lines2, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [right_img.shape[1], -(line[2] + line[0] * right_img.shape[1]) / line[1]])
        cv2.line(right_epipolar, (x0, y0), (x1, y1), color, 1)
        cv2.circle(right_epipolar, tuple(map(int, pt[0])), 5, color, -1)
    
    return jsonify({
        'disparity': image_to_base64(disp_vis),
        'points_3d': points_3d,
        'left_epipolar': image_to_base64(left_epipolar),
        'right_epipolar': image_to_base64(right_epipolar)
    })

# Part C: Image Stitching
def stitch_two_images(img1, img2):
    # Chuyển ảnh sang grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện và mô tả đặc trưng bằng ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # Khớp đặc trưng
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Vẽ matched keypoints
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Ước lượng homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(np.sum(mask))
    
    # Kiểm tra chất lượng homography
    if inliers < 50:  # Nếu số lượng inliers quá thấp, trả về ảnh gốc
        print(f"Warning: Low inliers ({inliers}). Skipping stitching for this pair.")
        return img2, matches_img, inliers
    
    # Ghép ảnh
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Tính kích thước canvas lớn hơn để chứa toàn bộ ảnh sau khi warp
    # Warp img1 để lấy các điểm biên
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners1, H)
    all_corners = np.concatenate((warped_corners, corners2), axis=0)
    
    # Tìm giới hạn của canvas
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    width = x_max - x_min
    height = y_max - y_min
    
    # Tạo ma trận dịch chuyển để đưa ảnh về vùng dương
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    H = translation @ H
    
    # Warp img1
    result = cv2.warpPerspective(img1, H, (width, height))
    
    # Tạo mask cho img2
    img2_translated = np.zeros_like(result)
    img2_translated[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2
    
    # Tạo mask để blending
    mask1 = np.zeros((height, width), dtype=np.float32)
    mask2 = np.zeros((height, width), dtype=np.float32)
    mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.float32), H, (width, height))
    mask2[-y_min:-y_min+h2, -x_min:-x_min+w2] = 1.0
    
    # Alpha blending đơn giản
    alpha = 0.5
    result = result * mask1[:, :, None] * alpha + img2_translated * mask2[:, :, None] * (1 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    print(f"Result shape after stitching: {result.shape}")
    return result, matches_img, inliers

@app.route('/stitch', methods=['POST'])
def stitch_images():
    files = request.files
    if len(files) < 4:  # Yêu cầu ít nhất 4 ảnh
        return jsonify({'error': 'At least four images are required'}), 400
    
    images = []
    for key in files:
        file = files[key]
        img = np.array(Image.open(file))
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    
    # Ghép tuần tự 4 ảnh
    result, matches_img, inliers = stitch_two_images(images[0], images[1])
    for i in range(2, 4):
        result, _, _ = stitch_two_images(result, images[i])
    
    return jsonify({
        'matches': image_to_base64(matches_img),
        'panorama': image_to_base64(result),
        'inliers': inliers
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)