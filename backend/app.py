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
    # Đảm bảo ảnh cùng kiểu dữ liệu và kích thước phù hợp
    img1 = cv2.resize(img1, (0, 0), fx=1.0, fy=1.0)
    img2 = cv2.resize(img2, (0, 0), fx=1.0, fy=1.0)
    
    # Chuyển ảnh sang grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Sử dụng SIFT thay vì ORB để có kết quả tốt hơn
    # Nếu SIFT không có sẵn, sử dụng ORB với số lượng features cao hơn
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # FLANN matcher hoạt động tốt hơn với SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lọc matches tốt bằng Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    except:
        # Fallback sang ORB nếu SIFT không có sẵn
        orb = cv2.ORB_create(nfeatures=3000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        # Brute force matcher cho ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = bf.match(des1, des2)
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]  # Lấy top 100 matches
    
    # Vẽ matched keypoints
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Ước lượng homography nếu có đủ matches
    if len(good_matches) < 10:
        print(f"Warning: Not enough good matches ({len(good_matches)}). Skipping stitching.")
        return img2, matches_img, 0
    
    # Ước lượng homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Thử RANSAC với nhiều iteration hơn và threshold phù hợp hơn
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=10000)
    inliers = int(np.sum(mask))
    
    # Kiểm tra chất lượng homography (giảm ngưỡng xuống)
    min_inliers = min(30, len(good_matches) // 4)  # Điều chỉnh ngưỡng dựa vào số matches
    if inliers < min_inliers or H is None:
        print(f"Warning: Low quality homography ({inliers} inliers). Skipping stitching.")
        return img2, matches_img, inliers
    
    # Tính kích thước canvas cho ảnh ghép
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Tính toán kích thước và vị trí canvas
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # Biến đổi góc của ảnh 1 theo homography
    warped_corners = cv2.perspectiveTransform(corners1, H)
    
    # Kết hợp tất cả các góc để tìm kích thước tối đa
    all_corners = np.concatenate((warped_corners, corners2), axis=0)
    
    # Tìm giới hạn canvas
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Đảm bảo kích thước hợp lý
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0 or width > 10000 or height > 10000:
        print(f"Warning: Invalid dimensions ({width}x{height}). Skipping stitching.")
        return img2, matches_img, inliers
    
    # Ma trận dịch chuyển để đưa về vùng dương
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    H_adjusted = translation @ H
    
    # Tạo ảnh warp từ ảnh 1
    warped_img = cv2.warpPerspective(img1, H_adjusted, (width, height))
    
    # Tạo mask cho vùng warp
    mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8), H_adjusted, (width, height))
    
    # Tạo canvas cho ảnh kết quả
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Đặt ảnh 2 vào vị trí thích hợp trong canvas
    y_offset = max(0, -y_min)
    x_offset = max(0, -x_min)
    
    # Đảm bảo không vượt quá kích thước
    y_end = min(height, y_offset + h2)
    x_end = min(width, x_offset + w2)
    y2_end = min(h2, y_end - y_offset)
    x2_end = min(w2, x_end - x_offset)
    
    # Chèn ảnh 2 vào vị trí thích hợp
    result[y_offset:y_offset + y2_end, x_offset:x_offset + x2_end] = img2[:y2_end, :x2_end]
    
    # Tạo mask cho ảnh 2
    mask2 = np.zeros((height, width), dtype=np.uint8)
    mask2[y_offset:y_offset + y2_end, x_offset:x_offset + x2_end] = 1
    
    # Tính vùng chồng lấp
    overlap_mask = mask1 & mask2
    
    # Tạo gradient mask cho vùng chồng lấp để blend mượt hơn
    if np.sum(overlap_mask) > 0:
        # Tạo gradient mask từ trái sang phải trong vùng chồng lấp
        y_indices, x_indices = np.where(overlap_mask > 0)
        left_edge = np.min(x_indices)
        right_edge = np.max(x_indices)
        
        # Tạo gradient từ 0 đến 1 trong vùng chồng lấp
        width_overlap = right_edge - left_edge + 1
        if width_overlap > 1:  # Tránh chia cho 0
            for x in range(left_edge, right_edge + 1):
                # Tạo gradient từ 0->1 theo chiều từ trái sang phải
                alpha = (x - left_edge) / (width_overlap - 1)
                # Áp dụng gradient chỉ trong vùng chồng lấp tại cột x
                col_overlap = overlap_mask[:, x].astype(bool)
                
                # Áp dụng alpha blending với gradient
                result[col_overlap, x, :] = (
                    (1 - alpha) * result[col_overlap, x, :] + 
                    alpha * warped_img[col_overlap, x, :]
                ).astype(np.uint8)
        
        # Áp dụng warped image ở những vùng không chồng lấp với ảnh 2
        non_overlap_warped = (mask1 > 0) & (mask2 == 0)
        result[non_overlap_warped] = warped_img[non_overlap_warped]
    else:
        # Nếu không có vùng chồng lấp, kết hợp cả hai ảnh
        result = np.where(mask2[:, :, np.newaxis] > 0, result, warped_img)
    
    return result, matches_img, inliers

@app.route('/stitch', methods=['POST'])
def stitch_images():
    files = request.files
    if len(files) < 2:  # Cần ít nhất 2 ảnh
        return jsonify({'error': 'At least two images are required'}), 400
    
    # Đọc và xử lý các ảnh
    images = []
    for key in sorted(files.keys()):  # Sắp xếp theo thứ tự key
        file = files[key]
        try:
            img = np.array(Image.open(file))
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                return jsonify({'error': f'Unsupported image format for {key}'}), 400
            
            # Resize ảnh nếu quá lớn để tăng hiệu suất
            h, w = img.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            images.append(img)
        except Exception as e:
            return jsonify({'error': f'Failed to process image {key}: {str(e)}'}), 400
    
    if len(images) < 2:
        return jsonify({'error': 'At least two valid images are required'}), 400
    
    # Ghép tuần tự từ ảnh đầu tiên
    result = images[0]
    matches_img = None  # Lưu kết quả của cặp ghép đầu tiên để hiển thị
    all_inliers = []
    
    try:
        # Thử ghép từng cặp ảnh liên tiếp
        for i in range(1, len(images)):
            temp_result, current_matches_img, inliers = stitch_two_images(result, images[i])
            
            # Lưu lại ảnh matches của cặp đầu tiên (thường quan trọng nhất)
            if i == 1:
                matches_img = current_matches_img
            
            result = temp_result
            all_inliers.append(inliers)
            
            # Log thông tin để debug
            print(f"Stitched image pair {i-1}-{i}, inliers: {inliers}")
    except Exception as e:
        print(f"Error during stitching: {str(e)}")
        # Trả về lỗi nhưng vẫn gửi kết quả trung gian nếu có
        if matches_img is None:
            return jsonify({'error': f'Stitching failed: {str(e)}'}), 500
    
    # Đảm bảo rằng luôn có ảnh matches để trả về
    if matches_img is None and len(images) >= 2:
        # Nếu không có ảnh matches, tạo một ảnh matches đơn giản giữa hai ảnh đầu tiên
        orb = cv2.ORB_create()
        gray1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:10]
        matches_img = cv2.drawMatches(images[0], kp1, images[1], kp2, matches, None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return jsonify({
        'matches': image_to_base64(matches_img),
        'panorama': image_to_base64(result),
        'inliers': all_inliers
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)