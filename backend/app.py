from flask import Flask, request, jsonify
import cv2
from PIL import Image
import io
import numpy as np
import base64
from flask_cors import CORS
import time
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173", 
    "https://lewisMVP.github.io",
    "https://cv-midterm.onrender.com"  # Thêm domain production
]}})

def image_to_base64(img):
    # Không chuyển đổi nếu ảnh đã là RGB hoặc ảnh grayscale
    if img is None:
        # Trả về ảnh trống nếu img là None
        img = np.zeros((100, 100, 3), dtype=np.uint8)
    
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
    
    # Lưu lại ảnh màu gốc nếu có
    if len(img.shape) == 3:
        img_color = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    noisy_img = img_gray  # Sử dụng ảnh gốc thay vì tạo ảnh nhiễu
    
    # Áp dụng các bộ lọc làm nét theo yêu cầu
    filters = {}
    psnr_values = {}
    ssim_values = {}
    computation_times = {}
    
    # 1. Mean filter
    start_time = time.time()
    filters['mean'] = cv2.blur(noisy_img, (5, 5))
    computation_times['mean'] = time.time() - start_time
    
    # 2. Gaussian filter
    start_time = time.time()
    filters['gaussian'] = cv2.GaussianBlur(noisy_img, (5, 5), 0)
    computation_times['gaussian'] = time.time() - start_time
    
    # 3. Median filter
    start_time = time.time()
    filters['median'] = cv2.medianBlur(noisy_img, 5)
    computation_times['median'] = time.time() - start_time
    
    # 4. Laplacian sharpening (nâng cao biên cạnh)
    start_time = time.time()
    laplacian = cv2.Laplacian(noisy_img, cv2.CV_64F)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    filters['laplacian'] = cv2.addWeighted(noisy_img, 1.0, laplacian, 0.5, 0)
    computation_times['laplacian'] = time.time() - start_time
    
    # Tính PSNR và SSIM để so sánh hiệu quả giảm nhiễu của từng bộ lọc
    for key, filtered in filters.items():
        # Tính PSNR
        mse = np.mean((img_gray - filtered) ** 2)
        if mse == 0:
            psnr_values[key] = 100
        else:
            max_pixel = 255.0
            psnr_values[key] = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # Tính SSIM
        try:
            # Hàm compareSSIM trong cv2 được đổi tên thành compare_ssim trong skimage.metrics
            from skimage.metrics import structural_similarity as ssim
            ssim_values[key] = ssim(img_gray, filtered, data_range=255)
        except ImportError:
            # Nếu không có skimage, thử dùng cv2 (phiên bản cũ)
            try:
                ssim_values[key] = cv2.compareSSIM(img_gray, filtered)
            except:
                ssim_values[key] = -1  # không tính được SSIM
    
    # Tính biên cạnh bằng Canny để so sánh khả năng bảo toàn biên của từng bộ lọc
    edge_preservation = {}
    edges_original = cv2.Canny(img_gray, 50, 150)
    for key, filtered in filters.items():
        edges_filtered = cv2.Canny(filtered, 50, 150)
        # Tính độ tương đồng giữa hai ảnh biên cạnh (để đo lường việc giữ biên cạnh)
        edge_preservation[key] = cv2.compareHist(
            cv2.calcHist([edges_original], [0], None, [256], [0, 256]), 
            cv2.calcHist([edges_filtered], [0], None, [256], [0, 256]),
            cv2.HISTCMP_CORREL
        )
    
    # Chuyển các ảnh grayscale sang BGR để hiển thị
    img_display_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # Đổi tên để tránh nhầm lẫn
    filters_display = {key: cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for key, img in filters.items()}
    
    # Nếu ảnh đầu vào là ảnh màu, giữ nguyên màu gốc, ngược lại chuyển sang RGB
    if len(img.shape) == 3:
        img_display = img
    else:
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return jsonify({
        'original': image_to_base64(img_display),
        'grayscale': image_to_base64(img_display_gray),  # Trả về ảnh grayscale thay vì ảnh nhiễu
        'filtered': {key: image_to_base64(img) for key, img in filters_display.items()},
        'psnr': psnr_values,
        'ssim': ssim_values,
        'computation_time': computation_times,
        'edge_preservation': edge_preservation
    })

# Part B: 3D Reconstruction
@app.route('/3dconstruction', methods=['POST'])
def reconstruct_3d():
    # Kiểm tra input
    if 'left_image' not in request.files or 'right_image' not in request.files:
        return jsonify({'error': 'Both left and right images are required'}), 400
    
    left_file = request.files['left_image']
    right_file = request.files['right_image']
    num_disparities = int(request.form.get('num_disparities', 64))
    method = request.form.get('method', 'StereoBM')

    # Đọc ảnh màu
    left_img_color = cv2.imdecode(np.frombuffer(left_file.read(), np.uint8), cv2.IMREAD_COLOR)
    right_img_color = cv2.imdecode(np.frombuffer(right_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if left_img_color is None or right_img_color is None:
        return jsonify({'error': 'Failed to read images. Ensure they are valid image files (JPEG, PNG, or PGM)'}), 400
    
    # Chuyển đổi sang grayscale CHỈ cho việc tính toán stereo matching
    left_gray = cv2.cvtColor(left_img_color, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img_color, cv2.COLOR_BGR2GRAY)
    
    # Kiểm tra kích thước ảnh
    if left_gray.shape != right_gray.shape:
        return jsonify({'error': 'Left and right images must have the same dimensions'}), 400

    # Kiểm tra kích thước hợp lệ
    h, w = left_gray.shape
    if h <= 0 or w <= 0:
        return jsonify({'error': 'Invalid image dimensions'}), 400
    
    # Lưu bản sao ảnh màu gốc để hiển thị
    left_img_bgr = left_img_color.copy()
    right_img_bgr = right_img_color.copy()
    
    # Hiệu chỉnh ảnh (rectification) - Giả định thông số camera đơn giản
    focal_length = 600.0
    baseline = 80.0
    
    # Fix type mismatch in stereoRectify
    try:
        camera_matrix = np.array([[focal_length, 0, w/2],
                                [0, focal_length, h/2],
                                [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        T = np.array([baseline, 0, 0], dtype=np.float64)
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            camera_matrix, dist_coeffs, camera_matrix, dist_coeffs,
            (w, h), R, T, alpha=0
        )
        
        # Tạo bản đồ ánh xạ để hiệu chỉnh ảnh
        map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)
        
        # Hiệu chỉnh ảnh grayscale cho việc tính toán
        left_rectified = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)
        
        # Hiệu chỉnh ảnh màu cho việc hiển thị
        left_color_rectified = cv2.remap(left_img_color, map1x, map1y, cv2.INTER_LINEAR)
        right_color_rectified = cv2.remap(right_img_color, map2x, map2y, cv2.INTER_LINEAR)
        
        # Sử dụng ảnh màu đã hiệu chỉnh cho hiển thị
        left_img_bgr = left_color_rectified
        right_img_bgr = right_color_rectified
    except cv2.error as e:
        print(f"Stereo rectification failed: {str(e)}")
        left_rectified = left_gray
        right_rectified = right_gray
        Q = np.float32([[1, 0, 0, -0.5 * w],
                      [0, -1, 0, 0.5 * h],
                      [0, 0, 0, focal_length],
                      [0, 0, -1/baseline, 0]])
    
    # Tính bản đồ disparity sử dụng ảnh grayscale
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
    
    # Tính disparity map
    disparity = stereo.compute(left_rectified, right_rectified)
    disparity = cv2.filterSpeckles(disparity, 0, 4000, num_disparities)[0]
    print(f"Disparity min: {disparity.min()}, max: {disparity.max()}, mean: {disparity.mean()}")
    
    # Tạo disparity map chuẩn hóa
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Tạo disparity map có màu (áp dụng colormap)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    
    # Tái tạo điểm 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    
    # Lấy thông tin màu từ ảnh trái đã hiệu chỉnh
    h, w = left_rectified.shape
    colors = left_color_rectified.reshape(-1, 3)  # Chuyển thành mảng các điểm màu
    
    # Reshape points_3d để xử lý
    points_3d_flat = points_3d.reshape(-1, 3)
    
    # Filter out invalid points
    valid_indices = np.isfinite(points_3d_flat).all(axis=1)
    points_3d_valid = points_3d_flat[valid_indices]
    colors_valid = colors[valid_indices]
    
    # Use percentile-based filtering to preserve structure
    z_vals = points_3d_valid[:, 2]
    x_vals = points_3d_valid[:, 0]
    y_vals = points_3d_valid[:, 1]

    # Remove extreme outliers (1st and 99th percentiles)
    z_min, z_max = np.percentile(z_vals, [1, 99])
    x_min, x_max = np.percentile(x_vals, [1, 99])
    y_min, y_max = np.percentile(y_vals, [1, 99])

    # Apply filters
    mask = (z_vals > z_min) & (z_vals < z_max)
    mask &= (x_vals > x_min) & (x_vals < x_max)
    mask &= (y_vals > y_min) & (y_vals < y_max)
    points_3d_filtered = points_3d_valid[mask]
    colors_filtered = colors_valid[mask]
    
    # Apply better scale factor depending on the range of values
    z_range = np.max(np.abs(points_3d_filtered[:, 2]))
    scale_factor = 10.0 / (z_range + 1e-6)  # Adaptive scaling
    points_3d_filtered = points_3d_filtered * scale_factor
    
    # Ensure points are within reasonable bounds for visualization
    max_distance = 20.0
    dist_mask = np.sum(points_3d_filtered**2, axis=1) < max_distance**2
    points_3d_filtered = points_3d_filtered[dist_mask]
    colors_filtered = colors_filtered[dist_mask]
    
    # Downsample if necessary to maintain performance
    if len(points_3d_filtered) > 10000:
        indices = np.random.choice(len(points_3d_filtered), 10000, replace=False)
        points_3d_filtered = points_3d_filtered[indices]
        colors_filtered = colors_filtered[indices]
    
    # Convert to list for JSON serialization
    points_with_colors = []
    for i in range(len(points_3d_filtered)):
        # Each point is [x, y, z, r, g, b]
        point_with_color = points_3d_filtered[i].tolist() + colors_filtered[i].tolist()
        points_with_colors.append(point_with_color)
    
    print(f"Final colored point cloud size: {len(points_with_colors)} points")

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
        x1, y1 = map(int, [left_img_bgr.shape[1], -(line[2] + line[0] * left_img_bgr.shape[1]) / line[1]])
        cv2.line(left_epipolar, (x0, y0), (x1, y1), color, 1)
        cv2.circle(left_epipolar, tuple(map(int, pt[0])), 5, color, -1)
    
    # Vẽ đường epipolar trên ảnh phải
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    right_epipolar = right_img_bgr.copy()
    for line, pt in zip(lines2, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [right_img_bgr.shape[1], -(line[2] + line[0] * right_img_bgr.shape[1]) / line[1]])
        cv2.line(right_epipolar, (x0, y0), (x1, y1), color, 1)
        cv2.circle(right_epipolar, tuple(map(int, pt[0])), 5, color, -1)
    
    return jsonify({
        'disparity': image_to_base64(disp_color),
        'points_3d': points_with_colors,
        'left_epipolar': image_to_base64(left_epipolar),
        'right_epipolar': image_to_base64(right_epipolar)
    })

# Part C: Image Stitching
def stitch_two_images(img1, img2):
    # Đảm bảo ảnh cùng kiểu dữ liệu và kích thước phù hợp
    img1 = cv2.resize(img1, (0, 0), fx=1.0, fy=1.0)
    img2 = cv2.resize(img2, (0, 0), fx=1.0, fy=1.0)
    
    # Chuyển ảnh sang grayscale cho việc tìm đặc trưng
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
    result[y_offset:y_offset + y2_end, x_offset:x2_end] = img2[:y2_end, :x2_end]
    
    # Tạo mask cho ảnh 2
    mask2 = np.zeros((height, width), dtype=np.uint8)
    mask2[y_offset:y_offset + y2_end, x_offset:x2_end] = 1
    
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
    if len(files) < 4:
        return jsonify({'error': 'At least four images are required'}), 400
    
    # Đọc và xử lý các ảnh
    images = []
    for key in sorted(files.keys()):
        file = files[key]
        try:
            # Đọc ảnh đúng cách để đảm bảo màu sắc chính xác
            img_pil = Image.open(file)
            img = np.array(img_pil)
            
            # Chuyển đổi từ RGB sang BGR nếu cần (cho OpenCV)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # Xử lý ảnh RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            images.append(img)
        except Exception as e:
            return jsonify({'error': f'Failed to process image {key}: {str(e)}'}), 400
    
    # Phần code còn lại giữ nguyên
    # Đọc và xử lý các ảnh
    images = []
    for key in sorted(files.keys()):
        file = files[key]
        try:
            img = np.array(Image.open(file))
            # Xử lý định dạng ảnh...
            images.append(img)
        except Exception as e:
            return jsonify({'error': f'Failed to process image {key}: {str(e)}'}), 400
    
    if len(images) < 4:
        return jsonify({'error': 'At least four valid images are required'}), 400
    
    # Tạo ảnh matches giữa các cặp ảnh theo yêu cầu
    all_matches_imgs = []
    all_inliers = []
    
    # Ghép giữa các cặp ảnh liên tiếp: 0-1, 1-2, 2-3
    for i in range(len(images) - 1):
        # Gọi hàm match giữa hai ảnh liên tiếp
        _, current_matches_img, inliers = stitch_two_images(images[i], images[i+1])
        all_matches_imgs.append(current_matches_img)
        all_inliers.append(inliers)
    
    # Thực hiện ghép panorama từ tất cả các ảnh
    result = images[0]
    try:
        for i in range(1, len(images)):
            temp_result, _, _ = stitch_two_images(result, images[i])
            result = temp_result
    except Exception as e:
        print(f"Error during stitching: {str(e)}")
        if not all_matches_imgs:
            return jsonify({'error': f'Stitching failed: {str(e)}'}), 500
    
    # Nếu có nhiều hơn 4 ảnh, chỉ lấy 3 cặp đầu tiên
    if len(all_matches_imgs) > 3:
        all_matches_imgs = all_matches_imgs[:3]
        all_inliers = all_inliers[:3]
    
    # Ghép các ảnh matches lại với nhau để hiển thị
    combined_matches = None
    if len(all_matches_imgs) > 0:
        # Điều chỉnh kích thước các ảnh matches để có cùng chiều cao
        target_height = 400
        resized_matches = []
        for img in all_matches_imgs:
            h, w = img.shape[:2]
            aspect = w / h
            new_width = int(target_height * aspect)
            resized = cv2.resize(img, (new_width, target_height))
            resized_matches.append(resized)
        
        # Ghép ngang các ảnh matches
        combined_matches = np.hstack(resized_matches)
    
    # Chuyển đổi panorama và matches từ BGR sang RGB trước khi trả về
    if combined_matches is not None:
        combined_matches = cv2.cvtColor(combined_matches, cv2.COLOR_BGR2RGB)
    
    if result is not None:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return jsonify({
        'matches': image_to_base64(combined_matches),
        'panorama': image_to_base64(result),
        'inliers': all_inliers
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sẽ cung cấp PORT
    app.run(host='0.0.0.0', port=port, debug=False)  # Phải bind 0.0.0.0