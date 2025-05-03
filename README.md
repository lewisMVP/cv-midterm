# INS3155 Computer Vision Midterm Project

This repository contains the implementation of the INS3155 (Computer Vision) midterm project at VNU-IS, Spring 2025. The project focuses on three key tasks: image filtering, 3D reconstruction from stereo images, and image stitching to create panoramic views. It adheres to the guideline of using only traditional image processing methods, avoiding deep learning models or pretrained networks.

## Project Overview

The project is divided into three main parts:

- **Part A: Image Filtering** - Implements traditional image filtering techniques (Mean, Gaussian, Median, and Laplacian Sharpening) to reduce Gaussian noise and enhance image quality. Metrics like PSNR and SSIM are used for evaluation.
- **Part B: 3D Reconstruction** - Uses stereo image pairs to compute disparity maps with StereoBM and StereoSGBM, followed by 3D point cloud reconstruction. The fundamental matrix and epipolar lines validate the stereo geometry.
- **Part C: Image Stitching** - Stitches multiple images into a panorama using ORB feature detection, homography estimation with RANSAC, and linear alpha blending for seamless transitions.

## Technologies Used

- **Back-end**: 
  - Python 3.12.6 for core processing and scripting.
  - OpenCV (version 4.10.0) for image processing and computer vision tasks.
  - NumPy for numerical computations and array operations.
  - Flask for building RESTful APIs to handle requests from the frontend, process images, and return results (e.g., filtered images, 3D point clouds, panoramas).
- **Front-end**: 
  - React.JS, TailwindCSS for building an interactive user interface.
  - @react-three/fiber for 3D visualization of point clouds and other graphical elements.
    
## Installation Guide

### Prerequisites
- Python 3.12.6 or higher.
- Node.js and npm for the Front-end.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/lewisMVP/cv-midterm.git
   cd cv-midterm
2. **Set Up the Back-end**
- Create a virtual environment and activate it:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
- Install the required Python packages:
  ```bash
  pip install -r backend/requirements.txt
- Ensure OpenCV is installed correctly (version 4.10.0 is recommended).
- Navigate to the backend directory:
  ```bash
  cd backend
- Start the Back-end server:
  ```bash
  python app.py
3. **Set Up the Front-end**
- Navigate to the frontend directory:
  ```bash
  cd frontend
- Install dependences:
  ```bash
  npm install
- Start the development server:
  ```bash
  npm run dev
- The Front-end server will be available at http://localhost:5173/cv-midterm/

### Usagge
- Back-end: Run the backend scripts from backend/app.py to process images and generate results for image filtering, 3D reconstruction, or image stitching.
- Front-end: Use the ReactJS interface to upload images and visualize the outcomes (e.g., filtered images, 3D point clouds, panoramas). Access it at http://localhost:5173 after starting the development server.

## Acknowledgements
- Thanks to the INS3155 course instructors at VNU-IS for providing the project guidelines.
- OpenCV documentation and community for valuable resources.
