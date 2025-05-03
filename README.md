# INS3155 Computer Vision Midterm Project

This repository contains the implementation of the INS3155 (Computer Vision) midterm project at VNU-IS, Spring 2025. The project focuses on three key tasks: image filtering, 3D reconstruction from stereo images, and image stitching to create panoramic views. It adheres to the guideline of using only traditional image processing methods, avoiding deep learning models or pretrained networks.

## Project Overview

The project is divided into three main parts:

- **Part A: Image Filtering** - Implements traditional image filtering techniques (Mean, Gaussian, Median, and Laplacian Sharpening) to reduce Gaussian noise and enhance image quality. Metrics like PSNR and SSIM are used for evaluation.
- **Part B: 3D Reconstruction** - Uses stereo image pairs to compute disparity maps with StereoBM and StereoSGBM, followed by 3D point cloud reconstruction. The fundamental matrix and epipolar lines validate the stereo geometry.
- **Part C: Image Stitching** - Stitches multiple images into a panorama using ORB feature detection, homography estimation with RANSAC, and linear alpha blending for seamless transitions.

The source code is available in this repository, with detailed documentation in the `report/main.pdf`.

## Technologies Used

- **Backend**: 
  - Python 3.12.6 for core processing and scripting.
  - OpenCV (version 4.10.0) for image processing and computer vision tasks.
  - NumPy for numerical computations and array operations.
- **Frontend**: 
  - React for building an interactive user interface.
  - @react-three/fiber for 3D visualization of point clouds and other graphical elements.
- **Documentation**:
  - LaTeX for generating the project report (`report/main.tex`).

## Installation Guide

### Prerequisites
- Python 3.12.6 or higher.
- Node.js and npm for the frontend.
- A LaTeX distribution (e.g., TeX Live) for generating the project report.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/lewisMVP/cv-midterm.git
   cd cv-midterm
