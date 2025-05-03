INS3155 Computer Vision Midterm Project
This repository contains the implementation of the INS3155 (Computer Vision) midterm project at VNU-IS, Spring 2025. The project focuses on three key tasks: image filtering, 3D reconstruction from stereo images, and image stitching to create panoramic views. It adheres to the guideline of using only traditional image processing methods, avoiding deep learning models or pretrained networks.
Technologies Used

Backend: 
Python 3.12.6 for core processing and scripting.
OpenCV (version 4.10.0) for image processing and computer vision tasks.
NumPy for numerical computations and array operations.


Frontend: 
React for building an interactive user interface.
@react-three/fiber for 3D visualization of point clouds and other graphical elements.



Installation Guide
Prerequisites

Python 3.12.6 or higher.
Node.js and npm for the frontend.
A LaTeX distribution (e.g., TeX Live) for generating reports.

Steps

Clone the Repository
git clone https://github.com/lewisMVP/cv-midterm.git
cd cv-midterm


Set Up the Backend

Create a virtual environment and activate it:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required Python packages:pip install -r backend/requirements.txt


Ensure OpenCV is installed correctly (version 4.10.0 is recommended).


Set Up the Frontend

Navigate to the frontend directory:cd frontend


Install dependencies:npm install


Start the development server:npm start


The frontend will be available at http://localhost:3000.


Generate the Report (Optional)

Navigate to the report directory:cd report


Compile the LaTeX document:latexmk -pdf main.tex


View the generated main.pdf for the project report.



Usage

Run the backend scripts from backend/app.py to process images and generate results.
Use the frontend interface to upload images and visualize the outcomes (e.g., filtered images, 3D point clouds, panoramas).
Modify the source code or configuration files in backend/ and frontend/ as needed.

Notes

Replace placeholder image paths (e.g., path/to/image.jpg) in the code with actual file locations.
Ensure sufficient memory and GPU support for 3D reconstruction tasks if available.
For issues, refer to the OpenCV documentation or the project issue tracker.

Happy coding and exploring computer vision!
