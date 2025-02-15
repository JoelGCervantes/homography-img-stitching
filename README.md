# homography-img-stitching-
Homography panorama using SIFT, RANSAC, and bilinear interpolation

# Image stitching using homography and RANSAC 
This project implements image stitching to create a panorama by aligning and blending two images using feature extraction, homography estimation, and warping techniques.

# Features 
- Extracts keypoints using cv2 SIFT() (Scale-Invariant Feature Transform)
- Matches keypoints using cv2 BFMatcher() (Brute Force Matcher)
- Computes a homography matrix with RANSAC for robust alignment
- Warps and blends images using barward warping with bilinear interpolation

# installation 
git clone https://github.com/JoelGCervantes/homography-img-stitching.git

# Usage 
python img-stitch-panorama.py <image_1> <image_2> <output_image>
example: python img-stitch-panorama.py img1.jpg img2.jpg panorama.jpg
