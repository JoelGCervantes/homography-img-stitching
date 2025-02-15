# homography-img-stitching
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

# How it works 
1. Feature Extraction & Matching
   - Converts images to grayscale
   - Detects SIFT keypoints and descriptors
   - Uses Brute Force Matcher (BFMatcher) to find corresponding points
  
2.  Homography Estimation (RANSAC)
   - Uses RANSAC to estimate a homography matrix, filtering out incorrect matches
   - Computes transformation matrix H that aligns img_1 with img_2

3. Warping & Blending
   - Uses the inverse homography to warp img_1
   - Applies bilinear interpolation for smooth warping
   - Performs average blending to merge both images into a panorama









