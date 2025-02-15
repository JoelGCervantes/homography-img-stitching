import cv2
import sys
import numpy as np
import random

def compute_homography_matrix(list_matched_pairs):
  A = []
  for m in list_matched_pairs:
    homogeneous_c1 = [m[0][0], m[0][1], 1]
    homogeneous_c2 = [m[1][0], m[1][1], 1]
    x_prime_row = [-homogeneous_c1[0], -homogeneous_c1[1], -1, 0, 0, 0, homogeneous_c1[0]*homogeneous_c2[0], homogeneous_c1[1]*homogeneous_c2[0], homogeneous_c2[0]]
    y_prime_row = [0, 0, 0, -homogeneous_c1[0], -homogeneous_c1[1], -1, homogeneous_c1[0]*homogeneous_c2[1], homogeneous_c1[1]*homogeneous_c2[1], homogeneous_c2[1]]
    A.append(x_prime_row)
    A.append(y_prime_row)

  # compute homography matrix and normalize
  u, s, v = np.linalg.svd(A)
  H = np.reshape(v[8], (3, 3))
  H = (H/H.item(8))

  return H

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None

    total_inliers = []
    key_points_l = len(list_pairs_matched_keypoints) - 1  # for random purpose

    if(len(list_pairs_matched_keypoints) < 4):  # need at least 4 correspondances
        print ("not enough matched keypoints for homography")
        sys.exit(1)

    for _ in range(max_num_trial):

        # choose random points
        matches = []
        for _ in range(4):
            m = list_pairs_matched_keypoints[random.randint(0, key_points_l)]
            matches.append(m)

        # compute initial homography matrix
        H = compute_homography_matrix(matches)


        num_inliers = []

        # Euclidean distance calculation
        for pair in list_pairs_matched_keypoints:
            source_p = [pair[0][0], pair[0][1], 1] # (x,y,1) in img 1
            dest_p = [pair[1][0], pair[1][1], 1] # (x',y',1) in img2

            estimated_dest_p = np.dot(H, source_p)
            estimated_dest_p = estimated_dest_p/estimated_dest_p[2]


            reproj_error = np.sqrt(np.sum((dest_p - estimated_dest_p) **2))
            if reproj_error < threshold_reprojtion_error:
                num_inliers.append(pair)

        # update amount of inliers
        if len(num_inliers) > len(total_inliers):
            total_inliers = num_inliers
            current_best_H = H
            if (len(total_inliers) / len(list_pairs_matched_keypoints) ) > threshold_ratio_inliers:
                best_H = H

    # if no homography matrix with current threshold,create new homography matrix with all inliers

    if best_H is None:
        best_H = current_best_H

    H = compute_homography_matrix(total_inliers)
    best_H = H

    return best_H



def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):

    list_pairs_matched_keypoints = []

    # to be completed ....

    # convert image 1 and 2 to grayscale
    gray1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

    # create SIFT object
    SIFT = cv2.SIFT_create()

    # kp1, des1 = sift.compute(gray1,kp1)
    kp1, des1 = SIFT.detectAndCompute(gray1,None) # kp1 = list of keypoints, des = numpy array shape (number of keypoints) X 128
    kp2, des2 = SIFT.detectAndCompute(gray2,None)

    # brute force matcher
    BF_matcher = cv2.BFMatcher()
    matches = BF_matcher.knnMatch(des1,des2, k=2)

    # ratio test
    for m,n in matches:
      if m.distance < ratio_robustness*n.distance:
        kp1_match = kp1[m.queryIdx]  # Keypoint in img_1
        kp2_match = kp2[m.trainIdx]  # Keypoint in img_2

        # Extract coordinates and append to the list
        list_pairs_matched_keypoints.append([
             [kp1_match.pt[0], kp1_match.pt[1]],  # [p1x, p1y]
             [kp2_match.pt[0], kp2_match.pt[1]]   # [p2x, p2y
        ])
    return list_pairs_matched_keypoints



def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...

    # ===== blend images: average blending
    # to be completed ...

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...
    inverse_H = np.linalg.inv(H_1)

    # corner points corresponding to warped img_1
    corners = np.array([
        [0, 0, 1],
        [0, img_1.shape[0], 1],
        [img_1.shape[1], 0, 1],
        [img_1.shape[1], img_1.shape[0], 1]
    ]).T

    transformed_corners = np.dot(H_1, corners)
    transformed_corners /= transformed_corners[2]
    f11, f12, f13, f14 = transformed_corners.T

    hh = [f11[1], f12[1], f13[1], f14[1]]
    ww = [f11[0], f12[0], f13[0], f14[0]]


    h_min, h_max = np.floor(min(hh)), np.ceil(max(hh))
    w_min, w_max = np.floor(min(ww)), np.ceil(max(ww))

    h0 = int(abs(h_min))
    h1, h2 = int(h_min), int(h_max)
    w0 = int(abs(w_min))
    w1, w2 = int(w_min), int(w_max)
    right_w = img_2.shape[1] - w2

    warp = np.zeros([abs(h1) + abs(h2), abs(w1) + abs(w2) + right_w, 3], dtype=np.float32)

    # backward warping
    for y in range(h1, h2):
        for x in range(w1, w2 + right_w):

            # map panorama point back to img1 for bilinear interpolation
            mapped_p = np.dot(inverse_H, [x, y, 1])
            mapped_p /= mapped_p[-1]
            pixel_x = int(np.floor(mapped_p[0]))
            pixel_y = int(np.floor(mapped_p[1]))
            weight_x = mapped_p[0] - pixel_x
            weight_y = mapped_p[1] - pixel_y

            # chek pixel is not img1 bounds
            if pixel_x < 0 or pixel_y < 0 or pixel_x >= img_1.shape[1]-1 or pixel_y >= img_1.shape[0]-1:
                #  check it pixel is img2 pixel bounds, and copy pixel
                if y >= 0 and y < img_2.shape[0] and x >= 0 and x < img_2.shape[1]:
                    result = img_2[y, x]
                else: # set pixel to black if out of bounds
                    result = [0, 0, 0]

            # bilinear interpolation for pixel in img_1 bounds for warping
            else:
                # Extract neighboring pixel values
                top_left = img_1[pixel_y, pixel_x]
                top_right = img_1[pixel_y, pixel_x + 1]
                bottom_right = img_1[pixel_y + 1, pixel_x + 1]
                bottom_left = img_1[pixel_y + 1, pixel_x]
                # Compute bilinear interpolation using matrix multiplication
                result = (
                    (1 - weight_x) * (1 - weight_y) * top_left +
                    weight_x * (1 - weight_y) * top_right +
                    weight_x * weight_y * bottom_right +
                    (1 - weight_x) * weight_y * bottom_left
                )
                # blending
                if x >= 0 and x < img_2.shape[1] and y >= 0 and y < img_2.shape[0]:
                    r2 = img_2[y, x]
                    result = (result + r2) / 2.0

            warp[y + h0, x + w0] = result

    img_panorama = warp
    return img_panorama



def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)

    return img_panorama



if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]



    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))
