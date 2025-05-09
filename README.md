# Computer Vision Coursework 2: Panorama Stitching

*Group Members:* Jeswin Thomas & Ryan Thomas

---

## Overview

In this coursework, we set out to build an image stitching pipeline from scratch. Our goal was to take two separate images and merge them into a seamless panorama by stepping through core computer vision processes like feature detection, feature matching, estimating homography with RANSAC, and image warping.

We followed the brief by not using OpenCV’s built-in functions for matching, RANSAC, or warping. Instead, we coded everything manually to get a clearer understanding of how each stage works. We also added some optional improvements like blending and border cropping to enhance the final stitched image.

We shared all our work on GitHub and collaborated closely throughout, dividing tasks in a way that allowed us to work independently while frequently checking and testing each other’s work.

## Pipeline Summary

### 1. Image Loading

We used cv2.imread() to load the images and included checks to make sure the files were read correctly before continuing.

### 2. Feature Detection & Description

To detect and describe keypoints, we used the SIFT algorithm (cv2.SIFT_create()) on both input images.

### 3. Custom Matching

We built our own matching function using Euclidean distance between descriptor vectors. We then filtered the matches using Lowe’s ratio test to keep only reliable correspondences.

### 4. Drawing Matches

For match visualization, we used OpenCV’s drawMatches() after resizing both images to the same height and adjusting the keypoints accordingly so everything aligned visually.

### 5. RANSAC + Homography

We implemented RANSAC from scratch by:

* Randomly picking 4 matching pairs at a time.
* Computing homography using the DLT algorithm.
* Checking how many inliers fit using a reprojection error threshold.
* Refining the homography using all the inliers from the best sample.

### 6. Warping and Stitching

Once the homography was ready, we warped the right image to align with the left. We then created a large enough canvas to hold the combined image and handled overlaps using simple alpha blending.

---

## Extra Features

### Linear Blending

To smooth transitions in overlapping regions, we used cv2.addWeighted() for basic blending. This helped reduce visible seams between the two images.

### Black Border Removal

After stitching, we used a binary threshold and contour detection to find the valid part of the image. The stitched result was then cropped to remove any empty black borders.

---

## OpenCV Stitcher Comparison

As a benchmark, we also tested OpenCV’s high-level cv2.Stitcher_create() method. It was useful for comparing our manual results with an optimized built-in tool and helped us validate the quality of our own output.

---

## Group Contributions

### Jeswin Thomas

* Wrote the code for SIFT keypoint and descriptor extraction
* Developed the custom matching algorithm
* Implemented RANSAC-based homography estimation
* Handled image warping and stitching logic
* Added blending and black border removal steps
* Integrated OpenCV’s stitcher for comparison
* Took the lead in debugging and refining the full pipeline

### Ryan Thomas

* Worked on image loading and initial testing
* Helped adjust RANSAC parameters and verified inlier results
* Assisted in refining match visualizations
* Validated the pipeline with mobile phone images
* Ran tests across different setups to ensure everything worked reliably

---
To use your own mobile photos in the stitching pipeline, make sure the images are named mobile_left.jpeg and mobile_right.jpeg. These filenames are already referenced in the code, so as long as you replace the existing files with your own using the same names, the script will run without needing any further changes.
## Final Output

The final panorama stitches the two images with minimal visible seams. We saved both the manually stitched image and the one generated using OpenCV’s stitcher as stitched_output.jpg so they can be compared easily.

We tested with both sample images and real-world photos taken on a phone, and the results were consistently good.

---

## Files in This Repo

* stitcher.py – Core stitching pipeline script
* s1.jpg, s2.jpg – Sample images used for testing
* stitched_output.jpg – Final panorama result
* README.md – This file

---

## Notes

* Use Python 3 with OpenCV installed to run the script.
* Make sure image paths are correct in your local setup.
* You can adjust the matching threshold or RANSAC settings depending on your input images.
