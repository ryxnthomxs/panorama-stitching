# panorama-stitching
Computer Vision Coursework 2: Panorama Stitching (Ryan Thomas)

Student Name: Ryan Thomas

Overview

In this coursework, I worked with Jeswin Thomas to develop an image stitching pipeline from scratch. Our goal was to understand the key computer vision concepts involved in creating a panorama—specifically, feature detection, matching, homography estimation with RANSAC, and image warping.

We deliberately avoided OpenCV's high-level functions for matching and warping, as required, and instead implemented these components ourselves. I focused mainly on testing, validating, and supporting core parts of the pipeline.

Pipeline Summary

1. Image Loading

Used cv2.imread() to load and validate the images before processing.

2. Feature Detection & Description

SIFT (cv2.SIFT_create()) was used to extract features—this was handled by Jeswin.

3. Matching

Assisted in testing and validating Jeswin’s custom matching implementation using Euclidean distance and Lowe’s ratio test.

4. Visualization

Helped debug and adjust the match visualization process, ensuring keypoints aligned correctly after image resizing.

5. RANSAC + Homography

Worked with Jeswin to tune the RANSAC parameters and reviewed match quality to ensure consistent inlier selection and transformation accuracy.

6. Warping and Stitching

Tested the final stitching pipeline with different images and helped verify that the blending and cropping steps worked as expected.

Testing Contributions

Sourced various image pairs, including mobile phone photos.

Ran multiple pipeline tests across different systems.

Helped troubleshoot compatibility and path issues.

OpenCV Stitcher Comparison

Ran and compared OpenCV’s built-in stitching tool to validate the accuracy of our custom pipeline.

Contributions

Image acquisition and pre-testing

RANSAC parameter tuning and validation

Testing and debugging visualizations

Multi-system compatibility testing

OpenCV stitcher evaluation

Final Output

The final result showed good alignment and minimal seams. The custom-stitched output was close in quality to OpenCV’s version. Results were saved for comparison.
