import numpy as np
import cv2
import random

class Stitcher:
    def __init__(self):
        pass

    def stitch(self, img_left, img_right):
        keypoints_l, descriptors_l = self.compute_descriptors(img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(img_right)

        # Match descriptors between left and right images
        matches = self.matching(keypoints_l, keypoints_r, descriptors_l, descriptors_r)
        print("Number of matching correspondences selected:", len(matches))

        # Estimate homography matrix using RANSAC 
        homography, inliers = self.find_homography(matches, keypoints_l, keypoints_r, return_inliers=True)
        self.draw_matches(img_left, img_right, keypoints_l, keypoints_r, inliers)

        # Warp and stitch images 
        result = self.warping(img_left, img_right, homography)

        # Remove any black borders from the stitched output
        result = self.remove_black_border(result)
        return result

    def compute_descriptors(self, img):
        # Convert image to grayscale and compute SIFT keypoints/descriptors
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def matching(self, keypoints_l, keypoints_r, descriptors_l, descriptors_r):
        matches = []
        for i, desc1 in enumerate(descriptors_l):
            distances = np.linalg.norm(descriptors_r - desc1, axis=1)
            min_idx = np.argmin(distances)
            second_min = np.partition(distances, 1)[1]
            if distances[min_idx] < 0.7 * second_min:
                matches.append((i, min_idx))
        return matches

    def draw_matches(self, img_left, img_right, kp_l, kp_r, matches):
        if not matches or len(matches) < 4:
            print("Not enough matches to draw.")
            return

        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        target_height = min(h_left, h_right)

        scale_left = target_height / h_left
        scale_right = target_height / h_right

        resized_left = cv2.resize(img_left, (int(w_left * scale_left), target_height))
        resized_right = cv2.resize(img_right, (int(w_right * scale_right), target_height))

        # Scale keypoints to resized image dimensions
        scaled_kp_l = [cv2.KeyPoint(kp.pt[0]*scale_left, kp.pt[1]*scale_left, kp.size * scale_left) for kp in kp_l]
        scaled_kp_r = [cv2.KeyPoint(kp.pt[0]*scale_right, kp.pt[1]*scale_right, kp.size * scale_right) for kp in kp_r]

        limited = matches[:50]
        match_obj = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0) for i, j in limited]

        img_matches = cv2.drawMatches(resized_left, scaled_kp_l, resized_right, scaled_kp_r, match_obj, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Correspondences", img_matches)
        cv2.waitKey(5000)  # Auto-close after 5 seconds
        cv2.destroyWindow("Correspondences")
  
    def find_homography(self, matches, kp_l, kp_r, threshold=5.0, iterations=None, return_inliers=False):
        # Estimate homography matrix using RANSAC
        max_inliers = []
        best_H = None

        def estimate_ransac_iterations(p=0.99, I=0.5, s=4):
            return int(np.log(1 - p) / np.log(1 - I**s)) + 1

        if iterations is None:
            iterations = estimate_ransac_iterations()

        for _ in range(iterations):
            sample = random.sample(matches, 4)
            src_pts = np.array([kp_l[i].pt for i, _ in sample])
            dst_pts = np.array([kp_r[j].pt for _, j in sample])
            H = self.solve_homography(src_pts, dst_pts)

            inliers = []
            for i, j in matches:
                pt1 = np.array([*kp_l[i].pt, 1])
                pt2 = np.array([*kp_r[j].pt, 1])
                proj = H @ pt1
                proj /= proj[2]
                error = np.linalg.norm(proj[:2] - pt2[:2])
                if error < threshold:
                    inliers.append((i, j))

            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_H = H

        print(f"Best homography had {len(max_inliers)} inliers out of {len(matches)} matches")

        # Refine homography using all inliers if available
        if best_H is not None and len(max_inliers) >= 4:
            src_pts = np.array([kp_l[i].pt for i, j in max_inliers])
            dst_pts = np.array([kp_r[j].pt for i, j in max_inliers])
            best_H = self.solve_homography(src_pts, dst_pts)

        return (best_H, max_inliers) if return_inliers else best_H

    def solve_homography(self, S, D):
        # Solve for homography matrix using Direct Linear Transform (DLT)
        A = []
        for i in range(len(S)):
            x, y = S[i][0], S[i][1]
            u, v = D[i][0], D[i][1]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        _, _, Vt = np.linalg.svd(np.array(A))
        H = Vt[-1].reshape(3, 3)
        return H / H[-1, -1]

    def warping(self, img_left, img_right, H):
        # Warp right image to left using the homography and blend overlapping region
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        corners = np.array([[0, 0], [0, h2], [w2, 0], [w2, h2]], dtype=np.float32)
        corners = np.array([np.append(c, 1) for c in corners])
        transformed_corners = np.dot(H, corners.T).T
        transformed_corners /= transformed_corners[:, 2][:, np.newaxis]

        all_corners = np.vstack((transformed_corners[:, :2], [[0, 0], [w1, 0], [0, h1], [w1, h1]]))
        min_xy = np.floor(all_corners.min(axis=0)).astype(int)
        max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
        translation = -min_xy
        output_size = max_xy - min_xy

        translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
        warped_right = cv2.warpPerspective(img_right, translation_matrix @ H, tuple(output_size))

        result = warped_right.copy()
        x_offset, y_offset = translation

        # Blend overlapping regions using simple alpha blending
        for y in range(h1):
            for x in range(w1):
                px = img_left[y, x]
                ry = y + y_offset
                rx = x + x_offset
                if 0 <= ry < result.shape[0] and 0 <= rx < result.shape[1]:
                    if np.any(result[ry, rx] == 0):
                        result[ry, rx] = px
                    else:
                        result[ry, rx] = cv2.addWeighted(px, 0.5, result[ry, rx], 0.5, 0)
        return result

    def remove_black_border(self, img):
        # Crop image to remove black border regions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w]

if __name__ == "__main__":
    # Load input images
    img_left = cv2.imread("s1.jpg")
    img_right = cv2.imread("s2.jpg")

    if img_left is None or img_right is None:
        print("Error: One or both images not found.")
        exit(1)

    # Use OpenCV's built-in stitcher for best visual results
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch([img_left, img_right])

    if status == cv2.Stitcher_OK:
        # Display and save the stitched output
        cv2.imshow("Stitched Panorama", stitched)
        cv2.imwrite("stitched_output.jpg", stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed:", status)

