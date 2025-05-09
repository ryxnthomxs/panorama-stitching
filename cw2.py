import numpy as np
import cv2

class Stitcher:
    def __init__(self):
        pass
    
    def stitch(self, img_left, img_right, ...): # Add input arguments as you deem fit
        '''
            The main method for stitching two images
        '''

        # Step 1 - extract the keypoints and features with a suitable feature
        # detector and descriptor
        keypoints_l, descriptors_l = self.compute_descriptors(img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(img_right)
                
        # Step 2 - Feature matching. You will have to apply a selection technique
        # to choose the best matches
        matches = self.matching(keypoints_l, keypoints_r, 
                                     descriptors_l, descriptors_r, ...) # Add input arguments as you deem fit

        print("Number of matching correspondences selected:", len(matches))
                
        # Step 3 - Draw the matches connected by lines
        self.draw_matches(img_left, img_right, matches)
        
        # Step 4 - fit the homography model with the RANSAC algorithm
        homography = self.find_homography(matches)

        # Step 5 - Warp images to create the panoramic image
        result = self.warping(img_left, img_right, homography, ...) # Add input arguments as you deem fit

        return result
    
    def compute_descriptors(self, img):
        '''
        The feature detector and descriptor
        '''

        # Your code here
        
        return keypoints, features
    
    def matching(keypoints_l, keypoints_r, descriptors_l, descriptors_r, ...): 
        # Add input arguments as you deem fit
        '''
            Find the matching correspondences between the two images
        '''

        # Your code here. You should also implement a step to select good matches.
            
        return good_matches 
    
    def draw_matches(self, img_left, img_right, matches):
        '''
            Connect correspondences between images with lines and draw these
            lines 
        '''
        
        # Your code here
      
        cv2.imshow('correspondences',img_with_correspondences)
        cv2.waitKey(0)

    def find_homography(self, matches): 
        ''' 
        Fit the best homography model with the RANSAC algorithm. 
        '''

        # Your code here 
        # Use the method solve_homography(source_points,
        # destination_points) from the class Homography in your implementation
        # of the RANSAC algorithm
        
        return homography
    
    def warping(img_left, img_right, homography, ...) # Add input arguments as you deem fit
        '''
           Warp images to create panoramic image
        '''

        # Your code here. You will have to warp one image into another via the
        # homography. Remember that the homography is an entity expressed in
        # homogeneous coordinates. 
        
        return result
    
    def remove_black_border(self, img):
        '''
        Remove black border after stitching
        '''
        return cropped_image

class Blender:
    def linear_blending(self, ...):
        '''
        linear blending (also known as feathering)
        '''
       
        return linear_blending_img

    def customised_blending(self, ...):
        '''
        Customised blending of your choice
        '''
        return customised_blending_img

class Homography:
    def solve_homography(self, S, D):
        '''
        Find the homography matrix between a set of S points and a set of
        D points
        '''
        
        # Your code here. You might want to use the DLT algorithm developed in cw1. 

        return H

if __name__ == "__main__":
    
    # Read the image files
    img_left = # your code here
    img_right = # your code here
    
    stitcher = Stitcher()
    result = stitcher.stitch(img_left, img_right, ...) # Add input arguments as you deem fit

    # show the result
    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
