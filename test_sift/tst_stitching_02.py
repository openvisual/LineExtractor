# coding: utf-8

import cv2
import imutils
import numpy as np


class Stitcher:

    def __init__(self):
        pass
    pass

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        m = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if not m : return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = m
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        result = result / 2

        result[0:imageB.shape[0], 0:imageB.shape[1]] += imageB/2

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

            # return a tuple of the stitched image and the
            # visualization
            return result, vis
        pass

        # return the stitched image
        return result

    pass

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect keypoints in the image
        detector = cv2.SIFT_create()
        kps = detector.detect(gray)

        # extract features from the image
        extractor = cv2.SIFT_create()
        (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return kps, features

    pass

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    pass

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]

        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
            pass
        pass

        # return the visualization
        return vis
    pass

    def boundary_cut(self, result):
        # transform the panorama image to grayscale and threshold it
        result = result.astype(np.uint8)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result_thresh = cv2.threshold(result_gray, 0, 255, cv2.THRESH_BINARY)[1]

        import imutils
        # Finds contours from the binary image
        cnts = cv2.findContours(result_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # get the maximum contour area
        c = max(cnts, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        result = result[y:y + h, x:x + w]

        return result
    pass
pass

if __name__ == '__main__':

    print( "Hello ..." )

    import os

    img_path = "../data_yegan/set_01/_1018843.JPG"

    os.system( f'cp {img_path} c:/temp')
    image1 = cv2.imread( img_path )

    img_path = "../data_yegan/set_01/_1018844.JPG"

    os.system(f'cp {img_path} c:/temp')
    image2 = cv2.imread(img_path)

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([image1, image2], showMatches=True)

    cv2.imwrite( 'c:/temp/tmp_stitch_result.jpg', result )

    result = stitcher.boundary_cut( result )

    cv2.imwrite('c:/temp/tmp_stitch_boundary.jpg', result)

    print( "Good bye!")

pass
