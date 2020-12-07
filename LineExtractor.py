# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from os.path import join
from glob import glob

import os, datetime
from Common import *

# 이미지 클래스 임포트
from Image import *

class LineExtractor ( Common ):

    def __init__(self):
        Common.__init__( self )

        self.width = None
        self.height = None

        self.margin_width = 0
        self.margin_height = 0

        self.grayscale = None
    pass

    @profile
    def my_line_extract(self, img_path, qtUi = None, mode="A", lineListA=None, grayscale_prev=None) :
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        Image.img_save_cnt = 0

        if 1 :
            pass
        elif os.path.exists( img_path ) and os.path.isdir( img_path ):
            log.info(f"ERROR: img_path={img_path} is a directory.")
            return -1
        else :
            log.info(f"ERROR: img_path={img_path} is invalid.")
            return -2
        pass

        prev_dir = os.getcwd()

        if True :
            # 이미지 파일이 있는 폴더로 경로를 이동함.
            fileName = img_path
            directory = os.path.dirname(fileName)
            fileBaseName = os.path.basename(fileName)

            if directory :
                log.info(f"Pwd 1: {os.getcwd()}")

                prev_dir = os.getcwd()

                os.chdir( directory )
                log.info(f"Pwd 2: {os.getcwd()}")
            pass

            img_path = f"./{fileBaseName}"

            log.info(f"dir = {directory}, fileBase={fileBaseName}")
        pass

        log.info(f"img_path to read = {img_path}")

        img_org = cv2.imread(img_path, 1)

        if prev_dir :
            os.chdir( prev_dir )
            log.info(f"Pwd 3: {os.getcwd()}")
        pass

        if img_org is None :
            log.info( f"ERROR: Failed to read the image file( {img_path} ).")

            return -3
        pass

        # 이미지 높이, 넓이, 채널수 획득
        height = img_org.shape[0]
        width = img_org.shape[1]
        channel_cnt = img_org.shape[2]

        margin_ratio = 5
        margin_ratio = 0

        self.width  = width
        self.height = height

        self.margin_width  = int( width * margin_ratio/100.0 )
        self.margin_height = int( height * margin_ratio/100.0 )

        log.info(f"Image width: {width}, height: {height}, channel: {channel_cnt}")

        image_org = Image( img_org )
        image_org.save_img_as_file( img_path, "org" )
        title = f'Original Image: { img_path.split("/")[-1] }'
        0 and image_org.plot_image(title=title, cmap=None, border_color = "green", qtUi=qtUi, mode=mode)

        curr_image = image_org

        if True :
            # roi image crop
            mw = self.margin_width
            mh = self.margin_height

            img_roi = img_org[ mh : (height - mh), mw : (width - mw) ]

            height = img_roi.shape[0]
            width  = img_roi.shape[1]

            self.height = height
            self.width  = width

            image_roi = Image( img_roi )
            image_roi.save_img_as_file( img_path, "roi" )

            curr_image = image_roi
        pass # -- roi image crop

        if 1 : # -- grayscale 변환
            #grayscale = curr_image.to_grayscale_multiotsu()
            grayscale = curr_image.grayscale()

            self.grayscale = grayscale

            curr_image = grayscale

            #grayscale.reverse_image( max=255 )
            grayscale.save_img_as_file( img_path, curr_image.algorithm )
            grayscale.plot_image( title=curr_image.algorithm, border_color = "green", qtUi=qtUi, mode=mode)
            grayscale.plot_histogram(qtUi=qtUi, mode=mode)
        pass

        remove_noise = True
        if remove_noise : # TODO 잡음 제거
            algorithm = "gaussianBlur"
            algorithm = "bilateralFilter"
            bsize = 7
            sigma_color = 75
            sigma_space = 75
            noise_removed = curr_image.remove_noise( algorithm=algorithm, bsize = bsize, sigma_color=sigma_color, sigma_space=sigma_space )
            curr_image = noise_removed

            title = curr_image.algorithm

            curr_image.save_img_as_file( img_path, curr_image.algorithm )
            curr_image.plot_image(title=title, border_color = "blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass

        if 1 : # TODO 평활화
            normalized = curr_image.normalize()

            curr_image = normalized

            curr_image.save_img_as_file( img_path, "image_normalized" )
            curr_image.plot_image(title="Normalization", border_color = "green", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass

        use_multi_ostus = False

        useLaplacian = False
        if useLaplacian : # TODO Laplacian
            laplacian = curr_image.laplacian(bsize=3)

            curr_image = laplacian

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass  # -- laplacian

        useGradient = not use_multi_ostus
        useGradient = False
        if useGradient:  # TODO Gradient
            gradient = curr_image.gradient(bsize=7, kernel_type="cross")

            curr_image = gradient

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass  # -- gradient

        useThread = True
        if useThread :  # TODO 이진화
            #algorithm = "multi_otsu"
            #algorithm = "otsu"
            #algorithm = "isodata"
            #algorithm = "yen"
            #algorithm = "balanced"
            algorithm = "adaptive_gaussian"
            algorithm = "sauvola"
            #algorithm = "adaptive_mean"
            #algorithm = "global"

            if use_multi_ostus :
                algorithm = "multi_otsu"
            pass

            bsize = 11
            c = 2
            thresh = 15

            if "gaussian" in algorithm :
                bsize = 21
                c = 3
            pass

            bin_image = curr_image.threshold(algorithm=algorithm, bsize=bsize, c=c, thresh=thresh)

            curr_image = bin_image

            if curr_image.reverse_required:
                curr_image = curr_image.reverse_image()
            pass

            curr_image.save_img_as_file(img_path, f"{curr_image.algorithm}")
            title = f"{curr_image.algorithm}"
            curr_image.plot_image(title=title, border_color="blue", qtUi=qtUi, mode=mode)
        pass  # -- 이진화

        use_morphology = False
        if use_morphology:  # TODO morphology
            morphology = curr_image.morphology(is_open=1, bsize=7, iterations=3, kernel_type="cross")

            curr_image = morphology

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
        pass  # -- morphology

        useCanny = False
        if useCanny:
            canny = curr_image.canny(min=0, max=255)

            curr_image = canny

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass  # -- canny

        lineList = None
        useContour = True
        useHoughLine = True
        if useContour:
            lineWidth = 1
            lineWidth = 2

            contours = curr_image.extract_contours()
            contours_image = curr_image.draw_contours(contours, lineWidth=lineWidth)
            contours_image.save_img_as_file(img_path, contours_image.algorithm)

            contour_filtering = False
            if contour_filtering :
                contours_filtered = contours_image.filter_contours(contours)
                contours_image = contours_image.draw_contours(contours_filtered, lineWidth=lineWidth)
                contours_image.save_img_as_file(img_path, contours_image.algorithm + "_filtered")
            else :
                contours_filtered = contours
            pass

            lines_only = contours_image.filter_lines_only( contours_filtered )
            contours_image = contours_image.draw_polylines(lines_only, lineWidth=lineWidth)
            contours_image.save_img_as_file(img_path, contours_image.algorithm + "_lines_only")

            curr_image = contours_image

            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
        pass

        if useHoughLine :
            # 허프 라인 추출
            lineList = curr_image.extract_lines( merge_lines=0, img_path=img_path )
            hough = curr_image.plot_lines( lineList )
            hough.save_img_as_file(img_path, hough.algorithm)

            error_deg = 3
            snap_dist = 15

            lineList = lineList.merge_lines(error_deg=error_deg, snap_dist=snap_dist)

            hough = curr_image.plot_lines( lineList )

            title = lineList.algorithm

            hough.save_img_as_file(img_path, title)
            hough.plot_image(title=title, border_color="blue", qtUi=qtUi, mode=mode)
        pass

        if lineList is not None and lineListA is not None :
            log.info( "Line tagging....")

            min_length = int( max( [curr_image.width(), curr_image.height()] ) * 0.1 )

            similarity_min = 0.7

            lineListIdentified = lineListA.line_identify( lineList, min_length = min_length, similarity_min = similarity_min  )

            identify = curr_image.plot_lines( lineListIdentified )
            identify.save_img_as_file(img_path, f"identify(min_length={min_length}")
            identify.plot_image(title="identify", border_color="blue", qtUi=qtUi, mode=mode)

            lineList.lineListIdentified = lineListIdentified
        pass

        if grayscale_prev is not None :
            # Initiate SIFT detector
            sift = cv2.SIFT_create()

            img_1 = self.grayscale.img
            img_2 = grayscale_prev.img

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute( img_1, None )
            kp2, des2 = sift.detectAndCompute( img_2, None )

            matches = []

            if True :
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)

                flann = cv2.FlannBasedMatcher(index_params, search_params)

                # Ratio test
                matches = flann.knnMatch(des1, des2, k=2)
            pass

            dim_square = max([len(img_1), len(img_1[0])]) * 0.3
            dim_square = dim_square * dim_square

            pts_src = []
            pts_dst = []

            goods = []
            matchesMask = []

            img_1_rgb = cv2.cvtColor(img_1, cv2.COLOR_GRAY2RGB)
            img_2_rgb = cv2.cvtColor(img_2, cv2.COLOR_GRAY2RGB)

            for _, (m1, m2) in enumerate(matches):
                valid = True

                ratio = 0.7
                # ratio = 0.9

                if valid and (m1.distance > ratio * m2.distance):
                    valid = False
                pass

                if valid:
                    pt1 = kp1[m1.queryIdx].pt
                    pt2 = kp2[m1.trainIdx].pt

                    dx = pt1[0] - pt2[0]
                    dy = pt1[1] - pt2[1]

                    if dx * dx + dy * dy > dim_square:
                        valid = False
                    pass

                    if valid:
                        pts_src.append(pt1)
                        pts_dst.append(pt2)

                        goods.append([m1, m2])
                        matchesMask.append([1, 0])

                        print(i, pt1, pt2)

                        if 1 or (i % 5 == 0):
                            # Draw pairs in purple, to make sure the result is ok
                            cv2.circle(img_1_rgb, (int(pt1[0]), int(pt1[1])), 20, (255, 255, 0), 8)
                            cv2.circle(img_2_rgb, (int(pt2[0]), int(pt2[1])), 20, (0, 255, 255), 8)
                        pass
                    pass
                pass
            pass

            curr_image.save_img_as_file(img_path, f"sift_a.jpg", img=img_1_rgb)

            curr_image.save_img_as_file(img_path, f"sift_b.jpg", img=img_2_rgb)

            draw_params = dict(matchColor=(255, 0, 0), singlePointColor=(0, 0, 255), matchesMask=matchesMask, flags=0)

            img3 = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, goods, None, **draw_params)

            curr_image.save_img_as_file(img_path, "sift_match.jpg", img=img3 )

            h, status = cv2.findHomography(np.array(pts_src), np.array(pts_dst))

            img4 = cv2.warpPerspective(img_1_rgb, h, (img_2_rgb.shape[1], img_2_rgb.shape[0]))

            curr_image.save_img_as_file(img_path, "sift_homograpy.jpg", img=img4)
        pass

        lineList.mode = mode

        return lineList
    pass # -- my_line_extract

pass # -- LineExtractor

if __name__ == '__main__':
    lineExtractor = LineExtractor()

    lineExtractor.show_versions()
    lineExtractor.chdir_to_curr_file()

    files = []

    #img_path = "./data_yegan/set_01"
    img_path = "./data_yegan/set_06/data4.JPG"
    img_path = "./data_yegan/set_06/DJI_0146.JPG"
    img_path = "./data_yegan/set_06/IMG_0129.JPG"

    img_path = "./data_yegan/set_05"
    img_path = "./data_yegan/set_04"

    img_path = "./data_yegan/set_04/P1010015.JPG"
    img_path = "./data_yegan/set_01/_1018843.JPG"

    if not os.path.isdir( img_path ) :
        files.append(img_path)
    else :
        folder = img_path

        for ext in [ '*.gif', '*.png', '*.jpg' ]:
            files.extend(glob.glob(join(folder, ext)))
        pass
    pass

    log.info( f"file count={ len(files )}" )

    lineListMatched = LineList()

    len_files = len( files )
    for i in range( 0 , len_files, 2 ) :
        file = files[i]

        img_path = file.replace( "\\", "/" )

        log.info( "" )
        log.info( "*"*80 )
        log.info( f"[{i:04d}] [{100*i/len_files:.1f} %] {img_path}" )
        log.info("*" * 80)
        log.info("")

        lineListA = lineExtractor.my_line_extract( img_path=img_path, qtUi=None )

        nextFile = lineExtractor.next_file( img_path )

        if nextFile is not None :
            grayscale_prev = lineExtractor.grayscale
            lineListB = lineExtractor.my_line_extract( img_path=nextFile, qtUi=None, lineListA=lineListA, grayscale_prev=grayscale_prev )

            lineListMatched.extend(lineListB.lineListIdentified)
        pass
    pass

    if lineListMatched :
        fileBase = os.path.basename(img_path)
        fileHeader, ext = os.path.splitext(fileBase)
        now = datetime.datetime.now()
        now_str = now.strftime('%m-%d_%H%M%S')
        now_str = now_str.split(".")[0]
        json_file_name = os.path.join( "C:/temp", f"z{fileHeader}_{now_str}.json")

        width = lineExtractor.width
        height = lineExtractor.height

        mw = lineExtractor.margin_width
        mh = lineExtractor.margin_height

        save = True

        save and lineListMatched.save_as_json(json_file_name=json_file_name, width=width, height=height, mw=mw, mh=mh )
    pass

    lineExtractor.print_profile()

    if 1:
        # 결과창 폴더 열기
        folder = "c:/temp"
        lineExtractor.open_file_or_folder(folder)

        #plt.show()
    pass

    usePlot = False
    if usePlot :
        log.info("Plot show.....")
        plt.show()
    pass

    log.info("Good bye!")
pass # -- main