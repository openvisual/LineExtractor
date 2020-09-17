# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, datetime
from Common import *

# 이미지 클래스 임포트
from Image import *

class LineExtractor ( Common ):

    def __init__(self):
        Common.__init__( self )
    pass

    @profile
    def my_line_extract(self, img_path, qtUi = None, mode="A", lineListA=None) :
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

        log.info(f"Image width: {width}, height: {height}, channel: {channel_cnt}")

        image_org = Image( img_org )
        image_org.save_img_as_file( img_path, "org" )
        title = f'Original Image: { img_path.split("/")[-1] }'
        0 and image_org.plot_image(title=title, cmap=None, border_color = "green", qtUi=qtUi, mode=mode)

        curr_image = image_org

        if 1 : # -- grayscale 변환
            #grayscale = curr_image.to_grayscale_multiotsu()
            grayscale = curr_image.grayscale()

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
            sigmaColor = 75
            sigmaSpace = 75
            noise_removed = curr_image.remove_noise( algorithm=algorithm, bsize = bsize, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace )
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
            #algorithm = "adaptive_gaussian"
            #algorithm = "adaptive_mean"
            algorithm = "global"

            if use_multi_ostus :
                algorithm = "multi_otsu"
            pass

            bin_image = curr_image.threshold(algorithm=algorithm, bsize=21, c=1, thresh=15)

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

        useCanny = True
        if useCanny:
            canny = curr_image.canny(min=0, max=255)

            curr_image = canny

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass  # -- canny

        useContour = True
        if useContour: # TODO Contour
            contour = curr_image.contours(lineWidth=2, useFilter=True)

            curr_image = contour

            curr_image.save_img_as_file(img_path, curr_image.algorithm)
            curr_image.plot_image(title=curr_image.algorithm, border_color="blue", qtUi=qtUi, mode=mode)
            curr_image.plot_histogram(qtUi=qtUi, mode=mode)
        pass

        lineList = None

        if 1 : # 허프 라인 추출
            lineList = curr_image.extract_lines( merge_lines=0, img_path=img_path )
            hough = curr_image.plot_lines( lineList )
            hough.save_img_as_file(img_path, hough.algorithm)

            error_deg = 2
            snap_dist = 15

            lineList = lineList.merge_lines(error_deg=error_deg, snap_dist=snap_dist)

            hough = curr_image.plot_lines( lineList )

            title = lineList.algorithm

            hough.save_img_as_file(img_path, title)
            hough.plot_image(title=title, border_color="blue", qtUi=qtUi, mode=mode)
        pass

        if lineList is not None and lineListA is not None :
            log.info( "Line tagging....")

            lineListIdentified = lineListA.line_identify( lineList )

            identify = curr_image.plot_lines( lineListIdentified )
            identify.save_img_as_file(img_path, "identify")
            identify.plot_image(title="identify", border_color="blue", qtUi=qtUi, mode=mode)

            lineList.lineListIdentified = lineListIdentified
        pass

        lineList.mode = mode

        return lineList
    pass
pass # -- LineExtractor

if __name__ == '__main__':

    lineExtractor = LineExtractor()

    lineExtractor.show_versions()

    lineExtractor.chdir_to_curr_file()

    from os.path import join
    from glob import glob

    files = []

    img_path = ""

    if 1 :
        img_path = "./data_yegan/set_01/_1018843.JPG"
        #img_path = "./data_yegan/set_01/_1018885.JPG"
    pass

    if img_path :
        files.append( img_path )
    else :
        folder = "./data_yegan/set_01"
        for ext in ('*.gif', '*.png', '*.jpg'):
            files.extend(glob(join( folder, ext)))
        pass
    pass

    log.info( f"file count={ len(files )}" )

    lineListAll = LineList()

    for i in range( 0 , len(files), 2 ) :
        file = files[i]

        img_path = file.replace( "\\", "/" )

        lineList = lineExtractor.my_line_extract( img_path=img_path, qtUi=None )

        nextFile = lineExtractor.next_file( img_path )

        if nextFile and nextFile is not None :
            lineList = lineExtractor.my_line_extract( img_path=nextFile, qtUi=None, lineListA=lineList )

            lineListAll.extend( lineList.lineListIdentified )
        pass
    pass

    if lineListAll :
        fileBase = os.path.basename(img_path)
        fileHeader, ext = os.path.splitext(fileBase)
        now = datetime.datetime.now()
        now_str = now.strftime('%m-%d_%H%M%S')
        now_str = now_str.split(".")[0]
        json_file_name = os.path.join("/temp", f"{fileHeader}_{now_str}.json")

        lineListAll.save_as_json(json_file_name=json_file_name)
    pass

    lineExtractor.print_profile()

    if 1 :
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

# end