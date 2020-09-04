# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

# profile import
from profile import *

# utility import
from Common import *

# 이미지 클래스 임포트
from Image import *

class Segmentor( Common ) :

    def __init__(self):
        Common.__init__(self)
    pass

    def my_image_process( self ) :

        # 이미지를 파일로 부터 RGB 색상으로 읽어들인다.
        # img_path = "../data_ocr/sample_01/messi5.png"
        # img_path = "../data_ocr/sample_01/hist_work_01.png"
        # img_path = "../data_ocr/sample_01/gosu_01.png"

        sentence = "오늘 비교적 온화한 날씨가"
        img_path = "../data_ocr/sample_01/sample_21.png"

        sentence = "가제 제안하다 호박 현대인"
        img_path = "../data_ocr/sample_01/sample_100.png"

        # action = "line extract"
        #img_path = "../data_yegan/ex_01/_1018877.JPG"
        #img_path = "../data_yegan/ex_01/1-56.JPG"

        # TODO    원천 이미지 획득

        img_org = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR order

        # 이미지 높이, 넓이, 채널수 획득
        height = img_org.shape[0]
        width = img_org.shape[1]
        channel_cnt = img_org.shape[2]

        log.info(f"Image path: {img_path}")
        log.info(f"Image width: {width}, height: {height}, channel: {channel_cnt}")

        image_org = Image( img_org )
        image_org.save_img_as_file( img_path, "org" )
        title = f'Original Image: { img_path.split("/")[-1] }'
        0 and image_org.plot_image( title = title , border_color = "green" )

        grayscale = image_org.convert_to_grayscale()
        grayscale.reverse_image( max = 255 )
        grayscale.save_img_as_file( img_path, "grayscale" )
        grayscale.plot_image( title="Grayscale", border_color = "green" )
        grayscale.plot_histogram()

        gs_avg = grayscale.average( )
        gs_std = grayscale.std( )
        sg_max = grayscale.max( )

        log.info( f"grayscale avg = {gs_avg}, std = {gs_std}" )
        #-- grayscale 변환

        if 1 :
            # TODO 잡음 제거
            ksize = 5
            noise_removed = grayscale.remove_noise( algorithm="gaussian blur", ksize = ksize )
            curr_image = noise_removed
            noise_removed.save_img_as_file( img_path, f"noise_removed({curr_image.algorithm})" )

            title = f"Noise removed ({curr_image.algorithm}, ksize={ksize})"
            noise_removed.plot_image( title=title, border_color = "blue" )
            noise_removed.plot_histogram()

            curr_image = noise_removed
        pass

        if 1 :
            # TODO 평활화
            normalized = curr_image.normalize_image_by_histogram()
            normalized.save_img_as_file( img_path, "image_normalized" )
            normalized.plot_image( title = "Normalization", border_color = "green" )
            normalized.plot_histogram()

            curr_image = normalized
        pass

        if 0 :
            # 라플라시안
            laplacian = curr_image.laplacian(ksize=5)
            laplacian.save_img_as_file(img_path, laplacian.algorithm)
            laplacian.plot_image(title="Laplacian", border_color="green")
            laplacian.plot_histogram()

            curr_image = laplacian

            # 평활화 2
            normalized2 = curr_image.normalize_image_by_histogram()
            normalized2.save_img_as_file(img_path, "laplacian_normalized")
            normalized2.plot_image(title="Laplacian Normalization", border_color="green")
            normalized2.plot_histogram()

            curr_image = normalized2
        pass

        #TODO 이진화
        #algorithm = "threshold_otsu"
        #algorithm = "threshold_isodata"
        #algorithm = "threshold_balanced"
        #algorithm = "threshold_adaptive_gaussian"
        #algorithm = "threshold_adaptive_mean"
        algorithm = "threshold_global"

        bin_image = curr_image.threshold(algorithm=algorithm)
        curr_image = bin_image
        if bin_image.reverse_required :
            bin_image = bin_image.reverse_image()
        pass

        #bin_image.reverse_image()

        bin_image.save_img_as_file( img_path, f"image_binarized({curr_image.algorithm})" )
        title = f"Binarization ({curr_image.algorithm})"
        bin_image.plot_image( title=title, border_color = "blue" )
        bin_image.plot_histogram()
        #-- 이진화

        if 0 :
            # TODO morphology
            morphology = bin_image.morphology( is_open = 0, bsize = 3, iterations = 1, kernel_type="cross" )
            morphology.save_img_as_file( img_path, morphology.algorithm )
            morphology.plot_image( title=morphology.algorithm, border_color = "blue" )
            morphology.plot_histogram()

            bin_image = morphology
        pass # -- morphology

       #TODO   Y 축 데이터 히스토그램

        vertical_histogram = bin_image.get_vertical_histogram(ksize = 1)
        bin_image.plot_vertical_histogram(vertical_histogram, sentence)
        bin_image.save_vertical_hist_as_excel(img_path, vertical_histogram, show_excel_file  = 0 )

        word_segments = bin_image.word_segements( vertical_histogram, sentence )

        save = bin_image.dimension_ratio() > 3
        if save :
            # 세그먼테이션 파일 저장
            for idx, word_segment in enumerate( word_segments ) :
                word_segment.save_img_as_file( img_path, f"word_{idx:02d}" )
            pass
        pass

        log.info( "Plot show....." )

        print_prof_last()

        plt.show()

        log.info( "Good bye!")
    pass
pass # -- Segmentor

if __name__ == '__main__':
    segmentor = Segmentor()

    # 현재 파일의 폴더로 실행 폴더를 이동함.
    segmentor.chdir_to_curr_file()

    segmentor.my_image_process()
pass # -- main

# end