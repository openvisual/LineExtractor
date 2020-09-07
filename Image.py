# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, numpy as np, sys, time, math, inspect
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functools import cmp_to_key

import cv2, cv2 as cv

# utility import
from Common import *

from Line import *
from LineList import *

class Image (Common) :

    # 이미지 저장 회수
    img_save_cnt = 0
    clear_work_files = 0

    gs_row_cnt = 4
    gs_col_cnt = 7

    gs_row = -1
    gs_col = 0

    gridSpec = None
    plt_windows_cnt = 0

    use_matplot = True

    def __init__(self, img, algorithm="" ):
        # 2차원 이미지 배열 데이터
        self.img = img
        self.algorithm = algorithm
        self.histogram = None
        self.histogram_acc = None
        self.fileName = None
    pass

    def img_file_name(self, img_path, work):
        # C:/temp 폴더에 결과 파일을 저정합니다.

        directory = "C:/temp"

        if os.path.exists(directory):
            if not os.path.isdir(directory):
                os.remove(directory)
                os.mkdir(directory)
            pass
        else:
            os.mkdir(directory)
        pass

        img_save_cnt = Image.img_save_cnt

        fileName = img_path

        fileBase = os.path.basename(fileName)

        fileHeader, ext = os.path.splitext( fileBase )
        ext = ext.lower()

        if Image.clear_work_files and img_save_cnt == 0 :
            # fn_hdr 로 시작하는 모든 파일을 삭제함.
            import glob
            for f in glob.glob( f"{fileHeader}*" ):
                log.info( f"file to delete {f}")
                try :
                    os.remove(f)
                except Exception as e:
                    error = str( e )
                    log.info( f"cannot file to delete. error: {error}" )
                pass
            pass
        pass

        fn = os.path.join( directory, f"{fileHeader}_{img_save_cnt:02d}_{work}{ext}" )

        fn = fn.replace("\\", "/")

        log.info( f"fn={fn}")

        Image.img_save_cnt += 1

        return fn
    pass  # -- img_file_name

    @profile
    def save_img_as_file(self, img_path, work, ):
        fileName = self.img_file_name( img_path, work)
        img = self.img

        cmap = "gray"

        if len( img ) == 3 :
            cmap = None
        pass

        plt.imsave(fileName, img, cmap=cmap)

        self.fileName = fileName

        log.info( f"Image saved as file name[ {fileName} ]" )

        return fileName
    pass  # -- save_img_as_file

    ''' -- 이미지 저장 함수 '''

    # pyplot ax 의 프레임 경계 색상 변경
    def change_ax_border_color(self, ax, color):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
        pass

    pass  # -- change_ax_border_color

    def plot_image( self, title="", border_color="black", qtUi=None, mode=None ):

        if qtUi is not None :
            qtUi.plot_image( title=title, border_color=border_color, image=self, mode=mode )
        else :
            return self.plot_image_by_matplot( title, border_color )
        pass
    pass # -- plot_image

    def plot_image_by_matplot(self, title="", border_color="black"):
        # TODO 이미지 그리기

        Image.gs_row += 1

        if Image.gridSpec is None or Image.gs_row >= Image.gs_row_cnt :
            Image.gs_row = 0

            Image.fig = plt.figure(figsize=(13, 10), constrained_layout=True)
            Image.gridSpec = GridSpec(Image.gs_row_cnt, Image.gs_col_cnt, figure=Image.fig)

            Image.plt_windows_cnt += 1

            cnt = Image.plt_windows_cnt

            # plot 윈도우 제목 변경
            win_title = "Segmentation"
            plt.get_current_fig_manager().canvas.set_window_title( f"{win_title} {cnt}" )
        pass

        gs_col = 1
        colspan = Image.gs_col_cnt - gs_col

        ax = plt.subplot(Image.gridSpec.new_subplotspec((Image.gs_row, gs_col), colspan=colspan))

        img = self.img

        cmap = "gray"

        if len( img ) == 3 :
            cmap = None
        pass

        img_show = ax.imshow(img, cmap=cmap )

        ax.set_xlabel(title)
        ax.set_ylabel('y', rotation=0)

        border_color and self.change_ax_border_color(ax, border_color)

        Image.fig.colorbar(img_show, ax=ax)

        return ax, img_show

    pass  # -- plot_image_by_matplotlib

    def plot_histogram(self, qtUi=None, mode="A" ):  # 히스토 그램 표출
        if qtUi is not None :
            qtUi.plot_histogram( image=self, mode=mode )
        else :
            self.plot_histogram_by_matplot()
        pass
    pass

    def plot_histogram_by_matplot(self):  # 히스토 그램 표출
        img = self.img
        h = len( img )
        w = len( img[0] )

        gs_col = 0
        colspan = 1

        ax = plt.subplot(Image.gridSpec.new_subplotspec((Image.gs_row, gs_col), colspan=colspan))

        histogram = self.make_histogram()
        histogram_acc = self.accumulate_histogram( histogram )

        max_y = 0

        if len( histogram ) > 10 :
            sum = histogram_acc[ - 1 ]
            f_10_sum = np.sum( histogram[ 0 : 10 ] )

            if f_10_sum > sum*0.8 :
                max_y = np.max( histogram[ 10 : ] )
            pass
        pass

        hist_avg = np.average(histogram)
        hist_std = np.std(histogram)
        hist_med = np.median(histogram)

        log.info( f"hist avg = {hist_avg}, std = {hist_std}, med={hist_med}" )

        gs_avg = self.average()
        gs_max = self.max()
        gs_std = self.std()

        charts = {}

        if len( histogram ) > 10 :
            # histogram bar chart
            y = histogram
            x = range(len(y))

            width = 2

            import matplotlib.colors as mcolors

            clist = [(0, "blue"), (0.125, "green"), (0.25, "yellow"), (0.5, "cyan"), (0.7, "orange"), (0.9, "red"), (1, "blue")]
            rvb = mcolors.LinearSegmentedColormap.from_list("", clist)

            clist_ratio = len( clist )/np.max( y )

            #charts["count"] = ax.bar(x, y, width=width, color=rvb(y*clist_ratio ) )

            charts["count"] = ax.bar(x, y, width=width, color='g', alpha=1.0)
        else :
            y = histogram
            x = range(len(y))
            width = 1

            charts["count"] = ax.bar(x, y, width=width, color='g', alpha=1.0)
        pass

        if 1:
            # accumulated histogram
            y = histogram_acc
            x = range( len(y) )

            charts["accumulated"] = ax.plot(x, y, color='r', alpha=1.0)
        pass

        if 0 :
            # histogram std chart
            x = [gs_avg - gs_std, gs_avg + gs_std]
            y = [hist_max * 0.95, hist_max * 0.95]

            charts["std"] = ax.fill_between(x, y, color='cyan', alpha=0.5)
        pass

        if 0:
            # histogram average chart
            x = [ gs_avg ]
            y = [ hist_max ]

            charts["average"] = ax.bar(x, y, width=0.5, color='b', alpha=0.5)
        pass

        if 0:  # 레전드 표출
            t = []
            l = list(charts.keys())
            l = sorted(l)
            for k in l:
                t.append(charts[k])
            pass

            for i, s in enumerate(l):
                s = remove_space_except_first( s )
                l[i] = s[:4]
            pass

            loc = "upper right"

            if gs_avg > 122:
                loc = "upper left"
            pass

            ax.legend(t, l, loc=loc, shadow=True)
        pass  # -- 레전드 표출

        if 1 :
            xlim_fr = 0
            xlim_to = len( histogram )
            xlim_to = xlim_to if xlim_to > 1 else 2.5 # data visualization
            xlim_fr = xlim_fr if xlim_to > 1 else -0.5  # data visualization

            ax.set_xlim( xlim_fr, xlim_to  )

            if not max_y :
                max_y = np.max( histogram )
            pass

            if max_y > 1_000 :
                ax.set_yscale('log')
            else :
                ax.set_ylim(0, max_y )
            pass
        pass

        histo_len = len(histogram)
        if histo_len > 10 :
            ax.set_xticks( [ 0, 50, 100, 150, 200, 255 ] )
        pass

        ax.grid( 1 )
        #x.set_ylabel('count', rotation=0)
        ax.set_xlabel( "Histogram")
    pass
    # -- plot_histogram

    # TODO  통계 함수

    def average(self): # 평균
        return np.average( self.img )
    pass

    def std(self): # 표준 편차
        return np.std( self.img )
    pass

    def max(self): # 최대값
        return np.max( self.img )
    pass

    # -- 통계 함수

    # grayscale 변환 함수
    def convert_to_grayscale( self ) :
        log.info( "Convert to grayscale...." )

        img = self.img

        # TODO   채널 분리
        # b, g, r 채널 획득
        # cv2.imread() 는 b, g, r 순서대로 배열에서 반환한다.
        b_channel = img[:, :, 0].copy()
        g_channel = img[:, :, 1].copy()
        r_channel = img[:, :, 2].copy()

        # RGB -> GrayScale 변환 공식
        # average  Y = (R + G + B / 3)
        # weighted Y = (0.3 * R) + (0.59 * G) + (0.11 * B)
        # Colorimetric conversion Y = 0.2126R + 0.7152G  0.0722B
        # OpenCV CCIR Y = 0.299 R + 0.587 G + 0.114 B

        grayscale = r_channel*0.299 + g_channel*0.587 + b_channel*0.114

        grayscale = grayscale.astype( np.int16 )

        return Image( grayscale )
    pass # -- convert_to_grayscale

    def width(self):
        # image width
        img = self.img
        w = len( img [0])

        return w
    pass

    def height(self):
        # image height
        img = self.img
        h = len( img )

        return h
    pass

    def dimension(self):
        return self.width(), self.height()
    pass

    def dimension_ratio(self):
        return self.width()/self.height()
    pass

    def reverse_image( self, max=None):
        # TODO   영상 역전 함수

        log.info("Reverse image....")

        img = self.img

        if max is None:
            max = np.max(img)

            if max < 1:
                max = 1
            elif max > 1:
                max = 255
            else:
                max = 1
            pass
        pass

        self.img = max - img

        return self
    pass # -- reverse_image

    def laplacian(self, ksize=5):
        # TODO   라플라시안
        # https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html

        msg = "Laplacian"
        log.info( f"{msg} ..." )

        ksize = 2*int(ksize/2) + 1

        img = self.img

        algorithm = f"laplacian ksize={ksize}"

        img = img.astype(np.float)

        data = cv.Laplacian(img, cv.CV_64F)

        # normalize to gray scale
        min = np.min( data )
        max = np.max( data )

        data = (255/(max - min))*(data - min)

        min = np.min(data)
        max = np.max(data)
        # -- # normalize to gray scale

        data = data.astype(np.int)

        min = np.min(data)
        max = np.max(data)

        log.info( f"min = {min}, max={max}")

        log.info( f"Done. {msg}" )

        return Image( img=data, algorithm=algorithm)
    pass  # -- laplacian

    def gradient(self, ksize=5, kernel_type="cross"):
        # TODO   그라디언트
        # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html

        msg = "Gradient"
        log.info( f"{msg} ..." )

        ksize = 2*int(ksize/2) + 1

        img = self.img

        algorithm = f"Gradient ksize={ksize}, ktype={kernel_type}"

        img = img.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

        if kernel_type == "rect":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        elif kernel_type == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
        elif kernel_type == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        pass

        data = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

        log.info( f"Done. {msg}" )

        return Image( img=data, algorithm=algorithm)
    pass  # -- gradient

    def contours(self, lineWidth=1):
        # TODO  등고선
        #  https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html

        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        img = self.img

        img = img.astype(np.uint8)

        h = len( img )
        w = len( img[0] )

        #mode = cv.RETR_TREE
        mode = cv2.RETR_EXTERNAL
        method = cv.CHAIN_APPROX_SIMPLE

        algorithm = f"contours(mode={mode}, method={method})"

        # Find Canny edges
        edged = cv2.Canny( img, 30, 255)

        ( contours, c ) = cv2.findContours(edged, mode, method)

        data = np.zeros((h, w, 3), dtype="uint8")

        useFilter = True

        if useFilter :
            filters = []
            diagonal = math.sqrt( w*w + h*h )
            ref_len = diagonal*0.1
            ref_width = min( w, h )/30
            ref_height = ref_width
            ref_area = w*h/10_000

            for i, cnt in enumerate( contours ):
                valid = True
                if valid :
                    rect = cv2.minAreaRect(cnt)

                    rect_width = rect[1][0]
                    rect_height = rect[1][1]

                    valid = ( rect_width > ref_width or rect_height > ref_height )

                    log.info(f"[{i:03d}] rect valid={valid}, width = {rect_width}, height = {rect_height}")
                pass

                if valid :
                    arc_len = cv2.arcLength( cnt , 0 )
                    log.info( f"[{i:03d} contour = {arc_len}" )
                    valid = ( arc_len > ref_len )
                pass

                if valid :
                    filters.append( cnt )
                pass
            pass

            log.info(f"org contours len = {len(contours)}")
            log.info(f"filters len = {len(filters)}")

            cv2.polylines(data, filters, 0, (255, 255, 255), lineWidth)
        else :
            cv2.drawContours(data, contours, -1, (255, 255, 255), lineWidth)
        pass

        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        return Image( img=data, algorithm=algorithm)
    pass  # -- contours

    def remove_noise(self, algorithm , ksize=5 ):
        # TODO   잡음 제거
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        img = self.img

        if algorithm == "gaussian blur"  :
            # Gaussian filtering
            algorithm = f"{algorithm} ksize={ksize}"

            img = img.astype(np.uint8)
            data = cv.GaussianBlur(img, (ksize, ksize), 0)
        elif algorithm == "bilateralFilter" :
            algorithm = f"{algorithm} ksize={ksize}, 75, 75"

            img = img.astype(np.uint8)
            data = cv2.bilateralFilter(img, ksize, 75, 75)
        elif algorithm == "medianBlur" :
            algorithm = f"{algorithm} ksize={ksize}"

            data = cv2.medianBlur(img, ksize)
        pass

        return Image( img=data, algorithm=algorithm)
    pass  # -- remove_noise

    @profile
    def make_histogram(self):
        # TODO     Histogram 생성
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        img = self.img

        size = 256 if np.max( img ) > 1 else 2

        histogram = cv2.calcHist([img], [0], None, [size], [0, size])

        return histogram
    pass  # -- make_histogram

    @profile
    def accumulate_histogram(self, histogram):
        # TODO    누적 히스토 그램
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        histogram_acc = np.add.accumulate(histogram)

        return histogram_acc
    pass  # accumulate_histogram

    @profile
    def normalize_image_by_histogram(self):
        # TODO    히스토그램 평활화
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        img = self.img
        w, h = self.dimension()

        algorithm = "normalize"

        data = np.empty([h, w], dtype=img.dtype)

        cv.normalize(img, data, 0, 255, cv.NORM_MINMAX)

        image = Image( data, algorithm=algorithm )

        return image
    pass # -- normalize_image_by_histogram

    # TODO     전역 임계치 처리
    def threshold_global(self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        reverse_required = 0

        img = self.img

        h = len( img )
        w = len( img[0] )

        histogram = None

        if not hasattr(self, "histogram" ) or self.histogram is None :
            histogram , _ = self.make_histogram()
        else :
            histogram = self.histogram
        pass

        histogram = histogram.copy()

        useMargin = False
        margin = 0
        if useMargin :
            margin = 50
            histogram[0: margin] = 0
        pass

        x = np.arange(0, len(histogram))

        avg =  margin + ( sum( histogram * x )/np.sum( histogram ) )

        threshold = avg

        data = np.where( img >= threshold , 1, 0 )

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"global thresholding ({ int(threshold) })"
        image.reverse_required = reverse_required

        return image
    pass  # -- 전역 임계치 처리

    def threshold_isodata(self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        reverse_required = 0

        img = self.img

        histogram, _ = self.make_histogram()

        threshold = 0
        t_diff = None
        for i in range(2, 256):
            mL_hist = histogram[0: i]
            mL = sum(mL_hist * np.arange(0, i)) / sum(mL_hist)
            mH_hist = histogram[i:]
            mH = sum(mH_hist * np.arange(i, 256)) / sum(mH_hist)

            diff = abs(i - (mL + mH) / 2)
            if t_diff is None or diff < t_diff:
                t_diff = diff
                threshold = i
            pass
        pass

        log.info( f"Threshold isodata = {threshold}" )

        data = np.where( img >= threshold , 1, 0 )

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"threshold isodata ({ int(threshold) })"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_isodata

    def threshold_balanced( self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        reverse_required = 0

        img = self.img

        histogram, _ = self.make_histogram()

        threshold = 0
        t_diff = None
        for i in range(2, 256):
            mL_hist = histogram[0: i]
            mH_hist = histogram[i:]

            diff = abs(sum(mL_hist) - sum(mH_hist))
            if t_diff is None or diff < t_diff:
                t_diff = diff
                threshold = i
            pass
        pass

        log.info( f"Threshold balanced = {threshold}" )

        data = np.where( img >= threshold , 1, 0 )

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"threshold balanced ({ int(threshold) })"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_balanced

    # TODO     지역 평균 적응 임계치 처리
    def threshold_adaptive_mean(self, bsize=3, c=0):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        reverse_required = 1

        img = self.img

        w, h = self.dimension()

        data = np.empty((h, w), dtype='B')

        b = int(bsize / 2)
        if b < 1:
            b = 1
        pass

        for y, row in enumerate(img):
            for x, gs in enumerate(row):
                y0 = y - b
                x0 = x - b

                if y0 < 0:
                    y0 = 0
                pass

                if x0 < 0:
                    x0 = 0
                pass

                window = img[y0: y + b + 1, x0: x + b + 1]
                window_avg = np.average(window)
                threshold = window_avg - c

                data[y][x] = [0, 1][gs >= threshold]
            pass
        pass

        image = Image( data )
        image.threshold = -1
        image.algorithm = "adaptive mean"
        image.reverse_required = reverse_required

        return image
    pass  # -- 지역 평균 적응 임계치 처리

    def threshold_otsu(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https: // docs.opencv.org / 3.4 / d7 / d4d / tutorial_py_thresholding.html

        reverse_required = 0

        img = self.img
        img = img.astype(np.uint8)

        # Gaussian filtering
        #blur = cv.GaussianBlur(img, (5, 5), 0)
        # Otsu's thresholding
        threshold, data = cv.threshold( img, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"otsu threshold={threshold}"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_otsu_opencv

    # TODO     지역 가우시안 적응 임계치 처리
    def threshold_adaptive_gaussian(self, bsize=5, c=0):
        algorithm = 0

        if algorithm == 0 :
            v = self.threshold_adaptive_gaussian_opencv(bsize=bsize, c=c)
        elif algorithm == 1:
            v = self.threshold_adaptive_gaussian_my(bsize=bsize, c=c)
        pass

        return v
    pass # -- threshold_adaptive_gaussian

    def threshold_adaptive_gaussian_opencv(self, bsize=5, c=0):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

        reverse_required = 0
        bsize = 2 * int(bsize / 2) + 1

        img = self.img
        img = img.astype(np.uint8)

        data = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c)

        image = Image( data )
        image.threshold = f"bsize = {bsize}"
        image.algorithm = f"adaptive gaussian, bsize={bsize}"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_adaptive_gaussian_opencv

    def threshold_adaptive_gaussian_my(self, bsize=3, c=0):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel

        reverse_required = 1

        bsize = 2 * int(bsize / 2) + 1

        w, h = self.dimension()

        data = np.empty((h, w), dtype='B')

        b = int(bsize / 2)

        if b < 1:
            b = 1
        pass

        # the threshold value T(x,y) is a weighted sum (cross-correlation with a Gaussian window)
        # of the blockSize×blockSize neighborhood of (x,y) minus C

        img = self.img

        image_pad = np.pad(img, b, 'constant', constant_values=(0))

        def gaussian(x, y, bsize):
            #  The default sigma is used for the specified blockSize
            #sigma = bsize
            # ksize	Aperture size. It should be odd ( ksizemod2=1 ) and positive.
            sigma = 0.3 * ((bsize - 1) * 0.5 - 1) + 0.8
            ss = sigma * sigma
            pi_2_ss = 2 * math.pi * ss

            b = int(bsize / 2)

            x = x - b
            y = y - b

            v = math.exp(-(x * x + y * y) / ss) / pi_2_ss
            # g(x,y) = exp( -(x^2 + y^2)/s^2 )/(2pi*s^2)

            return v
        pass  # -- gaussian

        def gaussian_sum(window, bsize):
            gs_sum = 0

            # L = len( window )*len( window[0] )
            for y, row in enumerate(window):
                for x, v in enumerate(row):
                    gs_sum += v * gaussian(y, x, bsize)
                pass
            pass

            return gs_sum

        pass  # -- gaussian_sum

        bsize_square = bsize*bsize

        for y, row in enumerate(image_pad):
            for x, gs in enumerate(row):
                if (b <= y < len(image_pad) - b) and (b <= x < len(row) - b):
                    window = image_pad[y - b: y + b + 1, x - b: x + b + 1]

                    gaussian_avg = gaussian_sum(window, bsize)/bsize_square

                    threshold = gaussian_avg - c

                    data[y - b][x - b] = [0, 1][gs >= threshold]
                pass
            pass
        pass

        image = Image(data)
        image.threshold = f"bsize = {bsize}"
        image.algorithm = "adaptive gaussian thresholding my"
        image.reverse_required = reverse_required

        return image
    pass  # -- 지역 가우시안 적응 임계치 처리

    def threshold(self, algorithm):
        # TODO 이진화

        v = None

        if "otsu" in algorithm :
            v = self.threshold_otsu()
        elif "gaussian" in algorithm :
            w, h = self.dimension()

            bsize = w if w > h else h
            bsize = bsize / 6
            bsize = 13

            v = self.threshold_adaptive_gaussian(bsize=bsize, c=0)
        elif "mean" in algorithm :
            bsize = 5
            v = self.threshold_adaptive_mean(bsize=bsize, c=0)
        elif "global" in algorithm :
            v = self.threshold_global()
        elif "isodata" in algorithm :
            v = self.threshold_isodata()
        elif "balanced" in algorithm:
            v = self.threshold_balanced()
        pass

        return v
    pass # -- binarize_image

    def morphology(self, is_open, bsize = 5, iterations = 1, kernel_type = "cross" ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        bsize = 2*int( bsize/2 ) + 1

        img = self.img
        img = img.astype(np.uint8)

        data = img

        if iterations < 1 :
            iterations = 1
        pass

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bsize, bsize))

        if kernel_type == "rect" :
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bsize, bsize))
        elif kernel_type == "cross" :
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (bsize, bsize))
        elif kernel_type == "ellipse" :
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bsize, bsize))
        pass

        for _ in range ( iterations ) :
            if is_open :
                data = cv2.erode( data, kernel, iterations = 1)
            else :
                data = cv2.dilate(data, kernel, iterations=1)
            pass

            if is_open :
                data = cv2.dilate( data, kernel, iterations=1)
            else :
                data = cv2.erode(data, kernel, iterations=1)
            pass
        pass

        op_close = "open" if is_open else "close"

        image = Image(data)
        image.algorithm = f"morphology, {op_close}, kernel={kernel_type}, bsize={bsize}, iterations={iterations}"

        return image
    pass  # -- morphology_closing

    def extract_lines(self, merge_lines=1, img_path=""):
        # hough line 추출
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        fileBase = os.path.basename(img_path)

        img = self.img

        h = len( img )
        w = len( img[0] )

        img = img.astype(np.uint8)

        if np.max( img ) < 2 :
            img = img*255
        pass

        '''
        rho – r 값의 범위 (0 ~ 1 실수)
        theta – 𝜃 값의 범위(0 ~ 180 정수)
        threshold – 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
        minLineLength – 선의 최소 길이. 이 값보다 작으면 reject.
        maxLineGap – 선과 선사이의 최대 허용간격. 이 값보다 작으며 reject.
        '''
        diagonal = math.sqrt(w * w + h * h)

        threshold = 100
        minLineLength = int( diagonal/50 )
        maxLineGap = 20

        lines_org = cv.HoughLinesP(img, 1, np.pi/180, threshold, lines=None, minLineLength=minLineLength, maxLineGap=maxLineGap )

        lines = []
        for line in lines_org :
            lines.append( Line( line = line[0], fileBase=fileBase ) )
        pass

        error_deg = 2
        snap_dist = int( diagonal/150 )

        if merge_lines :
            lines = Line.merge_lines(lines, error_deg=error_deg, snap_dist=snap_dist)
        pass

        lines = sorted( lines, key=cmp_to_key(Line.compare_line_length))
        lines = lines[ : : -1 ]

        for line in lines :
            line.fileBase = fileBase
        pass

        algorithm = f"hough lines(thresh={threshold}, legth={minLineLength}, gap={maxLineGap}, merge={merge_lines}, error_deg={error_deg}, snap={snap_dist})"

        lineList = LineList( lines = lines, algorithm = algorithm, w = w, h = h, fileBase = fileBase )

        return lineList
    pass # extract_lines

    def plot_lines(self, lineList ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        lines = lineList.lines
        algorithm = lineList.algorithm

        # colors
        colors = []

        if 1 :
            import matplotlib.colors as mcolors
            from matplotlib.colors import hex2color, rgb2hex

            color_dict = mcolors.BASE_COLORS
            color_dict = mcolors.TABLEAU_COLORS

            for name, hex_color in enumerate(color_dict):
                color = hex2color(hex_color)
                color = tuple([int(255 * x) for x in color])

                colors.append(color)
                #log.info(f"{name} = {hex_color} = {color}")
            pass
        pass

        colors_len = len(colors)
        # -- colors

        img = self.img

        h = len( img )
        w = len( img[0] )

        img = img.astype(np.uint8)

        if np.max( img ) < 2 :
            img = img*255
        pass

        data = cv.cvtColor( img, cv.COLOR_GRAY2BGR )
        data = data*0

        diagonal = math.sqrt(w * w + h * h)

        radius = int( diagonal/600 )
        radius = radius if radius > 5 else 5
        thickness = 3

        for i, line in enumerate( lines ) :
            color = colors[i % colors_len]
            thickness = line.thickness()

            a = line.a
            b = line.b

            cv.line(data, (a.x, a.y), (b.x, b.y), color, thickness=thickness, lineType=cv.LINE_AA)

            cv.circle(data, (a.x, a.y), radius, color, thickness=thickness, lineType=8)
            cv.circle(data, (b.x, b.y), radius, color, thickness=thickness, lineType=8)
        pass

        image = Image(data)
        image.algorithm = algorithm

        log.info(f"Done. {inspect.getframeinfo(inspect.currentframe()).function}")

        return image
    pass # plot_lines

pass
# -- class Image