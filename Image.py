# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, numpy as np, sys, time, math, inspect
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functools import cmp_to_key

import cv2, cv2 as cv
from skimage import filters

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
        self.fileName = None
        self.reverse_required = False
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

        if Image.clear_work_files and img_save_cnt == -1 :
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

        histogram = self.histogram()
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
    
    def diagonal(self):
        w, h = self.dimension()
        return math.sqrt( w*w + h*h )
    pass

    def dimension(self):
        return self.width(), self.height()
    pass

    def dimension_ratio(self):
        return self.width()/self.height()
    pass

    @profile
    def grayscale(self):
        # grayscale 변환 함수
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        algorithm = "grayscale"
        img = self.img

        img = img.astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image = Image(gray)
        image.algorithm = algorithm

        return image
    pass  # -- to_grayscale

    @profile
    def reverse_image( self, max=None):
        # TODO   영상 역전 함수

        log.info(inspect.getframeinfo(inspect.currentframe()).function)

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

    @profile
    def laplacian(self, bsize=7):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # TODO   라플라시안
        # https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html

        bsize = 2 * int(bsize / 2) + 1

        img = self.img

        algorithm = f"laplacian bsize={bsize}"

        img = img.astype(np.float)

        laplacian = cv.Laplacian(img, cv.CV_64F)

        w, h = self.dimension()

        data = np.empty([h, w], dtype=img.dtype)

        cv.normalize(laplacian, data, 0, 255, cv.NORM_MINMAX)

        return Image(img=data, algorithm=algorithm)
    pass  # -- laplacian

    @profile
    def gradient(self, bsize=5, kernel_type="cross"):
        # TODO   그라디언트
        # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html

        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        bsize = 2*int(bsize/2) + 1

        img = self.img

        algorithm = f"gradient(bsize={bsize}, ktype={kernel_type})"

        img = img.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bsize, bsize))

        if kernel_type == "rect":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bsize, bsize))
        elif kernel_type == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (bsize, bsize))
        elif kernel_type == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bsize, bsize))
        pass

        data = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

        use_scale = True
        if use_scale :
            # normalize to gray scale
            min = np.min( data )
            max = np.max( data )

            data = (255/(max - min))*(data - min)

            #data = data.astype(np.int)

            log.info( f"min = {min}, max={max}")
        pass

        return Image( img=data, algorithm=algorithm)
    pass  # -- gradient

    @profile
    def canny(self, min=5, max=255):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        debug = False

        img = self.img

        img = img.astype(np.uint8)

        max_img = np.max(img)

        if max_img <= 1 :
            min = 0
            max = 1
        pass

        algorithm = f"canny(min={min}, max={max})"

        data = cv2.Canny(img, min, max )

        return Image(img=data, algorithm=algorithm)

    pass  # -- canny

    @profile
    def contours(self, lineWidth=1, useFilter = True):
        # TODO  등고선
        #  https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html

        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        debug = False

        img = self.img

        img = img.astype(np.uint8)

        h = len( img )
        w = len( img[0] )

        mode = cv.RETR_TREE
        #mode = cv2.RETR_EXTERNAL
        method = cv.CHAIN_APPROX_SIMPLE
        method = cv.CHAIN_APPROX_TC89_L1
        method = cv.CHAIN_APPROX_TC89_KCOS

        algorithm = f"contours(mode={mode}, method={method})"

        max_img = np.max( img )

        edged = img

        contours = []

        if cv.__version__ in [ "4.4.0" ] :
            ( contours_cv, _ ) = cv2.findContours(edged, mode, method)
            contours = contours_cv
        else :
            ( _, contours_cv, _) = cv2.findContours(edged, mode, method)
            contours = contours_cv
        pass

        data = np.zeros((h, w, 3), dtype="uint8")

        useFilter = True
        if useFilter :
            filters = []
            diagonal = math.sqrt( w*w + h*h )
            ref_len = diagonal*0.1
            ref_width = min( w, h )/30
            ref_height = ref_width
            ref_area = w*h/10_000

            for i, contour in enumerate( contours ):
                valid = True
                if valid :
                    rect = cv2.minAreaRect(contour)

                    rect_width = rect[1][0]
                    rect_height = rect[1][1]

                    valid = ( rect_width > ref_width or rect_height > ref_height )

                    debug and log.info(f"[{i:03d}] rect valid={valid}, width = {rect_width}, height = {rect_height}")
                pass

                if valid :
                    arc_len = cv2.arcLength( contour , 0 )
                    debug and log.info( f"[{i:03d} contour = {arc_len}" )
                    valid = ( arc_len > ref_len )
                pass

                if valid :
                    filters.append( contour )
                pass
            pass

            log.info(f"org contours len = {len(contours)}")
            log.info(f"filters len = {len(filters)}")

            cv2.polylines(data, filters, 0, (255, 255, 255), thickness=lineWidth, lineType=cv2.LINE_AA )
        else :
            cv2.drawContours(data, contours, -1, (255, 255, 255), thickness=lineWidth, lineType=cv2.LINE_AA )
        pass

        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        _, data = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

        return Image( img=data, algorithm=algorithm)
    pass  # -- contours

    @profile
    def remove_noise(self, algorithm , bsize=5 , sigmaColor=75, sigmaSpace=75):
        # TODO   잡음 제거
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        img = self.img

        if algorithm == "gaussianBlur"  :
            # Gaussian blur
            algorithm = f"{algorithm} bsize={bsize}"

            img = img.astype(np.uint8)
            data = cv.GaussianBlur(img, (bsize, bsize), 0)
        elif algorithm == "bilateralFilter" :
            algorithm = f"{algorithm} bsize={bsize}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}"

            img = img.astype(np.uint8)
            data = cv2.bilateralFilter(img, bsize, sigmaColor, sigmaSpace)
        elif algorithm == "medianBlur" :
            algorithm = f"{algorithm} bsize={bsize}"

            data = cv2.medianBlur(img, bsize)
        pass

        return Image( img=data, algorithm=algorithm)
    pass  # -- remove_noise

    @profile
    def histogram(self):
        # TODO     Histogram 생성
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        img = self.img

        img = img.astype(np.uint8)

        size = 256 if np.max( img ) > 1 else 2

        histogram = cv2.calcHist([img], [0], None, [size], [0, size])

        histogram = histogram.flatten()

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
    def normalize(self):
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

    @profile
    def threshold(self, algorithm, thresh=None, bsize=None, c=None):
        import Threshold

        threshold = Threshold.Threshold( image = self )

        return threshold.threshold( algorithm = algorithm, thresh=thresh, bsize=bsize, c=c )
    pass # -- threshold

    @profile
    def morphology(self, is_open, bsize = None, iterations = 1, kernel_type = "cross" ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        if bsize is None or bsize < 1 :
            bsize = int( self.diagonal()/500 )

            if bsize < 3 :
                bsize = 3
            pass
        pass

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

    @profile
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

        if lines_org is None :
            lines_org = []
        pass

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

    @profile
    def plot_lines(self, lineList ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

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

        for i, line in enumerate( lineList ) :
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

        return image
    pass # plot_lines

pass
# -- class Image