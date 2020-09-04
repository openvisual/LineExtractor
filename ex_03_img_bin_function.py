# -*- coding: utf-8 -*-

# 변경 사항
# 함수 모듈화
# 히스토그램 정규화 추가
# 결과 이미지 저장
# Adaptive Thresholding 추가

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

''' profile functions '''

# -- usage
# @profile
# def your_function(...):
#       ....
#
# your_function( ... )
# print_prof_data()

import time
from functools import wraps

PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        pass

        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling
pass # -- profile(fn)

def print_prof_data():
    for fname, data in PROF_DATA.items() :
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        fmt = "Function[ %s ] called %d times. Exe. time max: %.3f, average: %.3f"
        log.info( fmt % (fname, data[0], max_time, avg_time) )
    pass

    PROF_DATA.clear()
pass # -- print_prof_data()

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
pass # -- clear_prof_data

''' --- profile functions '''

import os, cv2, numpy as np, sys, time
import math
from math import pi

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
log.info( "Done Import.".center( 80, "*") )

# 현재 파일의 폴더로 실행 폴더를 이동함.
print( "Pwd 1: %s" % os.getcwd())
# change working dir to current file
dirname = os.path.dirname(__file__)
if dirname :
    os.chdir( dirname )
    print( "Pwd 2: %s" % os.getcwd())
pass
#-- 현재 파일의 폴더로 실행 폴더를 이동함.

# 이미지를 파일로 부터 RGB 색상으로 읽어들인다.
#img_path = "../data_ocr/sample_01/messi5.png"
#img_path = "../data_ocr/sample_01/hist_work_01.png"
#img_path = "../data_ocr/sample_01/gosu_01.png"
img_path = "../data_ocr/sample_01/sample_21.png"
#img_path = "../data_yegan/ex_01/_1018877.JPG"

#TODO     이미지 저장 함수
img_save_cnt = 0

def img_file_name( work ) :
    global img_save_cnt
    img_save_cnt += 1

    fn = img_path

    root = fn[ : fn.rfind( "/") ]

    folder = root + "/temp"

    if os.path.exists( folder ):
        if not os.path.isdir( folder ) :
            os.remove( folder )
            os.mkdir( folder )
        else :
            # do nothing
            pass
        pass
    else :
        os.mkdir( folder )
    pass

    fn = fn.replace( root , "" )
    k = fn.rfind( "." )
    fn = folder + fn[ : k ] + ( "_%02d_" % img_save_cnt) + work + fn[k:]
    return fn
pass # -- img_file_name

def save_img_as_file( work, img, cmap="gray"):
    fn = img_file_name( work )

    plt.imsave( fn, img, cmap=cmap )

    log.info( "Image saved as file name[%s]" % fn )
pass # -- save_img_as_file

# -- 이미지 저장 함수

# pyplot ax 의 프레임 경계 색상 변경
def change_ax_border_color( ax, color ) :
    for spine in ax.spines.values():
        spine.set_edgecolor( color )
    pass
pass #-- change_ax_border_color

def plot_image( image, title = "", cmap="gray", border_color = "black" ) :
    # 그레이 스케일 이미지 표출
    global gs_row
    gs_row += 1
    gs_col = 0
    colspan = gs_col_cnt - gs_col

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    h = len( image )
    w = len( image[0] )

    img_show = ax.imshow( image, cmap=cmap, extent=[0, w, 0, h] )

    ax.set_xlabel( title )
    ax.set_ylabel( 'y', rotation=0 )

    border_color and change_ax_border_color( ax, border_color )

    fig.colorbar(img_show, ax=ax)

    return ax, img_show
pass # -- plot_image

#TODO    원천 이미지 획득

img_org = cv2.imread( img_path, cv2.IMREAD_COLOR ) #BGR order

# 이미지 높이, 넓이, 채널수 획득
height      = img_org.shape[0]
width       = img_org.shape[1]
channel_no  = img_org.shape[2]

print( "Image path: %s" % img_path )
print( "Image widh: %s, height: %s, channel: %s" % (width,height,channel_no ) )

save_img_as_file( "org", img_org )

fig = plt.figure(figsize=(10, 10), constrained_layout=True)

# org img, channel img, gray scale, median blur, histogram, bin, y_count
gs_row_cnt = 6
gs_col_cnt = 1

gs_row = -1
gs_col = 0

gridSpec = GridSpec( gs_row_cnt, gs_col_cnt, figure=fig )

title = 'Original Image: %s' % ( img_path.split("/")[-1] )
border_color = "green"
plot_image( img_org, title=title, cmap=None, border_color = border_color )

#-- 원천 이미지 획득

#TODO   채널 분리
# b, g, r 채널 획득
# cv2.imread() 는 b, g, r 순서대로 배열에서 반환한다.
b_channel = img_org[:,:,0].copy()
g_channel = img_org[:,:,1].copy()
r_channel = img_org[:,:,2].copy()

channels = [ r_channel, g_channel, b_channel ]

if 0 :  # 채널 이미지 표출
    gs_row += 1
    gs_col = 0
    colspan = 1

    for i, channel in enumerate( channels ):
        img_temp = np.zeros( (height, width, 3), dtype='uint8' )
        img_temp[ :, : , i ] = channel
        img = img_temp

        ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))
        img_show = ax.imshow( img )
        ax.set_ylabel( 'y', rotation=0 )

        title = "red channel"
        if i == 1 :
            title = "green channel"
        elif i == 2 :
            title = "blue channel"
        pass

        ax.set_xlabel( 'x\n%s' % title )

        ( i == 2 ) and fig.colorbar(img_show, ax=ax)

        gs_col += colspan
    pass
pass

#-- 채널 분리

#TODO    Grayscale 변환

# grayscale 변환 함수
def convert_to_grayscale( channels ) :
    log.info( "Convert to grayscale...." )

    r_channel = channels[ 0 ]
    g_channel = channels[ 1 ]
    b_channel = channels[ 2 ]

    data = r_channel*0.299 + g_channel*0.587 + b_channel*0.114

    data = data.astype( np.int16 )

    '''
    h = len( r_channel ) # image height
    w = len( r_channel[0] ) # image width

    data = np.empty( ( h, w ), dtype='f')

    for y in range( h ) :
        for x in range( w ) :
            # RGB -> GrayScale 변환 공식
            # average  Y = (R + G + B / 3)
            # weighted Y = (0.3 * R) + (0.59 * G) + (0.11 * B)
            # Colorimetric conversion Y = 0.2126R + 0.7152G  0.0722B
            # OpenCV CCIR Y = 0.299 R + 0.587 G + 0.114 B
            gs = 0.299*r_channel[y][x] + 0.587*g_channel[y][x] + 0.114*b_channel[y][x]
            data[y][x] = gs
        pass
    pass
    '''

    return data
pass
# -- grayscale 변환

#TODO   영상 역전 함수
def reverse_image( image , max = None ) :
    log.info( "Reverse image...." )

    h = len( image ) # image height
    w = len( image[0] ) # image width

    if max is None :
        max = np.max( image )

        if max < 1 :
            max = 1
        elif max > 1 :
            max = 255
        else :
            max = 1
        pass
    pass

    data = max - image

    '''
    data = np.empty( ( h, w ), dtype=image.dtype )

    for y, row in enumerate( image ) :
        for x, v in enumerate( row ) :
            data[y][x] = max - v
        pass
    pass
    '''

    return data
pass
# -- 영상 역전 함수

# grayscale 변환
grayscale = convert_to_grayscale( channels )
# 영상 역전
grayscale = reverse_image( grayscale )

save_img_as_file( "grayscale", grayscale )

#-- 그레이 스케일 이미지 표출
plot_image( grayscale, title="Grayscale", cmap="gray", border_color = "green" )

gs_avg = np.average( grayscale )
gs_std = np.std( grayscale )
sg_max = np.max( grayscale )

log.info( "grayscale avg = %s, std = %s" % (gs_avg, gs_std))
#-- grayscale 변환

#TODO   잡음 제거
#Median Blur Filter 적용

# 잡음 제거 함수
def remove_noise( image, ksize = 3 ) :
    msg = "Remove noise"
    log.info( "%s ..." % msg )

    algorithm = "cv.bilateralFilter"

    if "cv.bilateralFilter" == algorithm :
        log.info( "cv.bilateralFilter(img,ksize,75,75)" )

        image = image.astype(np.uint8)

        data = cv2.bilateralFilter( image,ksize,75,75)
    elif "cv2.medianBlur" == algorithm:
        log.info( "cv2.medianBlur( image, ksize )" )
        data = cv2.medianBlur( image, ksize )
    else :
        h = len( image ) # image height
        w = len( image[0] ) # image width

        b = int( ksize/2 )

        data = np.empty( [h, w], dtype=image.dtype )

        idx = 0
        for y in range( height ) :
            for x in range( width ) :
                y0 = y - b
                x0 = x - b

                if y0 < 0 :
                    y0 = 0
                pass

                if x0 < 0 :
                    x0 = 0
                pass

                window = image[ y0 : y + b + 1, x0 : x + b + 1 ]
                median = np.median( window )
                data[y][x] = median

                0 and log.info( "[%05d] data[%d][%d] = %.4f" % (idx, y, x, median) )
                idx += 1
            pass
        pass
    pass

    log.info( "Done. %s" % msg )

    return data , algorithm
pass #-- 잡음 제거 함수

ksize = 3
grayscale, algorithm = remove_noise( grayscale, ksize = ksize )

save_img_as_file( "noise_removed(%s)" % algorithm, grayscale )

title = "Noise removed (%s, ksize=%s)" % ( algorithm, ksize, )
border_color = "blue"
ax, show_img = plot_image( grayscale, title=title, cmap="gray", border_color = border_color )

#-- 잡음 제거를 위한 Median Blur Filter

#TODO     Histogram 생성

@profile
def make_histogram( grayscale ) :
    # this code is too slow
    msg = "Make histogram ..."
    log.info( msg )

    histogram = [ 0 ]*256

    for row in grayscale :
        for gs in row :
            histogram[ gs ] += 1
        pass
    pass

    histogram = np.array( histogram, 'u8' )

    log.info( "Done. %s" % msg )
    return histogram
pass # -- calculate histogram

# TODO    누적 히스토 그램
@profile
def accumulate_histogram( histogram ) :
    msg = "Accumulate histogram ..."
    log.info( msg )

    acc = np.add.accumulate( histogram )

    log.info( "Done. %s" % msg )

    return acc
pass # 누적 히스트 그램

histogram = make_histogram( grayscale )
histogram_acc = accumulate_histogram( histogram )

def show_histogram_on_plot( ax, image, histogram , histogram_acc ): # 히스토 그램 표출
    #global gs_row

    if 0 :
        x = range(300)
        ax.plot(x, x, '--', linewidth=2, color='firebrick')
    pass

    h = len( image )
    w = len( image[0] )

    charts = { }

    hist_avg = np.average( histogram )
    hist_std = np.std( histogram )
    hist_max = np.max( histogram )

    log.info( "hist avg = %s, std = %s" % (hist_avg, hist_std) )

    if 0 and histogram_acc is not None :
        # accumulated histogram
        y = histogram_acc

        max_y = np.max( y )

        x = [i for i, _ in enumerate( y ) ]
        charts["accumulated"] = ax.bar( x, y, width=0.4, color='yellow', alpha=0.3 )
    pass

    # histogram bar chart
    if 1 :
        y = histogram
        x = [i for i, _ in enumerate(y)]

        #charts["count"] = ax.plot( x, y, linewidth=2, color='green', alpha=0.5 )
        charts["count"] = ax.bar(x, y, width=2, color='green', alpha=0.5)
    pass

    if 0 :
        # histogram std chart
        x = [ gs_avg - gs_std, gs_avg + gs_std ]
        y = [ hist_max*0.95, hist_max*0.95 ]
        charts["std"] = ax.fill_between( x, y, color='cyan', alpha=0.5 )
    pass

    if 0 :
        # histogram average chart
        x = [ gs_avg, ]
        y = [ hist_max, ]
        charts["average"] = ax.bar(x, y, width=0.5, color='blue', alpha=0.5)
    pass

    if 0 : # 레전드 표출
        t = [ ]
        l = list( charts.keys() )
        l = sorted( l )
        for k in l :
            t.append( charts[ k ] )
        pass

        for i, s in enumerate( l ) :
            import re
            s = s[0] + re.sub(r'[aeiou]', '', s[1:])
            l[i] = s[:4]
        pass

        loc = "upper right"

        if gs_avg > 122 :
            loc = "upper left"
        pass

        ax.legend( t, l, loc=loc, shadow=True)
    pass #-- 레전드 표출

    if 0 : # x 축 최대, 최소 설정
        max_x = gs_avg + gs_std*1.2

        ax.set_xlim( 0, max_x )
    pass

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
pass #-- 히스토 그램 표출

0 and show_histogram_on_plot( ax, grayscale, histogram, histogram_acc  )

#-- histogram 생성

#TODO    히스토그램 평활화

@profile
def normalize_image_by_histogram( image, histogram_acc ) :
    msg = "Normalize histogram"
    log.info( "%s ..." % msg )

    # https://en.wikipedia.org/wiki/Histogram_equalization

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( [h, w], dtype=image.dtype )

    MN = h*w
    L = len( histogram_acc )

    cdf = histogram_acc

    #cdf_min = cdf[ 0 ]
    cdf_min = cdf[ 0 ]
    for c in cdf :
        if cdf_min == 0 :
            cdf_min = c
        else :
            break
        pass
    pass

    log.info( f"cdf_min = {cdf_min:,d}" )

    idx = 0
    L_over_MN_cdf_min = L/(MN - cdf_min + 0.0)

    cdf = np.array( cdf, 'float' )

    cdf -= cdf_min
    cdf *= L_over_MN_cdf_min
    cdf += 0.5

    for y, row in enumerate( image ):
        for x, gs in enumerate( row ):
            #data[y][x] = int( (cdf[gs] - cdf_min)*L_over_MN_cdf_min + 0.5 )

            data[y][x] = int( cdf[gs] )

            0 and log.info( "[%05d] gs = %d, v=%0.4f" % ( idx, gs, v ) )
            idx += 1
        pass
    pass

    log.info( "Done. %s" % msg )

    return data
pass #-- normalize_image_by_histogram old

image_normalized = normalize_image_by_histogram( grayscale, histogram_acc )

print_prof_data()

save_img_as_file( "image_normalized", image_normalized )

title = "Normalization"
border_color = "green"
plot_image( image_normalized, title=title, cmap="gray", border_color = border_color )

#-- 히스토그램 평활화

#TODO    이진화

#TODO     전역 임계치 처리
def threshold_golobal( image, threshold = None ):
    log.info( "Global Threshold" )

    reverse_required = 0

    if not threshold :
        threshold= np.average( image )
    pass

    log.info( "Threshold = %s" % threshold )

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( ( h, w ), dtype='B')

    for y, row in enumerate( image ) :
        for x, gs in enumerate( row ) :
            gs = round( gs ) # 반올림.
            data[y][x] = [0, 1][ gs >= threshold ]
        pass
    pass

    return data, threshold, "global thresholding", reverse_required
pass # -- 전역 임계치 처리

#TODO     지역 평균 적응 임계치 처리
def threshold_adaptive_mean( image, bsize = 3, c = 0 ):
    log.info( "Apdative threshold mean" )

    reverse_required = 1

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( ( h, w ), dtype='B')

    b = int( bsize/2 )
    if b < 1 :
        b = 1
    pass

    for y, row in enumerate( image ) :
        for x, gs in enumerate( row ) :
            y0 = y - b
            x0 = x - b

            if y0 < 0 :
                y0 = 0
            pass

            if x0 < 0 :
                x0 = 0
            pass

            window = image[ y0 : y + b + 1, x0 : x + b + 1 ]
            window_avg = np.average( window )
            threshold = window_avg - c

            data[y][x] = [0, 1][ gs >= threshold ]
        pass
    pass

    return data, -1, "adaptive mean thresholding", reverse_required
pass # -- 지역 평균 적응 임계치 처리

#TODO     지역 가우시안 적응 임계치 처리
def threshold_adaptive_gaussian( image, bsize = 3, c = 0 ):
    if 1 :
        v = threshold_adaptive_gaussian_opencv( image, bsize = bsize, c = c )
    else :
        v = threshold_adaptive_gaussian_my( image, bsize = bsize, c = c )
    pass

    return v
pass

def threshold_adaptive_gaussian_opencv( image, bsize = 3, c = 0 ):
    msg = "Apdative threshold gaussian opencv"
    log.info( msg )

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

    reverse_required = 1

    bsize = 2*int( bsize/2 )  + 1

    image = image.astype(np.uint8)

    data = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c)

    return data, ("bsize = %s" % bsize), "adaptive gaussian thresholding opencv", reverse_required
pass # -- 지역 가우시안 적응 임계치 처리

def threshold_adaptive_gaussian_my( image, bsize = 3, c = 0 ):
    log.info( "Apdative threshold gaussian my" )

    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel

    reverse_required = 1

    bsize = 2*int( bsize/2 )  + 1

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.empty( ( h, w ), dtype='B')

    b = int( bsize/2 )
    if b < 1 :
        b = 1
    pass

    # the threshold value T(x,y) is a weighted sum (cross-correlation with a Gaussian window)
    # of the blockSize×blockSize neighborhood of (x,y) minus C

    image_pad= np.pad(image, b,'constant', constant_values=(0))

    def gaussian(x, y, bsize) :
        #  The default sigma is used for the specified blockSize
        sigma = bsize
        # ksize	Aperture size. It should be odd ( ksizemod2=1 ) and positive.
        # sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        ss = sigma*sigma
        pi_2_ss = 2*math.pi*ss

        b = int( bsize/2 )

        x = x - b
        y = y - b

        v = math.exp( -(x*x + y*y)/ss )/pi_2_ss
        # g(x,y) = exp( -(x^2 + y^2)/s^2 )/(2pi*s^2)

        return v
    pass #-- gaussian

    def gaussian_sum( window, bsize ) :
        gs_sum = 0

        # L = len( window )*len( window[0] )
        for y, row in enumerate( window ) :
            for x, v in enumerate( row ) :
                gs_sum += v*gaussian( y, x, bsize )
            pass
        pass

        return gs_sum
    pass #-- gaussian_sum

    for y, row in enumerate( image_pad ) :
        for x, gs in enumerate( row ) :
            if ( b <= y < len(image_pad) - b ) and ( b<= x < len(row) - b ):
                window = image_pad[ y - b : y + b + 1 , x - b : x + b + 1 ]

                threshold = gaussian_sum( window, bsize ) - c

                data[y - b][x - b] = [0, 1][ gs >= threshold ]
            pass
        pass
    pass

    return data, ("bsize = %s" % bsize), "adaptive gaussian thresholding my", reverse_required
pass # -- 지역 가우시안 적응 임계치 처리

#TODO      이진화 계산
def binarize_image( image, threshold = None ):
    v = None

    if 1 :
        h = len(image)  # image height
        w = len(image[0])  # image width
        bsize = w if w > h else h
        bsize = bsize/2

        bsize = 5 # for line detection
        v = threshold_adaptive_gaussian( image, bsize = bsize, c = 5 )
    elif 0 :
        bsize = 3
        v = threshold_adaptive_mean( image, bsize = bsize, c = 0 )
    else :
        threshold = np.average( image )*1.1
        v = threshold_golobal( image, threshold = threshold )
    pass

    return v
pass #-- 이진화 계산

image_binarized, threshold, thresh_algo, reverse_required = binarize_image( image = grayscale )

if reverse_required :
    image_binarized = reverse_image( image_binarized )
pass

save_img_as_file( "image_binarized(%s)" % thresh_algo, image_binarized )

title = "Binarization (%s, %s)" % ( thresh_algo, threshold )
border_color = "blue"
plot_image( image_binarized, title=title, cmap="gray", border_color = border_color )

#-- 이진화

#TODO   Y 축 데이터 히스토그램

def count_y_axis_signal( image, ksize ) :
    msg = "y axis signal count"
    log.info( msg )

    h = len( image ) # image height
    w = len( image[0] ) # image width

    data = np.zeros( w, dtype='B')
    ksize = 1

    for x in range( w ) :
        window = image[ 0 : h , x : x + ksize ]
        signal_count = np.count_nonzero( window == 1 ) # 검정 바탕 흰색 카운트
        # signal_count = np.count_nonzero( window == 0 ) # 흰 바탕 검정 카운트
        data[x] = signal_count
    pass

    log.info( "Done. %s" % msg )
    return data
pass #-- count_y_axis_signal

target_image = image_binarized
y_counts = count_y_axis_signal( image= target_image, ksize = 1 )

# y count 데이터를 csv 파일로 저장

y_counts.tofile( "./y_count.csv", sep=',', format='%s')

if 1 : # y count 표출

    title = "y count"
    border_color = "blue"
    ax, img_show = plot_image(image_binarized, title=title, cmap="gray", border_color=border_color)

    charts = { }

    # y count bar chart
    y = y_counts
    x = [i for i, _ in enumerate( y ) ]
    charts["y count"] = ax.bar( x, y, width=0.5, color='red', align='center', alpha=1.0)

    if 0 : # 레전드 표출
        t = [ ]
        l = list( charts.keys() )
        l = sorted( l )
        for k in l :
            t.append( charts[ k ] )
        pass
        loc = "upper right"
        ax.legend( t, l, loc=loc, shadow=True)
    pass #-- 레전드 표출

    ax.set_ylabel( 'Y count', rotation=90 )
    ax.set_xlim( 0, width )
pass #-- y count 표출

#-- y count 표출

log.info( "Plot show....." )

plt.show()

log.info( "Good bye!")

# end