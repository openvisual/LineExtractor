# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, cv2, numpy as np, sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

log.info( "Done Import.".center( 80, "*") )

log.info( "Pwd 1: %s" % os.getcwd())
# change working dir to current file
dirname = os.path.dirname(__file__)
if dirname : 
    os.chdir( dirname )
    log.info( "Pwd 2: %s" % os.getcwd())
pass

# 원천 이미지 획득
# 이미지를 파일로 부터 RGB 색상으로 읽어들인다.
#img_path = '../data_opencv/messi5.jpg'
img_path = "../data_ocr/sample_01/sample_11.png"

img_org = cv2.imread( img_path, cv2.IMREAD_COLOR ) #BGR order

# 이미지 높이, 넓이, 채널수 획득 
height      = img_org.shape[0]
width       = img_org.shape[1]
channel_no  = img_org.shape[2]

print( "Image path: %s" % img_path )
print( "Image widh: %s, height: %s, channel: %s" % (width,height,channel_no ) )

fig = plt.figure(figsize=(10, 10), constrained_layout=True)

# org img, channel img, gray scale, median blur, histogram, bin, y_count
gs_row_cnt = 7 
gs_col_cnt = 3

gs_row = -1 
gs_col = 0 

gridSpec = GridSpec( gs_row_cnt, gs_col_cnt, figure=fig )

# pyplot ax 의 프레임 경계 색상 변경 
def change_ax_border_color( ax, color ) :
    for spine in ax.spines.values():
        spine.set_edgecolor( color ) 
    pass
pass

if 1 : # 원본 이미지 표출
    gs_row += 1 
    gs_col = 0 
    colspan = gs_col_cnt
    img = img_org

    title = 'Original Image: %s' % ( img_path.split("/")[-1] )

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))
    img_show = ax.imshow( img )
    ax.set_xlabel( 'x\n %s' % title )
    ax.set_ylabel( 'y', rotation=0 )  

    change_ax_border_color( ax, "green" )

    fig.colorbar(img_show, ax=ax)
pass

#-- 원천 이미지 획득

# 채널 분리 
# b, g, r 채널 획득
# cv2.imread() 는 b, g, r 순서대로 배열에서 반환한다.
b_channel = img_org[:,:,0].copy() 
g_channel = img_org[:,:,1].copy()
r_channel = img_org[:,:,2].copy()

channels = [ r_channel, g_channel, b_channel ]

if 1 :  # 채널 이미지 표출
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

# RGB -> GrayScale 변환 공식
print( "Grayscale" )

# grayscale 변환
grayscale = np.empty( ( height, width ), dtype='f') 

for y, row in enumerate( grayscale ) :
    for x, _ in enumerate( row ) :
        # average  Y = (R + G + B / 3)
        # weighted Y = (0.3 * R) + (0.59 * G) + (0.11 * B)
        # Colorimetric conversion Y = 0.2126R + 0.7152G  0.0722B
        # OpenCV CCIR Y = 0.299 R + 0.587 G + 0.114 B
        gs = 0.299*r_channel[y][x] + 0.587*g_channel[y][x] + 0.114*b_channel[y][x]
        # 이미지 역전, 입력 이미지가 흰 바탕에 검정색으로 가정 
        gs = 255 - gs
        if gs < 0 : 
            gs = 0 
        pass
        grayscale[y][x] = gs
    pass
pass
# -- grayscale 변환

if 1 : # 그레이 스케일 이미지 표출
    gs_row += 1 
    gs_col = 0 
    colspan = gs_col_cnt
    img = grayscale
    cmap = "gray"
    title = "Grayscale"
    
    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    img_show = ax.imshow( img, cmap=cmap )    
    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'y', rotation=0 ) 

    change_ax_border_color( ax, "green" )

    fig.colorbar(img_show, ax=ax)
pass #-- 그레이 스케일 이미지 표출

gs_avg = np.average( grayscale )
gs_std = np.std( grayscale )
sg_max = np.max( grayscale )

print( "grayscale avg = %s, std = %s" % (gs_avg, gs_std))
#-- grayscale 변환

# 잡음 제거를 위한 Median Blur Filter

print( "Noise Ellimination" )

noise_removed = np.empty( ( height, width ), dtype='f') 

ksize = 5

target_image = grayscale

for y in range( height ) : 
    for x in range( width ) :
        window = target_image[ y : y + ksize, x : x + ksize ]
        median = np.median( window )
        noise_removed[y][x] = median
    pass
pass

if 1 : # 잡음 제거  이미지 표출
    gs_row += 1 
    gs_col = 0 
    colspan = gs_col_cnt
    img = noise_removed
    cmap = "gray"
    title = "Noise removed (Median Blur)"
    
    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    img_show = ax.imshow( img, cmap=cmap )    
    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'y', rotation=0 ) 

    change_ax_border_color( ax, "blue" ) 

    fig.colorbar(img_show, ax=ax)
pass #-- 잡음 제거  이미지 표출

#-- 잡음 제거를 위한 Median Blur Filter

# histogram 생성 
print( "Histogram" )
# calculate histogram count
histogram = np.zeros( 256, dtype='u8' )

target_image = noise_removed

for y, row in enumerate( target_image ) : 
    for x, gs in enumerate( row ) :
        gs = (int)(round(gs))
        histogram[ gs ] += 1
    pass
pass
#-- calculate histogram

hist_avg = np.average( histogram )
hist_std = np.std( histogram )
hist_max = np.max( histogram )

print( "hist avg = %s, std = %s" % (hist_avg, hist_std)) 

if 1 : # 히스토 그램 표출
    gs_row += 1 
    gs_col = 0 
    colspan = gs_col_cnt
    
    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    charts = { }
    
    # histogram bar chart
    y = histogram
    x = [i for i, _ in enumerate( y ) ]
    charts["count"] = ax.bar( x, y, width=0.5, color='green', align='center', alpha=1.0)

    # histogram std chart
    x = [gs_avg - gs_std, gs_avg + gs_std]
    y = [ hist_max*0.95, hist_max*0.95 ]
    charts["std"] = ax.fill_between( x, y, color='cyan', alpha=0.5 ) 
    
    # histogram average chart
    x = [ gs_avg, ]
    y = [ hist_max, ]
    charts["average"] = ax.bar(x, y, width=1, color='blue', align='center', alpha=0.5) 

    loc = "upper right"

    if gs_avg > 122 :
        loc = "upper left"
    pass

    if 1 : # 레전드 표출
        t = ( charts["count"], charts["std"], charts["average"], )
        l = ( "count", "std", "average", )
        ax.legend( t, l, loc=loc, shadow=True)
    pass #-- 레전드 표출 

    if 1 : # x 축 최대, 최소 설정 
        min_x = gs_avg - gs_std*1.2

        if min_x < 0 :
            min_x = 0 
        pass

        ax.set_xlim( min_x, 255 ) 
    pass

    title = "Histogram"
    ax.set_xlabel( 'GrayScale\n%s' % title )
    ax.set_ylabel( 'Count', rotation=90 ) 
pass #-- 히스토 그램 표출

#-- histogram 생성 

# 이진화

print( "Binarization" )

threshold = gs_avg

print( "Threshold: %s" % threshold )

# 이진화 계산 
binarized = np.empty( ( height, width ), dtype='B')

target_image = noise_removed

for y, row in enumerate( target_image ) :
    for x, gs in enumerate( row ) :
        gs = round( gs )
        binarized[y][x] = (0, 1,)[ gs >= threshold ]
        '''
        if gs >= threshold :
            bin[y][x] = 1 
        else :
            bin[y][x] = 0
        pass
        '''
    pass
pass 
# -- 이진화 계산 

if 1 : # 이진 이미지 표출
    gs_row += 1 
    gs_col = 0 
    colspan = gs_col_cnt
    title = "Binarization (threshold=%s)" % threshold 
    img = binarized
    cmap = "gray"

    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))
    img_show = ax.imshow( img, cmap=cmap )
    
    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'y', rotation=0 ) 

    change_ax_border_color( ax, "green" )

    fig.colorbar(img_show, ax=ax)
pass #-- 이진 이미지 표출 

#-- 이진화

# y count 표출

print( "Y count" )

target_image = binarized

y_counts = np.zeros( ( width, ), dtype='B')
ksize = 1

for x in range( width ) :
    window = target_image[ 0 : height , x : x + ksize ] 
    count_signal = np.count_nonzero( window == 1 ) # 검정 바탕 흰색 카운트
    # count_signal = np.count_nonzero( window == 0 ) # 흰 바탕 검정 카운트
    y_counts[x] = count_signal
pass

if 1 : # y count 표출 
    gs_row += 1 
    gs_col = 0 
    colspan = gs_col_cnt
    
    ax = plt.subplot(gridSpec.new_subplotspec((gs_row, gs_col), colspan=colspan))

    # 이진 이미지 표출
    img = binarized
    cmap = "gray" 
    img_show = ax.imshow( img, cmap=cmap )
    fig.colorbar(img_show, ax=ax)
    #-- 이진 이미지 표출

    charts = { }
    
    # y count bar chart
    y = y_counts
    x = [i for i, _ in enumerate( y ) ]
    charts["y count"] = ax.bar( x, y, width=0.5, color='red', align='center', alpha=1.0) 

    if 0 : # 레전드 표출
        t = ( charts["y count"] , )
        l = ( "y count", )
        loc = "upper right"    
        ax.legend( t, l, loc=loc, shadow=True)
    pass #-- 레전드 표출  

    title = "y count"
    ax.set_xlabel( 'x\n%s' % title )
    ax.set_ylabel( 'Y count', rotation=90 ) 

    ax.set_xlim( 0, width ) 
pass #-- y count 표출

#-- y count 표출 

plt.show()

# end 