# -*- coding: utf-8 -*-

import cv2, cv2 as cv

from Common import *
from Image import Image
from skimage import filters

class Threshold(Common):

    def __init__(self, image ):
        self.image = image
    pass

    def threshold(self, algorithm, thresh=None, bsize=5, c=0):
        # TODO 이진화

        v = None

        if "_li" in algorithm :
            v = self.threshold_li()
        elif "yen" in algorithm:
            v = self.threshold_yen()
        elif "multi_otsu" in algorithm:
            v = self.threshold_multi_otsu()
        elif "otsu" in algorithm:
            v = self.threshold_otsu()
        elif "gaussian" in algorithm:
            v = self.threshold_adaptive_gaussian(bsize=bsize, c=c)
        elif "mean" in algorithm:
            v = self.threshold_adaptive_mean(bsize=bsize, c=c)
        elif "global" in algorithm:
            v = self.threshold_global(thresh=thresh)
        elif "isodata" in algorithm:
            v = self.threshold_isodata()
        elif "balanced" in algorithm:
            v = self.threshold_balanced()
        pass

        return v

    pass  # -- threshold

    @profile
    def threshold_global(self, thresh=None ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # TODO     전역 임계치 처리

        image = self.image

        img = image.img

        if thresh is None or thresh < 1 :

            histogram = image.histogram()

            useMargin = False
            margin = 0
            if useMargin :
                margin = 50
                histogram[0: margin] = 0
            pass

            x = np.arange(0, len(histogram))

            avg = margin + ( sum( histogram * x )/np.sum( histogram ) )

            thresh = avg
        pass

        data = np.where( img >= thresh , 1, 0 )

        image = Image( data )
        image.algorithm = f"global thresholding ({ int(thresh) })"

        return image
    pass  # -- 전역 임계치 처리

    @profile
    def threshold_isodata(self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        image = self.image
        img = image.img

        histogram = image.histogram()

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

        return image
    pass  # -- threshold_isodata

    @profile
    def threshold_balanced( self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        image = self.image

        img = image.img

        histogram = image.histogram()

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

        return image
    pass  # -- threshold_balanced

    @profile
    def threshold_li(self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        def quantile_95(image):
            # you can use np.quantile(image, 0.95) if you have NumPy>=1.15
            return np.percentile(image, 95)
        pass

        image = self.image

        img = image.img

        iter_thresholds = []
        opt_threshold = filters.threshold_li(img, initial_guess=quantile_95,
                                             iter_callback=iter_thresholds.append)

        #print(len(iter_thresholds), 'examined, optimum:', opt_threshold)

        data = img > opt_threshold

        algorithm = f"threshold_li( {opt_threshold:0.0f} )"

        image = Image(data)
        image.algorithm = algorithm

        return image
    pass  # -- threshold_li

    @profile
    def threshold_yen(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        image = self.image

        img = image.img

        thresh = filters.threshold_yen( img )

        data = img > thresh

        image = Image(data)
        image.algorithm = f"threshold_yen( {thresh:0.0f} )"

        return image
    pass  # -- threshold_yen

    @profile
    def threshold_otsu(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https: // docs.opencv.org / 3.4 / d7 / d4d / tutorial_py_thresholding.html

        image = self.image

        img = image.img
        img = img.astype(np.uint8)

        # Gaussian filtering
        #blur = cv.GaussianBlur(img, (5, 5), 0)
        # Otsu's thresholding
        threshold, data = cv.threshold( img, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"otsu threshold={threshold}"

        return image
    pass  # -- threshold_otsu_opencv

    @profile
    def threshold_multi_otsu(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_multiotsu

        image = self.image

        img = image.img
        img = img.astype(np.uint8)

        h = len(img)
        w = len(img[0])

        gray = img

        if isinstance(img[0][0], list) and len(img[0][0]) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pass

        norm = np.empty([h, w], np.uint8)

        cv.normalize(gray, norm, 0, 255, cv.NORM_MINMAX)

        data = None

        classes = 5

        # multi otsu
        from skimage.color import label2rgb
        thresholds = filters.threshold_multiotsu(norm, classes=classes)

        colorize = False
        if colorize:
            regions = np.digitize(norm, bins=thresholds)

            regions_colorized = label2rgb(regions)

            from skimage import img_as_ubyte

            regions_colorized = img_as_ubyte(regions_colorized)

            regions = regions.astype(np.uint8)

            data = np.empty([h, w], dtype=np.uint8)

            cv.normalize(regions, data, 0, 255, cv.NORM_MINMAX)
        else :
            data = gray

            thresholds = np.sort( thresholds )
            prev_thresh = 0

            for thresh in thresholds :
                data = np.where( (prev_thresh < data) & (data <= thresh), thresh, data)
                prev_thresh = thresh
            pass

            data = np.where( prev_thresh < data , 255, data)
        pass

        image = Image(data)
        image.algorithm = f"multi_otsu(classes={classes})"

        return image
    pass  # -- threshold_multiotsu

    @profile
    def threshold_adaptive_gaussian(self, bsize=5, c=0):
        # TODO     지역 가우시안 적응 임계치 처리
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

        image = self.image

        if bsize is None:
            bsize = self.default_bsize(image)
        pass

        if c is None:
            c = 0
        pass

        bsize = 2 * int(bsize / 2) + 1

        img = image.img
        img = img.astype(np.uint8)

        data = cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, bsize, c)

        image = Image( data )
        image.threshold = f"bsize = {bsize}"
        image.algorithm = f"adaptive gaussian(bsize={bsize}, c={c})"
        image.reverse_required = True

        return image
    pass  # -- threshold_adaptive_gaussian

    def default_bsize(self, image):
        diagonal = image.diagonal()

        bsize = diagonal // 150
        if bsize < 3:
            bsize = 3
        pass

        return bsize
    pass # -- default_bsize

    @profile
    def threshold_adaptive_mean(self, bsize=5, c=0):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # TODO     지역 평균 적응 임계치 처리
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

        image = self.image

        if bsize is None:
            bsize = self.default_bsize( image )
        pass

        if c is None:
            c = 0
        pass

        bsize = 2 * int(bsize / 2) + 1

        img = image.img
        img = img.astype(np.uint8)

        data = cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, bsize, c)

        image = Image(data)
        image.algorithm = f"adaptive mean(bsize={bsize}, c={c})"

        return image

    pass  # -- 지역 평균 적응 임계치 처리


pass