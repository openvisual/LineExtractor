# -*- coding: utf-8 -*-

from Common import *
from Image import Image

class Threshold(Common) :

    def __init__(self, image ):
        self.image = image
    pass

    def threshold(self, algorithm):
        # TODO 이진화

        v = None

        if "_li" in algorithm :
            v = self.threshold_li()
        elif "otsu" in algorithm:
            v = self.threshold_otsu()
        elif "gaussian" in algorithm:
            w, h = self.image.dimension()

            bsize = w if w > h else h
            bsize = bsize / 6
            bsize = 13

            v = self.threshold_adaptive_gaussian(bsize=bsize, c=0)
        elif "mean" in algorithm:
            bsize = 5
            v = self.threshold_adaptive_mean(bsize=bsize, c=0)
        elif "global" in algorithm:
            v = self.threshold_global()
        elif "isodata" in algorithm:
            v = self.threshold_isodata()
        elif "balanced" in algorithm:
            v = self.threshold_balanced()
        pass

        return v

    pass  # -- threshold

    @profile
    def threshold_global(self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # TODO     전역 임계치 처리

        reverse_required = 0

        image = self.image

        img = image.img

        histogram = image.histogram()

        useMargin = False
        margin = 0
        if useMargin :
            margin = 50
            histogram[0: margin] = 0
        pass

        x = np.arange(0, len(histogram))

        avg = margin + ( sum( histogram * x )/np.sum( histogram ) )

        threshold = avg

        data = np.where( img >= threshold , 1, 0 )

        image = Image( data )
        image.threshold = threshold
        image.algorithm = f"global thresholding ({ int(threshold) })"
        image.reverse_required = reverse_required

        return image
    pass  # -- 전역 임계치 처리

    @profile
    def threshold_isodata(self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        reverse_required = 0

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
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_isodata

    @profile
    def threshold_balanced( self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        reverse_required = 0

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
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_balanced

    @profile
    def threshold_li(self ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        from skimage import filters

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
    def threshold_adaptive_mean(self, bsize=3, c=0):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # TODO     지역 평균 적응 임계치 처리

        reverse_required = 1

        image = self.image

        img = image.img

        w, h = image.dimension()

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

    @profile
    def threshold_otsu(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https: // docs.opencv.org / 3.4 / d7 / d4d / tutorial_py_thresholding.html

        reverse_required = 0

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
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_otsu_opencv

    @profile
    def threshold_adaptive_gaussian(self, bsize=5, c=0):
        # TODO     지역 가우시안 적응 임계치 처리

        algorithm = 0

        if algorithm == 0 :
            v = self.threshold_adaptive_gaussian_opencv(bsize=bsize, c=c)
        elif algorithm == 1:
            v = self.threshold_adaptive_gaussian_my(bsize=bsize, c=c)
        pass

        return v
    pass # -- threshold_adaptive_gaussian

    @profile
    def threshold_adaptive_gaussian_opencv(self, bsize=5, c=0):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

        reverse_required = 0
        bsize = 2 * int(bsize / 2) + 1

        image = self.image

        img = image.img
        img = img.astype(np.uint8)

        data = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c)

        image = Image( data )
        image.threshold = f"bsize = {bsize}"
        image.algorithm = f"adaptive gaussian, bsize={bsize}"
        image.reverse_required = reverse_required

        return image
    pass  # -- threshold_adaptive_gaussian_opencv

    @profile
    def threshold_adaptive_gaussian_my(self, bsize=3, c=0):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel

        reverse_required = 1

        bsize = 2 * int(bsize / 2) + 1

        image = self.image

        img = image.img

        w, h = image.dimension()

        data = np.empty((h, w), dtype='B')

        b = int(bsize / 2)

        if b < 1:
            b = 1
        pass

        # the threshold value T(x,y) is a weighted sum (cross-correlation with a Gaussian window)
        # of the blockSize×blockSize neighborhood of (x,y) minus C

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

pass