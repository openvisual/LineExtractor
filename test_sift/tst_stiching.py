# coding: utf-8

import cv2 as cv

import argparse
import sys

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
parser.add_argument('--mode',
                    type=int, choices=modes, default=cv.Stitcher_PANORAMA,
                    help='Determines configuration of stitcher. The default is `PANORAMA` (%d), '
                         'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
                         'for stitching materials under affine transformation, such as scans.' % modes)
parser.add_argument('--output', default='result.jpg',
                    help='Resulting image. The default is `result.jpg`.')

#parser.add_argument('img', nargs='+', help='input images')

def main():
    args = parser.parse_args()

    # read input images
    img = []
    img.append("./data_yegan/set_01/_1018843.JPG")
    img.append("./data_yegan/set_01/_1018844.JPG")

    imgs = []
    for img_name in img:
        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            print("can't read image " + img_name)
            sys.exit(-1)
        pass
        imgs.append(img)
    pass

    stitcher = cv.Stitcher.create(args.mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    pass

    cv.imwrite(args.output, pano)
    print("stitching completed successfully. %s saved!" % args.output)

    print('Done')
pass

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
pass