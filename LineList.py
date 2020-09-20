# encoding: utf-8

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import math
from math import sin, cos

from functools import cmp_to_key
import numpy as np

from Line import *
from Common import profile

class LineList( list ) :
    def __init__(self, lines=[], algorithm="", w = 0 , h = 0, fileBase="", mode=""):
        self.extend( lines )
        self.mode = mode
        self.algorithm = algorithm

        self.w = w
        self.h = h

        self.diagonal = math.sqrt(w * w + h * h)

        self.fileBase = fileBase

        self.lineListIdentified = None
    pass # -- __init__

    def line_identify(self, lineList_b, min_length = 1, similarity_min = 0 ):
        debug = False

        fileBase = self.fileBase

        w = self.w
        h = self.h
        algorithm = self.algorithm

        lineList_b = lineList_b.copy()

        line_matches = []

        for line in self  :
            if line.length() > min_length :
                line_matched = line.get_match_line(lineList_b)
                if line_matched is not None :
                    if line_matched.length() > min_length :
                        line.line_matched = line_matched

                        line_matches.append( line )
                    pass

                    lineList_b.remove( line_matched )
                pass
            pass
        pass

        line_matches = sorted(line_matches, key=cmp_to_key(Line.compare_line_similarity))

        line_matches = [s for s in line_matches if s.similarity > similarity_min ]

        line_matches = line_matches[ :: -1 ]

        if debug :
            for i, line in enumerate( line_matches ) :
                log.info( f"[{i:03d}] similarity={line.similarity:0.4f} ")
            pass
        pass

        lineList = LineList( lines = line_matches, algorithm=algorithm, w=w, h=h, fileBase=fileBase)

        return lineList
    pass # -- identify

    def save_as_json(self, json_file_name ):
        debug = False
        import json
        #data = {'name': 'Scott', 'website': 'stackabuse.com', 'from': 'Nebraska'}
        data = {}

        lines = self

        for i, lineA in enumerate( lines ) :
            line_data = {}

            line = lineA
            fileBase = line.fileBase
            line_data[ fileBase ] = {"point1": [int(line.a.x), int(line.a.y)], "point2": [int(line.b.x), int(line.b.y)]}

            debug and log.info( f"id={line.id} , fileBase={fileBase}" )

            line = lineA.line_matched
            fileBase = line.fileBase

            line_data[ fileBase] = {"point1": [int(line.a.x), int(line.a.y)], "point2": [int(line.b.x), int(line.b.y)]}
            debug and log.info(f"id={line.id} , fileBase={fileBase}")

            data[ f"line{i +1}" ] = line_data
        pass

        with open( json_file_name, 'w') as f:
            json.dump(data, f, indent=4 )
        pass

    pass # -- save_as_json

    @profile
    def merge_lines(self, error_deg=1, snap_dist=5):
        debug = False

        lines = self

        lines = list(filter(lambda x: x.length() != 0 , lines))

        lineGroups = []

        for line in lines :
            line_found, _, lineGrp_found = line.get_most_mergeable_line_from_linegrps( lineGroups, error_deg=error_deg, snap_dist=snap_dist )
            if line_found is not None :
                lineGrp_found.append( line )
            else :
                lineGroups.append( [ line ] )
            pass
        pass

        merge_lines = []
        for lineGrp in lineGroups :
            merge_lines.append(LineList.merge_into_single_line(lineGrp))
            #merge_lines.extend(LineList.merge_into_single_lines(lineGrp))
        pass

        merge_lines = sorted(merge_lines, key=cmp_to_key(Line.compare_line_length))
        merge_lines = merge_lines[:: -1]

        lineList = LineList(lines=merge_lines, algorithm=self.algorithm, w=self.w, h=self.h, fileBase=self.fileBase, mode=self.mode)
        lineList.algorithm = f"line merge(error_deg={error_deg}, snap={snap_dist})"

        return lineList

    pass  # -- merge_lines

    @staticmethod
    def merge_into_single_line(lines, error_deg=1, snap_dist=5):
        debug = False
        lines = lines.copy()

        lines = sorted(lines, key=cmp_to_key(Line.compare_line_length))
        lines = lines[ :: -1 ]

        line = lines.pop(0)

        while len( lines ) > 0 :
            most_near_line, _ = line.get_most_mergeable_line_from_lines( lines )

            lines.remove( most_near_line )

            line = line.merge( most_near_line )
        pass

        return line

    pass  # -- merge_into_single_line

    @staticmethod
    def merge_into_single_line_failed( lines ):
        len_sum = 0

        lines = lines.copy()

        length_sum = 0
        slope_rad = 0

        if True :
            slope_rad_length_sum = 0

            slope_rad_ref = lines[0].slope_radian()

            for line in lines:
                slope_rad_length_sum += ( slope_rad_ref - line.slope_radian() )
                length_sum += line.length()
            pass

            slope_rad = slope_rad_ref + ( slope_rad_length_sum / len( lines ) )
        pass

        xg = 0
        yg = 0

        for line in lines :
            length = line.length()

            xg += length * ( line.a.x + line.b.x )
            yg += length * ( line.a.y + line.b.y )
        pass

        xg = xg / length_sum / len(lines)
        yg = yg / length_sum / len(lines)

        theta = slope_rad

        lines_rotated = [None] * len(lines)

        if True :
            rotate_project = np.array([cos(theta), sin(theta)] )
            r = rotate_project

            for i in range( len( lines_rotated ) ) :
                line = lines[ i ]

                p = line.a
                p = np.array([p.x - xg, p.y - yg])
                x = np.dot(r, p)
                y = 0
                a = Point(x, y)

                p = line.b
                p = np.array([p.x - xg, p.y - yg])
                x = np.dot(r, p)
                y = 0
                b = Point(x, y)

                lines_rotated[i] = Line( a=a, b=b, id=line.id )
            pass
        pass

        min_a = lines_rotated[0].a
        max_b = lines_rotated[0].b

        for line in lines_rotated :
            if line.a.x < min_a.x :
                min_a = line.a
            pass

            if line.b.x > max_b.x :
                max_b = line.b
            pass
        pass

        rotate_matrix = np.array([[cos( - theta), sin( - theta)], [-sin( - theta), cos( - theta)]])
        r = rotate_matrix

        p = np.matmul(r, [min_a.x, min_a.y])
        a = Point( int( xg + p[0] ), int( yg + p[1] ) )

        p = np.matmul(r, [max_b.x, max_b.y])
        b = Point( int( xg + p[0] ), int( yg + p[1] ) )

        merge_line = Line( a = a, b = b )

        return merge_line
    pass # -- merge_into_single_line

pass # -- LineList
