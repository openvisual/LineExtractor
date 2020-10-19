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

    @profile
    def merge_lines(self, error_deg=1, snap_dist=5):
        debug = False

        lines = self

        lines = list(filter(lambda x: x.length() != 0 , lines))

        i = 0
        while i < len( lines ) :
            line = lines[ i ]
            line_found, _ = line.get_most_mergeable_line_from_lines( lines, error_deg=error_deg, snap_dist=snap_dist )
            if line_found is not None and line != line_found :
                line_merge = line.merge( line_found )

                index = lines.index( line_found )
                if index < i :
                    lines[ index ] = line_merge

                    lines.pop( i )

                    i = index
                else :
                    lines[ i ] = line_merge

                    lines.pop( index )
                pass
            else :
                i += 1
            pass
        pass

        merge_lines = lines

        merge_lines = sorted(merge_lines, key=cmp_to_key(Line.compare_line_length))
        merge_lines = merge_lines[:: -1]

        lineList = LineList(lines=merge_lines, algorithm=self.algorithm, w=self.w, h=self.h, fileBase=self.fileBase, mode=self.mode)
        lineList.algorithm = f"line merge(error_deg={error_deg}, snap={snap_dist})"

        return lineList

    pass  # -- merge_lines

    @profile
    def merge_lines_old(self, error_deg=1, snap_dist=5):
        debug = False

        lines = self

        lines = list(filter(lambda x: x.length() != 0, lines))

        merge_lines = []

        idx = 0

        while idx < 5 and len(lines) != len(merge_lines):
            log.info(f"line merge[{idx}]")
            idx += 1

            lineGrps = []

            for line in lines:
                line_found, _, lineGrp_found = line.get_most_mergeable_line_from_linegrps(lineGrps, error_deg=error_deg,
                                                                                          snap_dist=snap_dist)
                if line_found is not None:
                    lineGrp_found.append(line)
                else:
                    lineGrps.append([line])
                pass
            pass

            if len(lines) > len(lineGrps):
                merge_lines = []
                for lineGrp in lineGrps:
                    merge_lines.append(LineList.merge_into_single_line(lineGrp))
                pass
            else:
                merge_lines = lines
            pass
        pass

        merge_lines = sorted(merge_lines, key=cmp_to_key(Line.compare_line_length))
        merge_lines = merge_lines[:: -1]

        lineList = LineList(lines=merge_lines, algorithm=self.algorithm, w=self.w, h=self.h, fileBase=self.fileBase,
                            mode=self.mode)
        lineList.algorithm = f"line merge(error_deg={error_deg}, snap={snap_dist})"

        return lineList

    pass  # -- merge_lines_old

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

    def save_as_json(self, json_file_name, width, height, mw = 0, mh = 0 ):
        debug = False

        import json

        #data = {'name': 'Scott', 'website': 'stackabuse.com', 'from': 'Nebraska'}
        data = {}

        width += 2*mw
        height += 2*mh

        def conv_coord(point, w, h):
            x = point.x - w / 2
            y = h / 2 - point.y

            if w % 2 == 0 :
                x = x + 0.5
            pass

            if h % 2 == 0 :
                y = y - 0.5
            pass

            return [x, y]
        pass

        lines = self

        for i, lineA in enumerate(lines):
            line_data = {}

            for line in [lineA, lineA.line_matched]:
                fileBase = line.fileBase
                line_data[fileBase] = {
                    "point1": conv_coord(line.a, width, height),
                    "point2": conv_coord(line.b, width, height)
                }

                debug and log.info(f"id={line.id} , fileBase={fileBase}")
            pass

            data[f"line{i + 1}"] = line_data
        pass

        with open(json_file_name, 'w') as f:
            json.dump(data, f, indent=4)
        pass

    pass  # -- save_as_json

pass # -- LineList
