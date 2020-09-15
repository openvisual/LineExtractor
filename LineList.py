# encoding: utf-8

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import math
from functools import cmp_to_key

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

    def line_identify(self, lineList_b):
        fileBase = self.fileBase

        w = self.w
        h = self.h
        algorithm = self.algorithm

        lines_identified = []

        for line in self  :
            line_identified = line.get_identified_line(lineList_b)
            if line_identified is not None :
                line.line_identified = line_identified

                lines_identified.append( line )
            pass
        pass

        lineList = LineList( lines = lines_identified, algorithm=algorithm, w=w, h=h, fileBase=fileBase)

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

            line = lineA.line_identified
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

        lineGroups = []

        for line in lines :
            line_found, _, lineGrp_found = line.get_most_mergable_line_from_linegrps( lineGroups, error_deg=error_deg, snap_dist=snap_dist )
            if line_found is not None :
                lineGrp_found.append( line )
            else :
                lineGroups.append( [ line ] )
            pass
        pass

        merge_lines = []
        for lineGrp in lineGroups :
            merge_lines.append( LineList.merge_into_single_line( lineGrp ) )
        pass

        merge_lines = sorted(merge_lines, key=cmp_to_key(Line.compare_line_length))
        merge_lines = merge_lines[:: -1]

        lineList = LineList(lines=merge_lines, algorithm=self.algorithm, w=self.w, h=self.h, fileBase=self.fileBase, mode=self.mode)
        lineList.algorithm = f"{self.algorithm} merge(error_deg={error_deg}, snap={snap_dist})"

        return lineList

    pass  # -- merge_lines

    @staticmethod
    def merge_into_single_line( lines ):
        debug = False

        slope_rad_length_sum = 0
        length_sum = 0

        for line in lines :
            slope_rad_length_sum += line.slope_radian()
            length_sum += line.length()
        pass

        slope_rad = slope_rad_length_sum/length_sum
        slope_rad = slope_rad % (2*math.pi)

        line = lines[0]
        min_a = line.a if line.a.x < line.b.x else line.b
        max_b = line.a if line.a.x > line.b.x else line.b

        for line in lines :
            points = [ line.a, line.b ]

            for p in points :
                if p.x < min_a.x :
                    min_a = p
                elif line.a.x > max_b.x :
                    max_b = p
                pass
            pass
        pass

        line = Line( a = min_a, b = max_b )

        return line
    pass

    @profile
    def merge_lines_old(self, error_deg=1, snap_dist=5):
        debug = False
        lines = self.copy()

        line_merged = True

        while line_merged:
            line_merged = False

            i = 0
            lines = sorted(lines, key=cmp_to_key(Line.compare_line_slope))

            while i < (len(lines) - 1):
                j = 0
                while i < (len(lines) - 1) and j < len(lines):
                    merge_line = None

                    if i is not j:
                        merge_line = lines[i].merge(lines[j], error_deg=error_deg, snap_dist=snap_dist)
                    pass

                    if merge_line is not None:
                        line_merged = True
                        lines[i] = merge_line
                        lines.pop(j)

                        debug and log.info(f"Line({i}, {j}) are merged.")
                    else:
                        j += 1
                    pass
                pass

                i += 1
            pass
        pass

        lines = sorted(lines, key=cmp_to_key(Line.compare_line_length))
        lines = lines[:: -1]

        lineList = LineList(lines=lines, algorithm=self.algorithm, w=self.w, h=self.h, fileBase=self.fileBase,
                            mode=self.mode)
        lineList.algorithm = f"{self.algorithm} merge(error_deg={error_deg}, snap={snap_dist})"

        return lineList

    pass  # -- merge_lines old

pass # -- LineList
