# encoding: utf-8

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from functools import cmp_to_key

from Point import *

class Line:

    ID = 0

    def __init__(self, a=None , b=None, line=None, fileBase="" ):
        self.id = Line.ID
        Line.ID += 1

        if line is not None :
            self.a = Point(line[0], line[1])
            self.b = Point(line[2], line[3])
        else :
            self.a = a
            self.b = b
        pass

        self.fileBase = fileBase
        self.line_identified = None
        self.similarity = None
    pass

    def __getitem__(self, i):
        if i == 0 :
            return self.a
        elif i == 1 :
            return self.b
        else :
            return None
        pass
    pass

    def __str__(self):
        return f"Line({self.a.x}, {self.a.y}, {self.b.x}, {self.b.y})"
    pass

    def dx(self):
        return self.b.x - self.a.x

    pass

    def dy(self):
        return self.b.y - self.a.y

    pass

    def length(self):
        return math.sqrt( self.distum() )
    pass

    def distum(self):
        return self.a.distum(self.b)
    pass

    def thickness(self):
        length = self.length()

        if length > 1000:
            thickness = 23 + length / 1000
        elif length > 100:
            thickness = 12 + length / 100
        else:
            thickness = 2 + length / 10
        pass

        thickness = int(thickness)

        return thickness
    pass # -- thickness

    def slope_radian(self):
        rad = math.atan2( self.dy() , self.dx() )
        rad = rad % (2*math.pi)

        return rad
    pass

    def is_same_slope(self, line, error_deg=1 ):
        sa = math.degrees( self.slope_radian() )
        sb = math.degrees( line.slope_radian() )

        diff_deg = abs( sa - sb )%360

        0 and log.info( f"sa = {sa:.2f}, sb = {sb:.2f}, diff = {diff_deg:.2f}")

        return diff_deg <= error_deg
    pass # -- is_same_slope

    def is_mergeable(self, line, error_deg, snap_dist ):
        if self.is_same_slope( line, error_deg = error_deg ) :
            from shapely.geometry import LineString

            line1 = LineString([(self.a.x, self.a.y), (self.b.x, self.b.y)])
            line2 = LineString([(line.a.x, line.a.y), (line.b.x, line.b.y)])

            dist = line1.distance(line2)

            0 and log.info( f"dist = {dist}" )

            return dist <= snap_dist
        else :
            return False
        pass
    pass # -- is_mergeable

    def merge(self, line, error_deg=1, snap_dist=5 ):
        merge_line = None
        debug = 0

        if self.is_mergeable( line , error_deg=error_deg, snap_dist=snap_dist) :
            points = [self.a, self.b, line.a, line.b]

            debug and log.info( f"points org = { ', '.join([str(p) for p in points]) }")

            points = sorted(points, key=cmp_to_key(Point.compare_point_x))

            debug and log.info( f"points sort = { ', '.join([str(p) for p in points]) }")

            merge_line = Line( a = points[0], b = points[-1], fileBase=self.fileBase )

            debug and log.info( f"merge line = {merge_line}")
        pass

        return merge_line
    pass # -- merge

    @staticmethod
    def compare_line_length(a, b):
        return a.distum() - b.distum()
    pass

    @staticmethod
    def compare_line_slope(a, b):
        return a.slope_radian() - b.slope_radian()
    pass

    def get_similarity(self, line_b):
        line_a = self

        two_pi = 2*math.pi

        diff_rad_ratio = ( abs(line_a.slope_radian() - line_b.slope_radian()) % two_pi )/two_pi

        a_length = line_a.length()
        b_length = line_b.length()

        max_length = max([a_length, b_length])

        diff_len_ratio = abs(a_length - b_length) / max_length

        dist_a = min( line_a.a.distance(line_b.a), line_a.a.distance(line_b.b))
        dist_b = min( line_a.b.distance(line_b.a), line_a.b.distance(line_b.b))

        dist = max([dist_a, dist_b])
        dist_ratio = dist / max_length

        similarity = (diff_rad_ratio + diff_len_ratio + dist_ratio)/3

        return similarity
    pass # -- get_similarity

    def get_identified_line(self, lineList ):

        line_found = None
        similarity_min = 100_000

        for line_b in lineList:
            similarity = self.get_similarity( line_b )
            if similarity < similarity_min :
                similarity_min = similarity
                line_found = line_b
                line_found.similarity = similarity_min
            pass
        pass

        return line_found

    pass  # -- get_identified_line


pass # -- Line
