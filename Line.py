# encoding: utf-8

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

from functools import cmp_to_key

from Point import *
from math import pi, cos, sin
import numpy as np

class Line:

    ID = 0

    def __init__(self, a=None, b=None, line=None, fileBase="", id = None ):
        if id is None :
            id = Line.ID
            Line.ID += 1
        pass

        self.id = id

        if line is not None :
            a = Point(line[0], line[1])
            b = Point(line[2], line[3])
        pass

        self.a = a if a.x < b.x else b
        self.b = b if b.x >= a.x else a

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
        #rad = rad % (2*math.pi)

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

    def merge(self, line_b, error_deg=1, snap_dist=5):
        merge_line = None
        debug = 0

        if not self.is_mergeable(line_b , error_deg=error_deg, snap_dist=snap_dist) :
            return merge_line
        pass

        la = self
        lb = line_b
        points = [ la.a, la.b, lb.a, lb.b]

        la_len = la.length()
        lb_len = lb.length()
        len_sum = la_len + lb_len

        xg = (la_len * (la.a.x + la.b.x) + lb_len * (lb.a.x + lb.b.x)) / (2 * len_sum)
        yg = (la_len * (la.a.y + la.b.y) + lb_len * (lb.a.y + lb.b.y)) / (2 * len_sum)

        theta_a = la.slope_radian()
        theta_b = lb.slope_radian()

        d_theta = abs( theta_a - theta_b ) % (2*math.pi)

        theta = theta_a

        if d_theta <= pi/2 :
            theta = (la_len*theta_a + lb_len*theta_b)/len_sum
        else :
            theta = (la_len*theta_a + lb_len*(theta_b - pi))/len_sum
        pass

        points_merge = [ None ]*len( points )
        if True :
            rotate_vec = np.array([cos(theta), sin(theta)])
            un_rot_matrix = np.array([[cos(- theta), sin(- theta)], [-sin(- theta), cos(- theta)]])

            for i, p in enumerate( points ) :
                p = np.array([p.x - xg, p.y - yg])
                x = np.dot( rotate_vec, p)
                y = 0

                q = np.matmul(un_rot_matrix, [x, y])

                a = Point(int(xg + q[0]), int(yg + q[1]))

                points_merge[i] = a
            pass
        pass

        debug and log.info( f"points org = { ', '.join([str(p) for p in points]) }")

        points_merge = sorted(points_merge, key=cmp_to_key(Point.compare_point_x))

        debug and log.info( f"points sort = { ', '.join([str(p) for p in points_merge]) }")

        merge_line = Line( a = points_merge[0], b = points_merge[-1], fileBase=la.fileBase )

        debug and log.info( f"merge line = {merge_line}")

        return merge_line
    pass # -- merge

    def merge_old(self, line, error_deg=1, snap_dist=5 ):
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
        cmp = a.distum() - b.distum()

        if cmp == 0 :
            cmp = a.slope_radian() - b.slope_radian()
        pass

        return cmp
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

    def get_most_mergable_line_from_lines(self, lineList, error_deg=1, snap_dist=5 ):
        line_found = None
        similarity_min = 100_000

        for line in lineList :
            if self.is_mergeable( line, error_deg=error_deg, snap_dist=snap_dist) :
                similarity = self.get_similarity(line)
                if similarity < similarity_min:
                    line_found = line
                    similarity_min = similarity
                pass
            pass
        pass

        return line_found, similarity_min

    pass  # --get_most_similar_line_from_list

    def get_most_mergable_line_from_linegrps(self, lineGrpList , error_deg=1, snap_dist=5 ):
        lineGrp_found = None
        line_found = None
        similarity_min = 100_000

        for lineGrp in lineGrpList:
            line, similarity = self.get_most_mergable_line_from_lines( lineGrp, error_deg=error_deg, snap_dist=snap_dist )
            if similarity < similarity_min:
                lineGrp_found = lineGrp
                line_found = line
                similarity_min = similarity
            pass
        pass

        return line_found, similarity_min, lineGrp_found

    pass  # --get_most_similar_line_from_linegrp_list

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
