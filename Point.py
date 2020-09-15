# encoding: utf-8

import math

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    pass

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return None
        pass

    pass

    def __str__(self):
        return f"Point( {self.x}, {self.y} )"

    pass

    def distum(self, p ):
        dx = self.x - p.x
        dy = self.y - p.y
        return dx*dx + dy*dy
    pass

    def distance(self, p):
        return math.sqrt(self.distum(p))
    pass

    @staticmethod
    def compare_point_x(a, b):
        return a.x - b.x
    pass

pass # -- Point
