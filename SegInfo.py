# -*- coding: utf-8 -*-

class Gap :
    # a gap between segments
    def __init__(self, coord):
        self.coord = coord
    pass

    def distance(self):
        coord = self.coord

        diff = coord[1] - coord[0]
        return abs( diff )
    pass
pass # -- Gap

class SegInfo :
    # segmentation info
    def __init__(self, coord):
        self.coord = coord
    pass

    def distance(self):
        coord = self.coord

        diff = coord[1] - coord[0]
        return diff
    pass
pass # -- Segment