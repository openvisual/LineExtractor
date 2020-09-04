# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

''' profile functions '''

# -- usage
# @profile
# def your_function(...):
#       ....
#
# your_function( ... )
# print_prof_data()

import time
from functools import wraps

PROF_DATA = {}
PROF_LAST = None

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        pass

        global PROF_LAST
        PROF_LAST = fn.__name__

        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling
pass # -- profile(fn)

def print_prof_name(fn_name):
    data = PROF_DATA[ fn_name ]

    max_time = max(data[1])
    avg_time = sum(data[1]) / len(data[1])
    msg = f"*** The function[{fn_name}] was called {data[0]} times. Exe. time max: {max_time:.3f}, average: {avg_time:.3f}"

    log.info( msg )
pass # -- print_prof_name()

def print_prof_last( ) :
    PROF_LAST and print_prof_name( PROF_LAST )
pass

def print_prof_data():
    for fn_name in PROF_DATA :
        print_prof_name( fn_name )
    pass
pass # -- print_prof_data()

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
pass # -- clear_prof_data

# end