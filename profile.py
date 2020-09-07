# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

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

class Profile :
    def __init__(self, fn_name):
        self.fn_name = fn_name
        self.call_times = 0
        self._max_time = 0
        self.exec_times = []
    pass

    def called_by(self, elapsed_time):
        self.call_times += 1
        self.exec_times.append( elapsed_time )
        self._max_time = max( [ self._max_time, elapsed_time ] )
    pass

    def max_time(self):
        return self._max_time
    pass

    def avg_time(self):
        exec_times = self.exec_times
        avg_time = sum(exec_times[1]) / len(exec_times[1])
        return avg_time
    pass
pass # -- Profile

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        fn_name = fn.__name__

        if fn_name not in PROF_DATA:
            PROF_DATA[ fn_name ] = Profile( fn_name )
        pass

        prof = PROF_DATA[ fn_name ]
        prof.called_by( elapsed_time )

        return ret

    return with_profiling
pass # -- profile(fn)

def print_profile_name(fn_name, max_len ):
    prof = PROF_DATA[ fn_name ]

    print_profile_data(fn_name, prof, max_len)
pass

def print_profile_data(fn_name, prof : Profile, max_len) :
    max_time = prof.max_time()
    avg_time = prof.avg_time()
    call_times = prof.call_times

    fn_name_show = fn_name.rjust( max_len, ' ' )
    msg = f"*** The function[ { fn_name_show } ] Average: {avg_time:.3f} sec(s), Max: {max_time:.3f} sec(s), Call : {call_times} times. "

    log.info( msg )
pass # -- print_prof_name()

def print_profile_last( ) :
    PROF_LAST and print_profile_name( PROF_LAST )
pass

def print_profile():
    max_len = max( len(fn_name) for fn_name in PROF_DATA )

    for fn_name, data in PROF_DATA :
        print_profile_data( fn_name, data, max_len )
    pass
pass # -- print_prof_data()

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
pass # -- clear_prof_data

# end