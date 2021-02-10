import time
import numpy as np
from datetime import datetime
from itertools import product

STIME_2_TIMESTAMP = lambda t: int(time.mktime(datetime(year=t[0], month=t[1], day=t[2], hour=t[3], minute=t[4], second=0).timetuple()))
ETIME_2_TIMESTAMP = lambda t, span: int(time.mktime(datetime(year=t[0], month=t[1], day=t[2], hour=t[3], minute=t[4] + span - 1, second=59).timetuple()))
ITER_TIME = lambda year, month, day, hour, minute, span: np.array([[STIME_2_TIMESTAMP(t), ETIME_2_TIMESTAMP(t, span)] for t in product(year, month, day, hour, minute)], dtype=np.int64)


def init_Range(style = None):
    if style == 'fault':
        years = [2020]
        months = [8]
        days = range(25, 32)       
        hours = range(0, 24)
    elif style == 'normal':
        years = [2020]
        months = [8]
        days = range(1, 8)       
        hours = range(0, 24)
    elif style == 'test_normal':
        years = [2020]
        months = [8]
        days = [15]       
        hours = range(0, 24)
    elif style == 'test_fault':
        years = [2020]
        months = [9]
        days = range(1, 6)       
        hours = range(0, 24)
    elif style == 'push_to_infuxdb':
        years = [2020]
        months = [9]
        days = range(1, 30)      
        hours = range(0, 24)
    else:
        years = [2020]
        months = [9]
        days = [7]       
        hours = range(0, 24)
    # minutes=[_ for _ in np.arange(0, 60, 20, dtype=np.int)]
    minutes = [0, 20, 40]
    fit_range = ITER_TIME(years, months, days, hours, minutes, 20)
    return fit_range