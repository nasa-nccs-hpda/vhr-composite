import datetime
import logging
import tqdm

import numpy as np


def get_post_str():
    sdtdate = datetime.datetime.now()
    year = sdtdate.year
    hm = sdtdate.strftime('%H%M')
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    post_str = '{}{:03}{}'.format(year, jdate, hm)
    return post_str


def convert_to_bit_rep(arr, nodata):
    converted_bit_array = np.zeros_like(arr, dtype=np.uint32)
    for i, value in np.ndenumerate(arr):
        converted_bit_array[i] = 1 << value
    converted_bit_array = np.where(arr == nodata, nodata, converted_bit_array)
    return converted_bit_array


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
