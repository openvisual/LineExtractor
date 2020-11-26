# encoding: utf-8
# module skimage.filters._multiotsu
# from C:\ProgramData\Anaconda3\envs\my_env\lib\site-packages\skimage\filters\_multiotsu.cp38-win_amd64.pyd
# by generator 1.147
# no doc

# imports
import builtins as __builtins__  # <module 'builtins' (built-in)>
import numpy as np  # C:\ProgramData\Anaconda3\envs\my_env\lib\site-packages\numpy\__init__.py


# functions

def _get_multiotsu_thresh_indices(*args, **kwargs):  # real signature unknown
    """
    Finds the indices of Otsu thresholds according to the values
        occurence probabilities.

        This implementation, as opposed to `_get_multiotsu_thresh_indices_lut`,
        does not use LUT. It is therefore slower.

        Parameters
        ----------
        prob : array
            Value occurence probabilities.
        thresh_count : int
            The desired number of threshold.

        Returns
        -------
        py_thresh_indices : array
            The indices of the desired thresholds.
    """
    pass


def _get_multiotsu_thresh_indices_lut(*args, **kwargs):  # real signature unknown
    """
    Finds the indices of Otsu thresholds according to the values
        occurence probabilities.

        This implementation uses a LUT to reduce the number of floating
        point operations (see [1]_). The use of the LUT reduces the
        computation time at the price of more memory consumption.

        Parameters
        ----------
        prob : array
            Value occurence probabilities.
        thresh_count : int
            The desired number of thresholds (classes-1).


        Returns
        -------
        py_thresh_indices : ndarray
            The indices of the desired thresholds.

        References
        ----------
        .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
               multilevel thresholding", Journal of Information Science and
               Engineering 17 (5): 713-727, 2001. Available at:
               <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
               :DOI:`10.6688/JISE.2001.17.5.1`
    """
    pass


def __pyx_unpickle_Enum(*args, **kwargs):  # real signature unknown
    pass


# no classes
# variables with complex values

__loader__ = None  # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x000002006AEE1130>'

__spec__ = None  # (!) real value is "ModuleSpec(name='skimage.filters._multiotsu', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x000002006AEE1130>, origin='C:\\\\ProgramData\\\\Anaconda3\\\\envs\\\\my_env\\\\lib\\\\site-packages\\\\skimage\\\\filters\\\\_multiotsu.cp38-win_amd64.pyd')"

__test__ = {}

