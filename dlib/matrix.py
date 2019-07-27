# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class matrix(__pybind11_builtins.pybind11_object):
    """ This object represents a dense 2D matrix of floating point numbers.Moreover, it binds directly to the C++ type dlib::matrix<double>. """
    def nc(self): # real signature unknown; restored from __doc__
        """
        nc(self: dlib.matrix) -> int
        
        Return the number of columns in the matrix.
        """
        return 0

    def nr(self): # real signature unknown; restored from __doc__
        """
        nr(self: dlib.matrix) -> int
        
        Return the number of rows in the matrix.
        """
        return 0

    def set_size(self, rows, cols): # real signature unknown; restored from __doc__
        """
        set_size(self: dlib.matrix, rows: int, cols: int) -> None
        
        Set the size of the matrix to the given number of rows and columns.
        """
        pass

    def __getitem__(self, arg0): # real signature unknown; restored from __doc__
        """ __getitem__(self: dlib.matrix, arg0: int) -> dlib._row """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.matrix) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.matrix) -> None
        
        2. __init__(self: dlib.matrix, arg0: list) -> None
        
        3. __init__(self: dlib.matrix, arg0: object) -> None
        
        4. __init__(self: dlib.matrix, arg0: int, arg1: int) -> None
        """
        pass

    def __len__(self): # real signature unknown; restored from __doc__
        """ __len__(self: dlib.matrix) -> int """
        return 0

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.matrix) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.matrix, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.matrix) -> str """
        return ""

    shape = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



