# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class rectangle(__pybind11_builtins.pybind11_object):
    """ This object represents a rectangular area of an image. """
    def area(self): # real signature unknown; restored from __doc__
        """ area(self: dlib.rectangle) -> int """
        return 0

    def bl_corner(self): # real signature unknown; restored from __doc__
        """
        bl_corner(self: dlib.rectangle) -> dlib.point
        
        Returns the bottom left corner of the rectangle.
        """
        pass

    def bottom(self): # real signature unknown; restored from __doc__
        """ bottom(self: dlib.rectangle) -> int """
        return 0

    def br_corner(self): # real signature unknown; restored from __doc__
        """
        br_corner(self: dlib.rectangle) -> dlib.point
        
        Returns the bottom right corner of the rectangle.
        """
        pass

    def center(self): # real signature unknown; restored from __doc__
        """ center(self: dlib.rectangle) -> dlib.point """
        pass

    def contains(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        contains(*args, **kwargs)
        Overloaded function.
        
        1. contains(self: dlib.rectangle, point: dlib.point) -> bool
        
        2. contains(self: dlib.rectangle, point: dlib.dpoint) -> bool
        
        3. contains(self: dlib.rectangle, x: int, y: int) -> bool
        
        4. contains(self: dlib.rectangle, rectangle: dlib.rectangle) -> bool
        """
        pass

    def dcenter(self): # real signature unknown; restored from __doc__
        """ dcenter(self: dlib.rectangle) -> dlib.point """
        pass

    def height(self): # real signature unknown; restored from __doc__
        """ height(self: dlib.rectangle) -> int """
        return 0

    def intersect(self, rectangle): # real signature unknown; restored from __doc__
        """ intersect(self: dlib.rectangle, rectangle: dlib.rectangle) -> dlib.rectangle """
        pass

    def is_empty(self): # real signature unknown; restored from __doc__
        """ is_empty(self: dlib.rectangle) -> bool """
        return False

    def left(self): # real signature unknown; restored from __doc__
        """ left(self: dlib.rectangle) -> int """
        return 0

    def right(self): # real signature unknown; restored from __doc__
        """ right(self: dlib.rectangle) -> int """
        return 0

    def tl_corner(self): # real signature unknown; restored from __doc__
        """
        tl_corner(self: dlib.rectangle) -> dlib.point
        
        Returns the top left corner of the rectangle.
        """
        pass

    def top(self): # real signature unknown; restored from __doc__
        """ top(self: dlib.rectangle) -> int """
        return 0

    def tr_corner(self): # real signature unknown; restored from __doc__
        """
        tr_corner(self: dlib.rectangle) -> dlib.point
        
        Returns the top right corner of the rectangle.
        """
        pass

    def width(self): # real signature unknown; restored from __doc__
        """ width(self: dlib.rectangle) -> int """
        return 0

    def __add__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __add__(*args, **kwargs)
        Overloaded function.
        
        1. __add__(self: dlib.rectangle, arg0: dlib.point) -> dlib.rectangle
        
        2. __add__(self: dlib.rectangle, arg0: dlib.rectangle) -> dlib.rectangle
        """
        pass

    def __eq__(self, arg0): # real signature unknown; restored from __doc__
        """ __eq__(self: dlib.rectangle, arg0: dlib.rectangle) -> bool """
        return False

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.rectangle) -> tuple """
        return ()

    def __iadd__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __iadd__(*args, **kwargs)
        Overloaded function.
        
        1. __iadd__(self: dlib.rectangle, arg0: dlib.point) -> dlib.rectangle
        
        2. __iadd__(self: dlib.rectangle, arg0: dlib.rectangle) -> dlib.rectangle
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.rectangle, left: int, top: int, right: int, bottom: int) -> None
        
        2. __init__(self: dlib.rectangle, rect: dlib::drectangle) -> None
        
        3. __init__(self: dlib.rectangle, rect: dlib.rectangle) -> None
        
        4. __init__(self: dlib.rectangle) -> None
        """
        pass

    def __ne__(self, arg0): # real signature unknown; restored from __doc__
        """ __ne__(self: dlib.rectangle, arg0: dlib.rectangle) -> bool """
        return False

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.rectangle) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.rectangle, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.rectangle) -> str """
        return ""


