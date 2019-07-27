# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class drectangle(__pybind11_builtins.pybind11_object):
    """ This object represents a rectangular area of an image with floating point coordinates. """
    def area(self): # real signature unknown; restored from __doc__
        """ area(self: dlib.drectangle) -> float """
        return 0.0

    def bl_corner(self): # real signature unknown; restored from __doc__
        """
        bl_corner(self: dlib.drectangle) -> dlib.dpoint
        
        Returns the bottom left corner of the rectangle.
        """
        pass

    def bottom(self): # real signature unknown; restored from __doc__
        """ bottom(self: dlib.drectangle) -> float """
        return 0.0

    def br_corner(self): # real signature unknown; restored from __doc__
        """
        br_corner(self: dlib.drectangle) -> dlib.dpoint
        
        Returns the bottom right corner of the rectangle.
        """
        pass

    def center(self): # real signature unknown; restored from __doc__
        """ center(self: dlib.drectangle) -> dlib.point """
        pass

    def contains(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        contains(*args, **kwargs)
        Overloaded function.
        
        1. contains(self: dlib.drectangle, point: dlib.point) -> bool
        
        2. contains(self: dlib.drectangle, point: dlib.dpoint) -> bool
        
        3. contains(self: dlib.drectangle, x: int, y: int) -> bool
        
        4. contains(self: dlib.drectangle, rectangle: dlib.drectangle) -> bool
        """
        pass

    def dcenter(self): # real signature unknown; restored from __doc__
        """ dcenter(self: dlib.drectangle) -> dlib.point """
        pass

    def height(self): # real signature unknown; restored from __doc__
        """ height(self: dlib.drectangle) -> float """
        return 0.0

    def intersect(self, rectangle): # real signature unknown; restored from __doc__
        """ intersect(self: dlib.drectangle, rectangle: dlib.drectangle) -> dlib.drectangle """
        pass

    def is_empty(self): # real signature unknown; restored from __doc__
        """ is_empty(self: dlib.drectangle) -> bool """
        return False

    def left(self): # real signature unknown; restored from __doc__
        """ left(self: dlib.drectangle) -> float """
        return 0.0

    def right(self): # real signature unknown; restored from __doc__
        """ right(self: dlib.drectangle) -> float """
        return 0.0

    def tl_corner(self): # real signature unknown; restored from __doc__
        """
        tl_corner(self: dlib.drectangle) -> dlib.dpoint
        
        Returns the top left corner of the rectangle.
        """
        pass

    def top(self): # real signature unknown; restored from __doc__
        """ top(self: dlib.drectangle) -> float """
        return 0.0

    def tr_corner(self): # real signature unknown; restored from __doc__
        """
        tr_corner(self: dlib.drectangle) -> dlib.dpoint
        
        Returns the top right corner of the rectangle.
        """
        pass

    def width(self): # real signature unknown; restored from __doc__
        """ width(self: dlib.drectangle) -> float """
        return 0.0

    def __eq__(self, arg0): # real signature unknown; restored from __doc__
        """ __eq__(self: dlib.drectangle, arg0: dlib.drectangle) -> bool """
        return False

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.drectangle) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.drectangle, left: float, top: float, right: float, bottom: float) -> None
        
        2. __init__(self: dlib.drectangle, rect: dlib.rectangle) -> None
        
        3. __init__(self: dlib.drectangle, rect: dlib.drectangle) -> None
        
        4. __init__(self: dlib.drectangle) -> None
        """
        pass

    def __ne__(self, arg0): # real signature unknown; restored from __doc__
        """ __ne__(self: dlib.drectangle, arg0: dlib.drectangle) -> bool """
        return False

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.drectangle) -> str """
        return ""

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.drectangle, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.drectangle) -> str """
        return ""


