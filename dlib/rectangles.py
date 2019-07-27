# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class rectangles(__pybind11_builtins.pybind11_object):
    """ An array of rectangle objects. """
    def append(self, x): # real signature unknown; restored from __doc__
        """
        append(self: dlib.rectangles, x: dlib.rectangle) -> None
        
        Add an item to the end of the list
        """
        pass

    def clear(self): # real signature unknown; restored from __doc__
        """ clear(self: dlib.rectangles) -> None """
        pass

    def count(self, x): # real signature unknown; restored from __doc__
        """
        count(self: dlib.rectangles, x: dlib.rectangle) -> int
        
        Return the number of times ``x`` appears in the list
        """
        return 0

    def extend(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        extend(*args, **kwargs)
        Overloaded function.
        
        1. extend(self: dlib.rectangles, L: dlib.rectangles) -> None
        
        Extend the list by appending all the items in the given list
        
        2. extend(self: dlib.rectangles, arg0: list) -> None
        """
        pass

    def insert(self, i, x): # real signature unknown; restored from __doc__
        """
        insert(self: dlib.rectangles, i: int, x: dlib.rectangle) -> None
        
        Insert an item at a given position.
        """
        pass

    def pop(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        pop(*args, **kwargs)
        Overloaded function.
        
        1. pop(self: dlib.rectangles) -> dlib.rectangle
        
        Remove and return the last item
        
        2. pop(self: dlib.rectangles, i: int) -> dlib.rectangle
        
        Remove and return the item at index ``i``
        """
        pass

    def remove(self, x): # real signature unknown; restored from __doc__
        """
        remove(self: dlib.rectangles, x: dlib.rectangle) -> None
        
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
        pass

    def resize(self, arg0): # real signature unknown; restored from __doc__
        """ resize(self: dlib.rectangles, arg0: int) -> None """
        pass

    def __bool__(self): # real signature unknown; restored from __doc__
        """
        __bool__(self: dlib.rectangles) -> bool
        
        Check whether the list is nonempty
        """
        return False

    def __contains__(self, x): # real signature unknown; restored from __doc__
        """
        __contains__(self: dlib.rectangles, x: dlib.rectangle) -> bool
        
        Return true the container contains ``x``
        """
        return False

    def __delitem__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __delitem__(*args, **kwargs)
        Overloaded function.
        
        1. __delitem__(self: dlib.rectangles, arg0: int) -> None
        
        Delete the list elements at index ``i``
        
        2. __delitem__(self: dlib.rectangles, arg0: slice) -> None
        
        Delete list elements using a slice object
        """
        pass

    def __eq__(self, arg0): # real signature unknown; restored from __doc__
        """ __eq__(self: dlib.rectangles, arg0: dlib.rectangles) -> bool """
        return False

    def __getitem__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __getitem__(*args, **kwargs)
        Overloaded function.
        
        1. __getitem__(self: dlib.rectangles, s: slice) -> dlib.rectangles
        
        Retrieve list elements using a slice object
        
        2. __getitem__(self: dlib.rectangles, arg0: int) -> dlib.rectangle
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.rectangles) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.rectangles) -> None
        
        2. __init__(self: dlib.rectangles, arg0: dlib.rectangles) -> None
        
        Copy constructor
        
        3. __init__(self: dlib.rectangles, arg0: iterable) -> None
        
        4. __init__(self: dlib.rectangles, initial_size: int) -> None
        """
        pass

    def __iter__(self): # real signature unknown; restored from __doc__
        """ __iter__(self: dlib.rectangles) -> iterator """
        pass

    def __len__(self): # real signature unknown; restored from __doc__
        """ __len__(self: dlib.rectangles) -> int """
        return 0

    def __ne__(self, arg0): # real signature unknown; restored from __doc__
        """ __ne__(self: dlib.rectangles, arg0: dlib.rectangles) -> bool """
        return False

    def __repr__(self): # real signature unknown; restored from __doc__
        """
        __repr__(self: dlib.rectangles) -> str
        
        Return the canonical string representation of this list.
        """
        return ""

    def __setitem__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __setitem__(*args, **kwargs)
        Overloaded function.
        
        1. __setitem__(self: dlib.rectangles, arg0: int, arg1: dlib.rectangle) -> None
        
        2. __setitem__(self: dlib.rectangles, arg0: slice, arg1: dlib.rectangles) -> None
        
        Assign list elements using a slice object
        """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.rectangles, arg0: tuple) -> None """
        pass


