# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class array(__pybind11_builtins.pybind11_object):
    """ This object represents a 1D array of floating point numbers. Moreover, it binds directly to the C++ type std::vector<double>. """
    def append(self, x): # real signature unknown; restored from __doc__
        """
        append(self: dlib.array, x: float) -> None
        
        Add an item to the end of the list
        """
        pass

    def clear(self): # real signature unknown; restored from __doc__
        """ clear(self: dlib.array) -> None """
        pass

    def count(self, x): # real signature unknown; restored from __doc__
        """
        count(self: dlib.array, x: float) -> int
        
        Return the number of times ``x`` appears in the list
        """
        return 0

    def extend(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        extend(*args, **kwargs)
        Overloaded function.
        
        1. extend(self: dlib.array, L: dlib.array) -> None
        
        Extend the list by appending all the items in the given list
        
        2. extend(self: dlib.array, arg0: list) -> None
        """
        pass

    def insert(self, i, x): # real signature unknown; restored from __doc__
        """
        insert(self: dlib.array, i: int, x: float) -> None
        
        Insert an item at a given position.
        """
        pass

    def pop(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        pop(*args, **kwargs)
        Overloaded function.
        
        1. pop(self: dlib.array) -> float
        
        Remove and return the last item
        
        2. pop(self: dlib.array, i: int) -> float
        
        Remove and return the item at index ``i``
        """
        pass

    def remove(self, x): # real signature unknown; restored from __doc__
        """
        remove(self: dlib.array, x: float) -> None
        
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
        pass

    def resize(self, arg0): # real signature unknown; restored from __doc__
        """ resize(self: dlib.array, arg0: int) -> None """
        pass

    def __bool__(self): # real signature unknown; restored from __doc__
        """
        __bool__(self: dlib.array) -> bool
        
        Check whether the list is nonempty
        """
        return False

    def __contains__(self, x): # real signature unknown; restored from __doc__
        """
        __contains__(self: dlib.array, x: float) -> bool
        
        Return true the container contains ``x``
        """
        return False

    def __delitem__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __delitem__(*args, **kwargs)
        Overloaded function.
        
        1. __delitem__(self: dlib.array, arg0: int) -> None
        
        Delete the list elements at index ``i``
        
        2. __delitem__(self: dlib.array, arg0: slice) -> None
        
        Delete list elements using a slice object
        """
        pass

    def __eq__(self, arg0): # real signature unknown; restored from __doc__
        """ __eq__(self: dlib.array, arg0: dlib.array) -> bool """
        return False

    def __getitem__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __getitem__(*args, **kwargs)
        Overloaded function.
        
        1. __getitem__(self: dlib.array, s: slice) -> dlib.array
        
        Retrieve list elements using a slice object
        
        2. __getitem__(self: dlib.array, arg0: int) -> float
        """
        pass

    def __getstate__(self): # real signature unknown; restored from __doc__
        """ __getstate__(self: dlib.array) -> tuple """
        return ()

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.array) -> None
        
        2. __init__(self: dlib.array, arg0: dlib.array) -> None
        
        Copy constructor
        
        3. __init__(self: dlib.array, arg0: iterable) -> None
        
        4. __init__(self: dlib.array, arg0: object) -> None
        """
        pass

    def __iter__(self): # real signature unknown; restored from __doc__
        """ __iter__(self: dlib.array) -> iterator """
        pass

    def __len__(self): # real signature unknown; restored from __doc__
        """ __len__(self: dlib.array) -> int """
        return 0

    def __ne__(self, arg0): # real signature unknown; restored from __doc__
        """ __ne__(self: dlib.array, arg0: dlib.array) -> bool """
        return False

    def __repr__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __repr__(*args, **kwargs)
        Overloaded function.
        
        1. __repr__(self: dlib.array) -> str
        
        Return the canonical string representation of this list.
        
        2. __repr__(self: dlib.array) -> str
        """
        pass

    def __setitem__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __setitem__(*args, **kwargs)
        Overloaded function.
        
        1. __setitem__(self: dlib.array, arg0: int, arg1: float) -> None
        
        2. __setitem__(self: dlib.array, arg0: slice, arg1: dlib.array) -> None
        
        Assign list elements using a slice object
        """
        pass

    def __setstate__(self, arg0): # real signature unknown; restored from __doc__
        """ __setstate__(self: dlib.array, arg0: tuple) -> None """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.array) -> str """
        return ""

    __pybind11_module_local_v1__ = None # (!) real value is '<capsule object NULL at 0x10bafd450>'


