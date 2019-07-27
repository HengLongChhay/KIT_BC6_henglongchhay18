# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class image_gradients(__pybind11_builtins.pybind11_object):
    """
    This class is a tool for computing first and second derivatives of an 
    image.  It does this by fitting a quadratic surface around each pixel and 
    then computing the gradients of that quadratic surface.  For the details 
    see the paper: 
        Quadratic models for curved line detection in SAR CCD by Davis E. King 
        and Rhonda D. Phillips 
     
    This technique gives very accurate gradient estimates and is also very fast 
    since the entire gradient estimation procedure, for each type of gradient, 
    is accomplished by cross-correlating the image with a single separable 
    filter.  This means you can compute gradients at very large scales (e.g. by 
    fitting the quadratic to a large window, like a 99x99 window) and it still 
    runs very quickly.
    """
    def get_scale(self): # real signature unknown; restored from __doc__
        """
        get_scale(self: dlib.image_gradients) -> int
        
        When we estimate a gradient we do so by fitting a quadratic filter to a window of size 
        get_scale()*2+1 centered on each pixel.  Therefore, the scale parameter controls the size 
        of gradients we will find.  For example, a very large scale will cause the gradient_xx() 
        to be insensitive to high frequency noise in the image while smaller scales would be more 
        sensitive to such fluctuations in the image.
        """
        return 0

    def get_xx_filter(self): # real signature unknown; restored from __doc__
        """
        get_xx_filter(self: dlib.image_gradients) -> numpy.ndarray[(rows,cols),float32]
        
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
        pass

    def get_xy_filter(self): # real signature unknown; restored from __doc__
        """
        get_xy_filter(self: dlib.image_gradients) -> numpy.ndarray[(rows,cols),float32]
        
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
        pass

    def get_x_filter(self): # real signature unknown; restored from __doc__
        """
        get_x_filter(self: dlib.image_gradients) -> numpy.ndarray[(rows,cols),float32]
        
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
        pass

    def get_yy_filter(self): # real signature unknown; restored from __doc__
        """
        get_yy_filter(self: dlib.image_gradients) -> numpy.ndarray[(rows,cols),float32]
        
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
        pass

    def get_y_filter(self): # real signature unknown; restored from __doc__
        """
        get_y_filter(self: dlib.image_gradients) -> numpy.ndarray[(rows,cols),float32]
        
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
        pass

    def gradient_x(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        gradient_x(*args, **kwargs)
        Overloaded function.
        
        1. gradient_x(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),uint8]) -> tuple
        
        2. gradient_x(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),float32]) -> tuple
        
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
        pass

    def gradient_xx(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        gradient_xx(*args, **kwargs)
        Overloaded function.
        
        1. gradient_xx(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),uint8]) -> tuple
        
        2. gradient_xx(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),float32]) -> tuple
        
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
        pass

    def gradient_xy(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        gradient_xy(*args, **kwargs)
        Overloaded function.
        
        1. gradient_xy(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),uint8]) -> tuple
        
        2. gradient_xy(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),float32]) -> tuple
        
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
        pass

    def gradient_y(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        gradient_y(*args, **kwargs)
        Overloaded function.
        
        1. gradient_y(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),uint8]) -> tuple
        
        2. gradient_y(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),float32]) -> tuple
        
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
        pass

    def gradient_yy(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        gradient_yy(*args, **kwargs)
        Overloaded function.
        
        1. gradient_yy(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),uint8]) -> tuple
        
        2. gradient_yy(self: dlib.image_gradients, img: numpy.ndarray[(rows,cols),float32]) -> tuple
        
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.image_gradients, scale: int) -> None
        
        Creates this class with the provided scale. i.e. get_scale()==scale. 
        scale must be >= 1.
        
        2. __init__(self: dlib.image_gradients) -> None
        
        Creates this class with a scale of 1. i.e. get_scale()==1
        """
        pass


