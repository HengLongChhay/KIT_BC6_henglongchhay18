# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class chip_details(__pybind11_builtins.pybind11_object):
    """
    WHAT THIS OBJECT REPRESENTS 
        This object describes where an image chip is to be extracted from within 
        another image.  In particular, it specifies that the image chip is 
        contained within the rectangle self.rect and that prior to extraction the 
        image should be rotated counter-clockwise by self.angle radians.  Finally, 
        the extracted chip should have self.rows rows and self.cols columns in it 
        regardless of the shape of self.rect.  This means that the extracted chip 
        will be stretched to fit via bilinear interpolation when necessary.
    """
    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.chip_details, rect: dlib.drectangle) -> None
        
        2. __init__(self: dlib.chip_details, rect: dlib.rectangle) -> None
        
        ensures 
            - self.rect == rect_ 
            - self.angle == 0 
            - self.rows == rect.height() 
            - self.cols == rect.width()
        
        3. __init__(self: dlib.chip_details, rect: dlib.drectangle, size: int) -> None
        
        4. __init__(self: dlib.chip_details, rect: dlib.rectangle, size: int) -> None
        
        ensures 
            - self.rect == rect 
            - self.angle == 0 
            - self.rows and self.cols is set such that the total size of the chip is as close 
              to size as possible but still matches the aspect ratio of rect. 
            - As long as size and the aspect ratio of of rect stays constant then 
              self.rows and self.cols will always have the same values.  This means 
              that, for example, if you want all your chips to have the same dimensions 
              then ensure that size is always the same and also that rect always has 
              the same aspect ratio.  Otherwise the calculated values of self.rows and 
              self.cols may be different for different chips.  Alternatively, you can 
              use the chip_details constructor below that lets you specify the exact 
              values for rows and cols.
        
        5. __init__(self: dlib.chip_details, rect: dlib.drectangle, size: int, angle: float) -> None
        
        6. __init__(self: dlib.chip_details, rect: dlib.rectangle, size: int, angle: float) -> None
        
        ensures 
            - self.rect == rect 
            - self.angle == angle 
            - self.rows and self.cols is set such that the total size of the chip is as 
              close to size as possible but still matches the aspect ratio of rect. 
            - As long as size and the aspect ratio of of rect stays constant then 
              self.rows and self.cols will always have the same values.  This means 
              that, for example, if you want all your chips to have the same dimensions 
              then ensure that size is always the same and also that rect always has 
              the same aspect ratio.  Otherwise the calculated values of self.rows and 
              self.cols may be different for different chips.  Alternatively, you can 
              use the chip_details constructor below that lets you specify the exact 
              values for rows and cols.
        
        7. __init__(self: dlib.chip_details, rect: dlib.drectangle, dims: dlib.chip_dims) -> None
        
        8. __init__(self: dlib.chip_details, rect: dlib.rectangle, dims: dlib.chip_dims) -> None
        
        ensures 
            - self.rect == rect 
            - self.angle == 0 
            - self.rows == dims.rows 
            - self.cols == dims.cols
        
        9. __init__(self: dlib.chip_details, rect: dlib.drectangle, dims: dlib.chip_dims, angle: float) -> None
        
        10. __init__(self: dlib.chip_details, rect: dlib.rectangle, dims: dlib.chip_dims, angle: float) -> None
        
        ensures 
            - self.rect == rect 
            - self.angle == angle 
            - self.rows == dims.rows 
            - self.cols == dims.cols
        
        11. __init__(self: dlib.chip_details, chip_points: dlib.dpoints, img_points: dlib.dpoints, dims: dlib.chip_dims) -> None
        
        12. __init__(self: dlib.chip_details, chip_points: dlib.points, img_points: dlib.points, dims: dlib.chip_dims) -> None
        
        requires 
            - len(chip_points) == len(img_points) 
            - len(chip_points) >= 2  
        ensures 
            - The chip will be extracted such that the pixel locations chip_points[i] 
              in the chip are mapped to img_points[i] in the original image by a 
              similarity transform.  That is, if you know the pixelwize mapping you 
              want between the chip and the original image then you use this function 
              of chip_details constructor to define the mapping. 
            - self.rows == dims.rows 
            - self.cols == dims.cols 
            - self.rect and self.angle are computed based on the given size of the output chip 
              (specified by dims) and the similarity transform between the chip and 
              image (specified by chip_points and img_points).
        """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ __repr__(self: dlib.chip_details) -> str """
        return ""

    def __str__(self): # real signature unknown; restored from __doc__
        """ __str__(self: dlib.chip_details) -> str """
        return ""

    angle = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    cols = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    rect = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    rows = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



