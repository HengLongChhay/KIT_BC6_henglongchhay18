# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class pyramid_down(__pybind11_builtins.pybind11_object):
    """
    This is a simple object to help create image pyramids.  In particular, it 
    downsamples images at a ratio of N to N-1. 
     
    Note that setting N to 1 means that this object functions like 
    pyramid_disable (defined at the bottom of this file).   
     
    WARNING, when mapping rectangles from one layer of a pyramid 
    to another you might end up with rectangles which extend slightly  
    outside your images.  This is because points on the border of an  
    image at a higher pyramid layer might correspond to points outside  
    images at lower layers.  So just keep this in mind.  Note also 
    that it's easy to deal with.  Just say something like this: 
        rect = rect.intersect(get_rect(my_image)); # keep rect inside my_image
    """
    def point_down(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        point_down(*args, **kwargs)
        Overloaded function.
        
        1. point_down(self: dlib.pyramid_down, p: dlib.point) -> dlib.dpoint
        
        2. point_down(self: dlib.pyramid_down, p: dlib.dpoint) -> dlib.dpoint
        
        Maps from pixels in a source image to the corresponding pixels in the downsampled image.
        
        3. point_down(self: dlib.pyramid_down, p: dlib.point, levels: int) -> dlib.dpoint
        
        4. point_down(self: dlib.pyramid_down, p: dlib.dpoint, levels: int) -> dlib.dpoint
        
        Applies point_down() to p levels times and returns the result.
        """
        pass

    def point_up(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        point_up(*args, **kwargs)
        Overloaded function.
        
        1. point_up(self: dlib.pyramid_down, p: dlib.point) -> dlib.dpoint
        
        2. point_up(self: dlib.pyramid_down, p: dlib.dpoint) -> dlib.dpoint
        
        Maps from pixels in a downsampled image to pixels in the original image.
        
        3. point_up(self: dlib.pyramid_down, p: dlib.point, levels: int) -> dlib.dpoint
        
        4. point_up(self: dlib.pyramid_down, p: dlib.dpoint, levels: int) -> dlib.dpoint
        
        Applies point_up() to p levels times and returns the result.
        """
        pass

    def pyramid_downsampling_rate(self): # real signature unknown; restored from __doc__
        """
        pyramid_downsampling_rate(self: dlib.pyramid_down) -> int
        
        Returns a number N that defines the downsampling rate.  In particular, images are downsampled by a factor of N to N-1.
        """
        return 0

    def rect_down(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        rect_down(*args, **kwargs)
        Overloaded function.
        
        1. rect_down(self: dlib.pyramid_down, rect: dlib.rectangle) -> dlib.rectangle
        
        2. rect_down(self: dlib.pyramid_down, rect: dlib.drectangle) -> dlib.drectangle
        
        returns drectangle(point_down(rect.tl_corner()), point_down(rect.br_corner()));
         (i.e. maps rect into a downsampled)
        
        3. rect_down(self: dlib.pyramid_down, rect: dlib.rectangle, levels: int) -> dlib.rectangle
        
        4. rect_down(self: dlib.pyramid_down, rect: dlib.drectangle, levels: int) -> dlib.drectangle
        
        Applies rect_down() to rect levels times and returns the result.
        """
        pass

    def rect_up(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        rect_up(*args, **kwargs)
        Overloaded function.
        
        1. rect_up(self: dlib.pyramid_down, rect: dlib.rectangle) -> dlib.rectangle
        
        2. rect_up(self: dlib.pyramid_down, rect: dlib.drectangle) -> dlib.drectangle
        
        returns drectangle(point_up(rect.tl_corner()), point_up(rect.br_corner()));
         (i.e. maps rect into a parent image)
        
        3. rect_up(self: dlib.pyramid_down, rect: dlib.rectangle, levels: int) -> dlib.rectangle
        
        4. rect_up(self: dlib.pyramid_down, p: dlib.drectangle, levels: int) -> dlib.drectangle
        
        Applies rect_up() to rect levels times and returns the result.
        """
        pass

    def __call__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __call__(*args, **kwargs)
        Overloaded function.
        
        1. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),uint8]) -> numpy.ndarray[(rows,cols),uint8]
        
        2. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),uint16]) -> numpy.ndarray[(rows,cols),uint16]
        
        3. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),uint32]) -> numpy.ndarray[(rows,cols),uint32]
        
        4. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),uint64]) -> numpy.ndarray[(rows,cols),uint64]
        
        5. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),int8]) -> numpy.ndarray[(rows,cols),int8]
        
        6. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),int16]) -> numpy.ndarray[(rows,cols),int16]
        
        7. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),int32]) -> numpy.ndarray[(rows,cols),int32]
        
        8. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),int64]) -> numpy.ndarray[(rows,cols),int64]
        
        9. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),float32]) -> numpy.ndarray[(rows,cols),float32]
        
        10. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols),float64]) -> numpy.ndarray[(rows,cols),float64]
        
        11. __call__(self: dlib.pyramid_down, img: numpy.ndarray[(rows,cols,3),uint8]) -> numpy.ndarray[(rows,cols,3),uint8]
        
        - Downsamples img to make a new image that is roughly (pyramid_downsampling_rate()-1)/pyramid_downsampling_rate()  
          times the size of the original image.   
        - The location of a point P in original image will show up at point point_down(P) 
          in the downsampled image.   
        - Note that some points on the border of the original image might correspond to  
          points outside the downsampled image.
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __init__(*args, **kwargs)
        Overloaded function.
        
        1. __init__(self: dlib.pyramid_down, N: int) -> None
        
        Creates this class with the provided downsampling rate. i.e. pyramid_downsampling_rate()==N. 
        N must be in the range 1 to 20.
        
        2. __init__(self: dlib.pyramid_down) -> None
        
        Creates this class with pyramid_downsampling_rate()==2
        """
        pass


