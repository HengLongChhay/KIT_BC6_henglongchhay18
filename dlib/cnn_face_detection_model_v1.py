# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class cnn_face_detection_model_v1(__pybind11_builtins.pybind11_object):
    """ This object detects human faces in an image.  The constructor loads the face detection model from a file. You can download a pre-trained model from http://dlib.net/files/mmod_human_face_detector.dat.bz2. """
    def __call__(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        __call__(*args, **kwargs)
        Overloaded function.
        
        1. __call__(self: dlib.cnn_face_detection_model_v1, imgs: list, upsample_num_times: int=0, batch_size: int=128) -> std::__1::vector<std::__1::vector<dlib::mmod_rect, std::__1::allocator<dlib::mmod_rect> >, std::__1::allocator<std::__1::vector<dlib::mmod_rect, std::__1::allocator<dlib::mmod_rect> > > >
        
        takes a list of images as input returning a 2d list of mmod rectangles
        
        2. __call__(self: dlib.cnn_face_detection_model_v1, img: array, upsample_num_times: int=0) -> std::__1::vector<dlib::mmod_rect, std::__1::allocator<dlib::mmod_rect> >
        
        Find faces in an image using a deep learning model.
                  - Upsamples the image upsample_num_times before running the face 
                    detector.
        """
        pass

    def __init__(self, filename): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.cnn_face_detection_model_v1, filename: str) -> None """
        pass


