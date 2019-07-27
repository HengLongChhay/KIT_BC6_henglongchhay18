# encoding: utf-8
# module dlib
# from /Users/henglongchay/face_recognition/lib/python3.7/site-packages/dlib.cpython-37m-darwin.so
# by generator 1.147
# no doc

# imports
import dlib.cuda as cuda # <module 'dlib.cuda'>
import dlib.image_dataset_metadata as image_dataset_metadata # <module 'dlib.image_dataset_metadata'>
import pybind11_builtins as __pybind11_builtins


class svm_rank_trainer_sparse(__pybind11_builtins.pybind11_object):
    # no doc
    def be_quiet(self): # real signature unknown; restored from __doc__
        """ be_quiet(self: dlib.svm_rank_trainer_sparse) -> None """
        pass

    def be_verbose(self): # real signature unknown; restored from __doc__
        """ be_verbose(self: dlib.svm_rank_trainer_sparse) -> None """
        pass

    def set_prior(self, arg0, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
        """ set_prior(self: dlib.svm_rank_trainer_sparse, arg0: dlib::decision_function<dlib::sparse_linear_kernel<std::__1::vector<std::__1::pair<unsigned long, double>, std::__1::allocator<std::__1::pair<unsigned long, double> > > > >) -> None """
        pass

    def train(self, *args, **kwargs): # real signature unknown; restored from __doc__
        """
        train(*args, **kwargs)
        Overloaded function.
        
        1. train(self: dlib.svm_rank_trainer_sparse, arg0: dlib.sparse_ranking_pair) -> dlib::decision_function<dlib::sparse_linear_kernel<std::__1::vector<std::__1::pair<unsigned long, double>, std::__1::allocator<std::__1::pair<unsigned long, double> > > > >
        
        2. train(self: dlib.svm_rank_trainer_sparse, arg0: dlib.sparse_ranking_pairs) -> dlib::decision_function<dlib::sparse_linear_kernel<std::__1::vector<std::__1::pair<unsigned long, double>, std::__1::allocator<std::__1::pair<unsigned long, double> > > > >
        """
        pass

    def __init__(self): # real signature unknown; restored from __doc__
        """ __init__(self: dlib.svm_rank_trainer_sparse) -> None """
        pass

    c = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    epsilon = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    force_last_weight_to_1 = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    has_prior = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    learns_nonnegative_weights = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    max_iterations = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



