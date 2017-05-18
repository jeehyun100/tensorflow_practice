# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.31
#
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _svmc

#from svmc import C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
#from svmc import LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED

import new
new_instancemethod = new.instancemethod
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


C_SVC = _svmc.C_SVC
NU_SVC = _svmc.NU_SVC
ONE_CLASS = _svmc.ONE_CLASS
EPSILON_SVR = _svmc.EPSILON_SVR
NU_SVR = _svmc.NU_SVR
LINEAR = _svmc.LINEAR
POLY = _svmc.POLY
RBF = _svmc.RBF
SIGMOID = _svmc.SIGMOID
PRECOMPUTED = _svmc.PRECOMPUTED
class svm_parameter(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, svm_parameter, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, svm_parameter, name)
    __repr__ = _swig_repr
    __swig_setmethods__["svm_type"] = _svmc.svm_parameter_svm_type_set
    __swig_getmethods__["svm_type"] = _svmc.svm_parameter_svm_type_get
    if _newclass:svm_type = _swig_property(_svmc.svm_parameter_svm_type_get, _svmc.svm_parameter_svm_type_set)
    __swig_setmethods__["kernel_type"] = _svmc.svm_parameter_kernel_type_set
    __swig_getmethods__["kernel_type"] = _svmc.svm_parameter_kernel_type_get
    if _newclass:kernel_type = _swig_property(_svmc.svm_parameter_kernel_type_get, _svmc.svm_parameter_kernel_type_set)
    __swig_setmethods__["degree"] = _svmc.svm_parameter_degree_set
    __swig_getmethods__["degree"] = _svmc.svm_parameter_degree_get
    if _newclass:degree = _swig_property(_svmc.svm_parameter_degree_get, _svmc.svm_parameter_degree_set)
    __swig_setmethods__["gamma"] = _svmc.svm_parameter_gamma_set
    __swig_getmethods__["gamma"] = _svmc.svm_parameter_gamma_get
    if _newclass:gamma = _swig_property(_svmc.svm_parameter_gamma_get, _svmc.svm_parameter_gamma_set)
    __swig_setmethods__["coef0"] = _svmc.svm_parameter_coef0_set
    __swig_getmethods__["coef0"] = _svmc.svm_parameter_coef0_get
    if _newclass:coef0 = _swig_property(_svmc.svm_parameter_coef0_get, _svmc.svm_parameter_coef0_set)
    __swig_setmethods__["cache_size"] = _svmc.svm_parameter_cache_size_set
    __swig_getmethods__["cache_size"] = _svmc.svm_parameter_cache_size_get
    if _newclass:cache_size = _swig_property(_svmc.svm_parameter_cache_size_get, _svmc.svm_parameter_cache_size_set)
    __swig_setmethods__["eps"] = _svmc.svm_parameter_eps_set
    __swig_getmethods__["eps"] = _svmc.svm_parameter_eps_get
    if _newclass:eps = _swig_property(_svmc.svm_parameter_eps_get, _svmc.svm_parameter_eps_set)
    __swig_setmethods__["C"] = _svmc.svm_parameter_C_set
    __swig_getmethods__["C"] = _svmc.svm_parameter_C_get
    if _newclass:C = _swig_property(_svmc.svm_parameter_C_get, _svmc.svm_parameter_C_set)
    __swig_setmethods__["nr_weight"] = _svmc.svm_parameter_nr_weight_set
    __swig_getmethods__["nr_weight"] = _svmc.svm_parameter_nr_weight_get
    if _newclass:nr_weight = _swig_property(_svmc.svm_parameter_nr_weight_get, _svmc.svm_parameter_nr_weight_set)
    __swig_setmethods__["weight_label"] = _svmc.svm_parameter_weight_label_set
    __swig_getmethods__["weight_label"] = _svmc.svm_parameter_weight_label_get
    if _newclass:weight_label = _swig_property(_svmc.svm_parameter_weight_label_get, _svmc.svm_parameter_weight_label_set)
    __swig_setmethods__["weight"] = _svmc.svm_parameter_weight_set
    __swig_getmethods__["weight"] = _svmc.svm_parameter_weight_get
    if _newclass:weight = _swig_property(_svmc.svm_parameter_weight_get, _svmc.svm_parameter_weight_set)
    __swig_setmethods__["nu"] = _svmc.svm_parameter_nu_set
    __swig_getmethods__["nu"] = _svmc.svm_parameter_nu_get
    if _newclass:nu = _swig_property(_svmc.svm_parameter_nu_get, _svmc.svm_parameter_nu_set)
    __swig_setmethods__["p"] = _svmc.svm_parameter_p_set
    __swig_getmethods__["p"] = _svmc.svm_parameter_p_get
    if _newclass:p = _swig_property(_svmc.svm_parameter_p_get, _svmc.svm_parameter_p_set)
    __swig_setmethods__["shrinking"] = _svmc.svm_parameter_shrinking_set
    __swig_getmethods__["shrinking"] = _svmc.svm_parameter_shrinking_get
    if _newclass:shrinking = _swig_property(_svmc.svm_parameter_shrinking_get, _svmc.svm_parameter_shrinking_set)
    __swig_setmethods__["probability"] = _svmc.svm_parameter_probability_set
    __swig_getmethods__["probability"] = _svmc.svm_parameter_probability_get
    if _newclass:probability = _swig_property(_svmc.svm_parameter_probability_get, _svmc.svm_parameter_probability_set)
    def __init__(self, *args): 
        this = _svmc.new_svm_parameter(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _svmc.delete_svm_parameter
    __del__ = lambda self : None;
svm_parameter_swigregister = _svmc.svm_parameter_swigregister
svm_parameter_swigregister(svm_parameter)

class svm_problem(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, svm_problem, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, svm_problem, name)
    __repr__ = _swig_repr
    __swig_setmethods__["l"] = _svmc.svm_problem_l_set
    __swig_getmethods__["l"] = _svmc.svm_problem_l_get
    if _newclass:l = _swig_property(_svmc.svm_problem_l_get, _svmc.svm_problem_l_set)
    __swig_setmethods__["y"] = _svmc.svm_problem_y_set
    __swig_getmethods__["y"] = _svmc.svm_problem_y_get
    if _newclass:y = _swig_property(_svmc.svm_problem_y_get, _svmc.svm_problem_y_set)
    __swig_setmethods__["x"] = _svmc.svm_problem_x_set
    __swig_getmethods__["x"] = _svmc.svm_problem_x_get
    if _newclass:x = _swig_property(_svmc.svm_problem_x_get, _svmc.svm_problem_x_set)
    def __init__(self, *args): 
        this = _svmc.new_svm_problem(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _svmc.delete_svm_problem
    __del__ = lambda self : None;
svm_problem_swigregister = _svmc.svm_problem_swigregister
svm_problem_swigregister(svm_problem)

svm_train = _svmc.svm_train
svm_cross_validation = _svmc.svm_cross_validation
svm_save_model = _svmc.svm_save_model
svm_load_model = _svmc.svm_load_model
svm_get_svm_type = _svmc.svm_get_svm_type
svm_get_nr_class = _svmc.svm_get_nr_class
svm_get_labels = _svmc.svm_get_labels
svm_get_svr_probability = _svmc.svm_get_svr_probability
svm_get_model_num_coefs = _svmc.svm_get_model_num_coefs
svm_predict_values = _svmc.svm_predict_values
svm_predict = _svmc.svm_predict
svm_predict_probability = _svmc.svm_predict_probability
svm_destroy_model = _svmc.svm_destroy_model
svm_check_parameter = _svmc.svm_check_parameter
svm_check_probability_model = _svmc.svm_check_probability_model
svm_get_model_rho = _svmc.svm_get_model_rho
svm_get_model_coefs = _svmc.svm_get_model_coefs
svm_get_model_perm = _svmc.svm_get_model_perm
svm_get_model_w2 = _svmc.svm_get_model_w2
svm_kernel_function = _svmc.svm_kernel_function
svm_get_model_SVs = _svmc.svm_get_model_SVs
get_num_SVs = _svmc.get_num_SVs
distance_of_closest_SV = _svmc.distance_of_closest_SV
new_int = _svmc.new_int
delete_int = _svmc.delete_int
int_getitem = _svmc.int_getitem
int_setitem = _svmc.int_setitem
new_double = _svmc.new_double
delete_double = _svmc.delete_double
double_getitem = _svmc.double_getitem
double_setitem = _svmc.double_setitem
svm_node_array = _svmc.svm_node_array
svm_node_array_set = _svmc.svm_node_array_set
svm_node_array_destroy = _svmc.svm_node_array_destroy
svm_node_matrix = _svmc.svm_node_matrix
svm_node_matrix_set = _svmc.svm_node_matrix_set
svm_node_matrix_destroy = _svmc.svm_node_matrix_destroy

