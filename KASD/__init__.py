from keras.utils.generic_utils import get_custom_objects

from types import FunctionType as FunctionType
from random import choice as choice

#same instance from keras.utils.generic_utils._GLOBAL_CUSTOM_OBJECTS
_GLOBAL_CUSTOM_OBJECTS = get_custom_objects()

class Collection():
    '''
    Description:
        Is a class used to store, refer and label keras objects
        by 'class_name'. When called, will return the name of a
        random custom or native keras object.
    
    Attributes:
        labels: #dict
            Is a dictionary where values are lists of keras
            objects and key are categorical labels.
        
        custom_objects: #list
            Contains custom function class_names for keras
            objects in a list. All class_names must be unique
            to self._custom_objects and self._native_objects.
        
        native_objects: #list
            Lists all native keras objects class_names.
    '''
    
    @property
    def labels(self): return self._labels
    @property
    def custom_objects(self): return self._custom_objects
    @property
    def native_objects(self): return self._native_objects
    @property
    def all(self): return self._native_objects+self._custom_objects
    
    def __init__(self, native_objects, _type='class'):
        assert _type == 'class' or _type =='function'
        
        self._labels = {}
        
        self._custom_objects = []
        self._native_objects = native_objects
        
        self._type = _type
    
    def choice(self, include=[], exclude=[], exclusive=[]):
        '''
            Returns a random str value from self.customs keys
            and self._native_objects filtered to consider include,
            exclude and exclusive lists. If exclusive is not
            empty, then all keras object names and 'include'
            that are not listed will be filtered out. Any names
            mentioned in 'exclude' will also be filtered out
            (In order include, exclusive then exclude filters
            are applied).
        '''
        def _filter(item):
            return (len(exclusive) == 0 or item in exclusive) and not item in exclude
        
        return choice(list(filter(_filter, self._native_objects + self._custom_objects + include)))
    
    ######Decorators/Wrappers######
        
    def __call__(self, func=None, labels=None):
        '''
            This is a wrapper that performs the function of
            both self.label and self.custom at the same time.
        '''
        def wrapper(func):
            self.label(labels, func=self.custom(func))
            return func
        
        return self.custom(func) if not func is None else self.custom if labels is None else wrapper
    
    def label(self, labels, func=None):
        '''
            Can be a decorator used to label a custom function,
            or can be a function when 'func' is defined (its 
            wrapper functionality will be disabled). Is used to
            label a keras object which can be seen in the dict
            self.label. labels can be a str or a list of str.
        '''
        assert isinstance(labels, (str, list, tuple))
        
        if not isinstance(labels, (list, tuple)):
            labels = (labels,)
        
        def wrapper(func):
            name = func if isinstance(func, str) else func.__name__
            
            if not name in self._native_objects+self._custom_objects:
                raise AttributeError("'{}' is not a recognized native or custom keras object.".format(func))
            
            for label in labels:
                assert isinstance(label, str)
                
                if label in self._labels:
                    if not name in self._labels[label]:
                        self._labels[label].append(name)
                else:
                    self._labels.update({label: [name]})
            
            return func
        
        if func is None:
            return wrapper
        else:
            wrapper(func)
    
    def custom(self, func):
        '''
            Is a decorator used to identify a custom keras
            object. func.__name__ must be unique in
            self._custom_objects and self._native_objects.
        '''
        name = func.__name__
        
        if self._type == 'class' and not isinstance(func, type):
            raise AttributeError("'func' must be a class type.")
        elif self._type == 'function' and not isinstance(func, FunctionType):
            raise AttributeError("'func' must be a function type.")

        if not name in self._custom_objects+self._native_objects and not name in _GLOBAL_CUSTOM_OBJECTS:
            self._custom_objects.append(name)
            
            #allows for the globalization of custom keras objects,
            #all names must be unique or they will be overwritten.
            _GLOBAL_CUSTOM_OBJECTS[name] = func
        else:
            raise AttributeError("'{}' has already been established as a custom or native objects.".format(name))
        
        return func

from . import layers
from . import activations
from . import constraints
from . import initializers
from . import regularizers













