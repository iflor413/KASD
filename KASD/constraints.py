'''
Description:
    Contains native tools for keras constraint objects such as
    identification, serialization and deserialization with minor
    modifications for compatibility. This module also offers novel
    functionality for custom keras objects and categorization.

Customization:
    Use the constraints.custom_object or custom decorator to label
    custom constraints for deserialization support. This decorator
    links custom objects to the _GLOBAL_CUSTOM_OBJECTS from
    keras.utils.generic_utils._GLOBAL_CUSTOM_OBJECTS for
    native support.
    
    Use the constraints.label or label decorator to label native or
    custom constraints.
    
    Example on how to add and label custom constraints:
        from KASD.constraints import constraints, custom, label
        
        @constraints.label('new')
        @constraints.custom
        class constraint(...):
            pass
         
        or
        
        @label('new')
        @custom
        class constraint(...):
            pass
            
        or
        
        @constraints
        class constraint(...):
            pass
        
        or
        
        @constraints(labels='new')
        class constraint(...):
            pass
    
    Labeling without using decorator.
        constraints.label('conv', func=keras.constraints.MaxNorm)
        
        or
        
        label('conv', func=keras.constraints.MaxNorm)
    
Functionality:
    *deserialize        : (func) Used to deserialize serials of constraints.
    *serialize          : (func) See keras.constraints.serialize.
    *get                : (func) Used to identify constraints.
    *constraints        : (class) Used to categorise keras constraints. 
    *custom             : (@func) Used to identify custom keras constraints.
    *label              : (@func) USed to label custom/native keras constraints.
'''

from keras.constraints import serialize, deserialize as _deserialize
import six

def deserialize(identifier, custom_objects=None):
    try:
        return _deserialize(identifier, custom_objects=custom_objects)
    except:
        raise AttributeError("Use the constraints.custom decorator for custom object support.")

def get(identifier):
    if isinstance(identifier, dict):
        try:
            return deserialize(identifier)
        except:
            return None
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return get(config)
    elif callable(identifier):
        return identifier
    else:
        return None




from . import Collection

class _Constraints(Collection):
    pass

_GLOBAL_LABELS = {
'constraints': ['Constraint', 'NonNeg', 'MaxNorm', 'UnitNorm', 'MinMaxNorm']}

constraints = _Constraints(_GLOBAL_LABELS['constraints'])
custom = constraints.custom
label = constraints.label

from keras import constraints as _constraints

_dir = dir(_constraints)
[constraints.label(key, func=name) for key, value in _GLOBAL_LABELS.items() for name in value if name in _dir] #ensures native keras objects

