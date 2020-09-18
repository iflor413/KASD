'''
Description:
    Contains native tools for keras regularizer objects such as
    identification, serialization and deserialization with minor
    modifications for compatibility. This module also offers novel
    functionality for custom keras objects and categorization.

Customization:
    Use the regularizers.custom_object or custom decorator to label
    custom regularizers for deserialization support. This decorator
    links custom objects to the _GLOBAL_CUSTOM_OBJECTS from
    keras.utils.generic_utils._GLOBAL_CUSTOM_OBJECTS for
    native support.
    
    Use the regularizers.label or label decorator to label native or
    custom regularizers.
    
    Example on how to add and label custom regularizers:
        from KASD.regularizers import regularizers, custom, label
        
        @regularizers.label('new')
        @regularizers.custom
        class regularizer(...):
            pass
         
        or
        
        @label('new')
        @custom
        class regularizer(...):
            pass
            
        or
        
        @regularizers
        class regularizer(...):
            pass
        
        or
        
        @regularizers(labels='new')
        class regularizer(...):
            pass
    
    Labeling without using decorator.
        regularizers.label('conv', func=keras.regularizers.L1L2)
        
        or
        
        label('conv', func=keras.regularizers.L1L2)
    
Functionality:
    *deserialize        : (func) Used to deserialize serials of regularizers.
    *serialize          : (func) See keras.regularizers.serialize.
    *get                : (func) Used to identify regularizers.
    *regularizers       : (class) Used to categorise keras regularizers. 
    *custom             : (@func) Used to identify custom keras regularizers.
    *label              : (@func) USed to label custom/native keras regularizers.
'''

from keras.regularizers import serialize, deserialize as _deserialize
import six

def deserialize(identifier, custom_objects=None):
    try:
        return _deserialize(identifier, custom_objects=custom_objects)
    except:
        raise AttributeError("Use the regularizers.custom decorator for custom object support.")

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

class _Regularizers(Collection):
    pass

_GLOBAL_LABELS = {
'regularizers': ['L1L2']}

regularizers = _Regularizers(_GLOBAL_LABELS['regularizers'])
custom = regularizers.custom
label = regularizers.label

from keras import regularizers as _regularizers

_dir = dir(_regularizers)
[regularizers.label(key, func=name) for key, value in _GLOBAL_LABELS.items() for name in value if name in _dir] #ensures native keras objects

