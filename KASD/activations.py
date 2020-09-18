'''
Description:
    Contains native tools for keras activation objects such as
    identification, serialization and deserialization with minor
    modifications for compatibility. This module also offers novel
    functionality for custom keras objects and categorization.

Customization:
    Use the activations.custom_object or custom decorator to label
    custom activations for deserialization support. This decorator
    links custom objects to the _GLOBAL_CUSTOM_OBJECTS from
    keras.utils.generic_utils._GLOBAL_CUSTOM_OBJECTS for
    native support.
    
    Use the activations.label or label decorator to label native or
    custom activations.
    
    Example on how to add and label custom activations:
        from KASD.activations import activations, custom, label
        
        @activations.label('new')
        @activations.custom
        def activation(...):
            pass
         
        or
        
        @label('new')
        @custom
        def activation(...):
            pass
            
        or
        
        @activations
        def activation(...):
            pass
        
        or
        
        @activations(labels='new')
        def activation(...):
            pass
    
    Labeling without using decorator.
        activations.label('conv', func=keras.activations.linear)
        
        or
        
        label('conv', func=keras.activations.linear)
    
Functionality:
    *deserialize        : (func) Used to deserialize serials of activations.
    *serialize          : (func) See keras.activations.serialize.
    *get                : (func) Used to identify activations.
    *activations        : (class) Used to categorise keras activations. 
    *custom             : (@func) Used to identify custom keras activations.
    *label              : (@func) USed to label custom/native keras activations.
'''

from keras.activations import serialize, deserialize as _deserialize
import warnings
import six

def deserialize(identifier, custom_objects=None):
    try:
        return _deserialize(identifier, custom_objects=custom_objects)
    except:
        raise AttributeError("Use the activations.custom decorator for custom object support.")

def get(identifier):
    """Get the `identifier` activation function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The activation function, `linear` if `identifier` is None.
    """
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        try:
            return deserialize(identifier)
        except:
            return None
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                'Do not pass a layer instance (such as {identifier}) as the '
                'activation argument of another layer. Instead, advanced '
                'activation layers should be used just like any other '
                'layer in a model.'.format(
                    identifier=identifier.__class__.__name__))
        return identifier
    else:
        return None




from . import Collection

class _Activations(Collection):
    pass

_GLOBAL_LABELS = {
'activations': ['linear', 'exponential', 'hard_sigmoid', 'sigmoid', 'tanh', 'relu', 'softsign', 'softplus', 'selu', 'elu', 'softmax']}

activations = _Activations(_GLOBAL_LABELS['activations'], _type='function')
custom = activations.custom
label = activations.label

from keras import activations as _activations

_dir = dir(_activations)
[activations.label(key, func=name) for key, value in _GLOBAL_LABELS.items() for name in value if name in _dir] #ensures native keras objects

