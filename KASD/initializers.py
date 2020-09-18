'''
Description:
    Contains native tools for keras initializer objects such as
    identification, serialization and deserialization with minor
    modifications for compatibility. This module also offers novel
    functionality for custom keras objects and categorization.

Customization:
    Use the initializers.custom_object or custom decorator to label
    custom initializers for deserialization support. This decorator
    links custom objects to the _GLOBAL_CUSTOM_OBJECTS from
    keras.utils.generic_utils._GLOBAL_CUSTOM_OBJECTS for
    native support.
    
    Use the initializers.label or label decorator to label native or
    custom initializers.
    
    Example on how to add and label custom initializers:
        from KASD.initializers import initializers, custom, label
        
        @initializers.label('new')
        @initializers.custom
        class initializer(...):
            pass
         
        or
        
        @label('new')
        @custom
        class initializer(...):
            pass
            
        or
        
        @initializers
        class initializer(...):
            pass
        
        or
        
        @initializers(labels='new')
        class initializer(...):
            pass
    
    Labeling without using decorator.
        initializers.label('conv', func=keras.initializers.Zeros)
        
        or
        
        label('conv', func=keras.initializers.Zeros)
    
Functionality:
    *deserialize        : (func) Used to deserialize serials of initializers.
    *serialize          : (func) See keras.initializers.serialize.
    *get                : (func) Used to identify initializers.
    *initializers       : (class) Used to categorise keras initializers. 
    *custom             : (@func) Used to identify custom keras initializers.
    *label              : (@func) USed to label custom/native keras initializers.

FYI:
    These initializer attributes in keras layers/Tensors shown below do not use a 2D-Matrix. (exclude '2DMatrix')
        Dense.bias_initializer
        Conv1D.kernel_initializer
        Conv1D.bias_initializer
        Conv2D.kernel_initializer
        Conv2D.bias_initializer
        SeparableConv1D.depthwise_initializer
        SeparableConv1D.pointwise_initializer
        SeparableConv1D.bias_initializer
        SeparableConv2D.depthwise_initializer
        SeparableConv2D.pointwise_initializer
        SeparableConv2D.bias_initializer
        DepthwiseConv2D.depthwise_initializer
        DepthwiseConv2D.bias_initializer
        Conv2DTranspose.kernel_initializer
        Conv2DTranspose.bias_initializer
        Conv3D.kernel_initializer
        Conv3D.bias_initializer
        Conv3DTranspose.kernel_initializer
        Conv3DTranspose.bias_initializer
        LocallyConnected1D.kernel_initializer
        LocallyConnected2D.kernel_initializer
        LocallyConnected2D.bias_initializer
        SimpleRNN.bias_initializer
        GRU.bias_initializer
        LSTM.bias_initializer
        SimpleRNNCell.bias_initializer
        GRUCell.bias_initializer
        LSTMCell.bias_initializer
        CuDNNGRU.bias_initializer
        CuDNNLSTM.bias_initializer
        BatchNormalization.gamma_initializer
        BatchNormalization.beta_initializer
        BatchNormalization.moving_mean_initializer
        BatchNormalization.moving_variance_initializer
        PReLU.alpha_initializer #dependent on input_shape, shape: list(input_shape[1:]), only works for rank 3
        ConvLSTM2D.kernel_initializer
        ConvLSTM2D.recurrent_initializer
        ConvLSTM2D.bias_initializer
        ConvLSTM2DCell.kernel_initializer
        ConvLSTM2DCell.recurrent_initializer
        ConvLSTM2DCell.bias_initializer

    These initializer attributes in keras layers/Tensors show below do not use a >= 2D -Matrix (exclude, '>=2DMatrix')
        Dense.bias_initializer
        Conv1D.bias_initializer
        Conv2D.bias_initializer
        SeparableConv1D.bias_initializer
        SeparableConv2D.bias_initializer
        DepthwiseConv2D.bias_initializer
        Conv2DTranspose.bias_initializer
        Conv3D.bias_initializer
        Conv3DTranspose.bias_initializer
        SimpleRNN.bias_initializer
        GRU.bias_initializer
        LSTM.bias_initializer
        SimpleRNNCell.bias_initializer
        GRUCell.bias_initializer
        LSTMCell.bias_initializer
        CuDNNGRU.bias_initializer
        CuDNNLSTM.bias_initializer
        BatchNormalization.gamma_initializer
        BatchNormalization.beta_initializer
        BatchNormalization.moving_mean_initializer
        BatchNormalization.moving_variance_initializer
        PReLU.alpha_initializer #dependent on input_shape, shape: list(input_shape[1:]), only works for rank >= 3
        ConvLSTM2D.bias_initializer
        ConvLSTM2DCell.bias_initializer
'''

from keras.initializers import serialize, deserialize as _deserialize
import six

def deserialize(identifier, custom_objects=None):
    try:
        return _deserialize(identifier, custom_objects=custom_objects)
    except:
        raise AttributeError("Use the initializers.custom decorator for custom object support.")

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

class _Initializers(Collection):
    pass

_GLOBAL_LABELS = {
'initializers': ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'Identity'],
'2DMatrix': ['Identity'],
'>=2DMatrix': ['Identity', 'Orthogonal']}

initializers = _Initializers(_GLOBAL_LABELS['initializers'])
custom = initializers.custom
label = initializers.label

from keras import initializers as _initializers

_dir = dir(_initializers)
[initializers.label(key, func=name) for key, value in _GLOBAL_LABELS.items() for name in value if name in _dir] #ensures native keras objects

