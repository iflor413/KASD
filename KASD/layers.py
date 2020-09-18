'''
Description:
    Contains advanced tools for keras layer objects such as
    identification, serialization and deserialization based
    on native keras functionality. This module also offers novel
    functionality for custom keras objects and categorization.
    
    This module was based on the functional API.

Customization:
    Use the layers.custom_object or custom decorator to label
    custom layers for deserialization support. This decorator
    links custom objects to the _GLOBAL_CUSTOM_OBJECTS from
    keras.utils.generic_utils._GLOBAL_CUSTOM_OBJECTS for
    native support.
    
    Use the layers.label or label decorator to label native or
    custom layers.
    
    Example on how to add and label custom layers:
        from KASD.layers import layers, custom, label
        
        @layers.label('new')
        @layers.custom
        class layer(...):
            pass
         
        or
        
        @label('new')
        @custom
        class layer(...):
            pass
            
        or
        
        @layers
        class layer(...):
            pass
        
        or
        
        @layers(labels='new')
        class layer(...):
            pass
    
    Labeling without using decorator.
        layers.label('conv', func=keras.layers.Conv1D)
        
        or
        
        label('conv', func=keras.layers.Conv1D)
    
Functionality:
    *is_advanced_serial : (func) Used to identify advanced serials.
    *is_advanced_series : (func) Used to identity advanced series.
    *deserialize        : (func) Used to deserialize layers.
    *serialize          : (func) Used to serialize native and advanced series/serials of layers.
    *update             : (func) Used to update an advanced serial to accomodate attribute changes.
    *get                : (func) Used to identify layers and tensors.
    *layers             : (class) Used to categorise keras layers. 
    *custom             : (@func) Used to identify custom keras layers.
    *label              : (@func) USed to label custom/native keras layers.
'''
try:
    from keras.backend import is_tensor
except:
    def is_tensor(obj):
        return hasattr(obj, '_keras_history') and hasattr(obj._keras_history[0], 'built') and obj._keras_history[0].built is True

from keras.layers import serialize as _serialize, deserialize as _deserialize
from keras.layers import Input, Flatten, Dense, Reshape

from copy import deepcopy
import numpy as np

def is_advanced_serial(identifier):
    return isinstance(identifier, dict) and (
            'config' in identifier and
            'class_name' in identifier and
            'input' in identifier and
            'input_shape' in identifier and
            'output_shape' in identifier)

def is_advanced_series(identifier):
    if isinstance(identifier, dict):
        is_true = True
        
        for value in identifier.values():
            if not is_advanced_serial(value):
                is_true = False
                break
        
        return is_true
    else:
        return False

def deserialize(identifier, custom_objects=None, catch_input_errors=False):
    '''
        This function is used to deserialize native and
        advanced serials into built/unbuilt layers or a list
        of tensors. This function also inherits the native
        functionality of deserialization from
        keras.layers.deserialize.
        
        *catch_input_errors:    When enabled, allows for the
                                function to catch and patch
                                input errors. This only occurs
                                when the output shape of a tensor
                                does not match the expected input
                                shape. A converter is used which
                                include atleast one of the following:
                                Flatten, Dense, and Reshape layer.
        
        returns tensor/layer/[tensors]
    '''
    def patch_name(input_names, class_name):
        if not isinstance(input_names, (list, tuple)):
            input_names = [input_names]
        
        return "-".join(input_names)+"/{}/patch".format(class_name)

    def get_input(input_name, input_shape, series={}):
        if input_name in series:
            return series[input_name]
        else:
            new = Input(batch_shape=input_shape, name=input_name)
            series.update({input_name: new})
            return new

    def create_tensor(cls, _input, adv_serial):
        if catch_input_errors:
            tensor = None
            new_tensors = None
            
            try:
                tensor = deepcopy(cls)(_input) #even if building tensor fails, in keras/tensorflow the layer class is still built even in exeption. Error fixed with deepcopy.
            except:
                print("Addendum between {}({}) identified. Patching discrepency.".format(adv_serial['config']['name'], adv_serial['input'][0] if len(adv_serial['input']) == 1 else adv_serial['input']))
                
                if not isinstance(_input, (list, tuple)):
                    _input = [_input]
                
                new_tensors = []
                new_input = []
                for i in range(len(_input)):
                    intended_input_shape = tuple(adv_serial['input_shape'][1:] if len(_input) == 1 else adv_serial['input_shape'][i][1:]) #ignore batch_size
                    current_input_shape = tuple(_input[i]._keras_history[0].output_shape[1:]) #ignore batch_size
                    
                    if np.all(intended_input_shape==current_input_shape):
                        new_input.append(_input[i])
                    else:
                        layer = _input[i]
                        
                        if len(current_input_shape) > 1: #flatten OG shape if >= 3D tensor
                            layer = Flatten(name=patch_name(layer._keras_history[0].name, 'Flatten'))(layer)
                            new_tensors.append(layer)
                        
                        if np.prod(intended_input_shape) != np.prod(layer.shape[1:]):
                            layer = Dense(np.prod(intended_input_shape), name=patch_name(layer._keras_history[0].name, 'Dense'))(layer) #correct size
                            new_tensors.append(layer)
                        
                        if len(intended_input_shape) > 1: #correct shape if intended shape >= 3D tensor
                            layer = Reshape(target_shape=intended_input_shape, name=patch_name(layer._keras_history[0].name, 'Reshape'))(layer)
                            new_tensors.append(layer)
                        
                        new_input.append(layer)
                    
                if len(new_input) == 1:
                    new_input = new_input[0]
                
                tensor = cls(new_input)
            
            return new_tensors, tensor
        else:
            return None, cls(_input)

    if is_advanced_serial(identifier): #identifier is an advanced_serial
        native_serial = {'class_name': identifier['class_name'], 'config': deepcopy(identifier['config'])}
        cls = _deserialize(native_serial, custom_objects=custom_objects)
        
        if len(identifier['input']) == 1:
            _input = get_input(identifier['input'][0], identifier['input_shape'])
        else:
            _input = [get_input(identifier['input'][i], identifier['input_shape'][i]) for i in range(len(identifier['input']))]
        
        return create_tensor(cls, _input, identifier)[1]
    elif is_advanced_series(identifier): #identifier is an advanced_series
        series = {}
        for key, value in identifier.items():
            cls = _deserialize({'class_name': value['class_name'], 'config': deepcopy(value['config'])}, custom_objects=custom_objects)
            
            if len(value['input']) == 1:
                _input = get_input(value['input'][0], value['input_shape'], series=series)
            else:
                _input = [get_input(value['input'][i], value['input_shape'][i], series=series) for i in range(len(value['input']))]
            
            new_tensors, tensor = create_tensor(cls, _input, value)
            
            if not new_tensors is None:
                [series.update({new_tensor._keras_history[0].name: new_tensor}) for new_tensor in new_tensors]
            series[key] = tensor
        
        return list(series.values())
    else:
        try:
            return _deserialize(identifier, custom_objects=custom_objects)
        except:
            raise AttributeError("Use the layers.custom decorator for custom object support.")

def serialize(identifier):
    '''
        This function is used to convert a list or a single
        built layer into an advanced series or serial,
        respectivly. This function also inherits the native
        serialization functionality from keras.layers.serialize.
        
        An advanced serial is composed of the layer's
        native serial components, 'class_name' and 'config',
        and 3 new components which includes:
            'input':        List of names of layer(s) that act
                            as an input for the serialized layer.
                            Name is derived directly from
                            input serial['config']['name'].
                            
            'input_shape':  Describes the output shape of each
                            input.
                            
            'output_shape': Described the output shape of the
                            serialized layer.
        
        An advanced series is composed of advanced serials
        represented by their unique names as keys.
        
        returns dict
    '''
    if isinstance(identifier, (list, tuple)):
        series = {}
        for item in identifier:
            if isinstance(item, list): #tensors with multiple outputs have the same serial
                item = item[0]
            
            if is_tensor(item):
                item = item._keras_history[0]
            
            assert hasattr(item, 'built') and item.built #assert tensor is built
            
            if item.__class__.__name__ != 'InputLayer': #Inputs are assumed when serialized as 'input' and 'input_shape' keys
                series[item.name] = serialize(item)
        
        return series
    elif is_tensor(identifier) or (not is_tensor(identifier) and hasattr(identifier, 'built') and identifier.built):
        if is_tensor(identifier):
            identifier = identifier._keras_history[0]
        
        serial = _serialize(identifier)
        serial['input'] = [input_._keras_history[0].name for input_ in identifier.input] if isinstance(identifier.input, list) else [identifier.input._keras_history[0].name]
        serial['input_shape'] = identifier.input_shape
        serial['output_shape'] = identifier.output_shape
        
        return serial
    else:
        return _serialize(identifier)

def update(serial):
    '''
        This function is used to update the output_shape
        of a serial after modifying attributes.
        
        returns None
    '''
    assert is_advanced_serial(serial)
    
    try:
        layer = deserialize({'class_name': serial['class_name'], 'config': deepcopy(serial['config'])})
        serial['output_shape'] = layer.compute_output_shape(serial['input_shape'])
    except: #if serial is invalid, do nothing
        pass

def get(identifier):
    '''
        Used to identify the 'identifier' through
        deserialization. None is returned if 'identifier'
        cannot be deserialization nor identified.
        
        returns tensor/layer/None
    '''
    try:
        if callable(identifier):
            return identifier
        else:
            return deserialize(identifier)
    except:
        return None




from . import Collection

class _Layers(Collection):
    pass

_GLOBAL_LABELS = {
'layers': ['InputLayer', 'Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'Lambda', 'Dot', 'Dense', 'Activation', 'Dropout', 'Flatten', 'Reshape', 'Permute', 'RepeatVector',
'ActivityRegularization', 'Masking', 'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D', 'Conv1D', 'Conv2D', 'SeparableConv1D', 'SeparableConv2D', 'DepthwiseConv2D', 'Conv2DTranspose', 'Conv3D',
'Conv3DTranspose', 'Cropping1D', 'Cropping2D', 'Cropping3D', 'UpSampling1D', 'UpSampling2D', 'UpSampling3D', 'ZeroPadding1D', 'ZeroPadding2D', 'ZeroPadding3D', 'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D',
'AveragePooling1D', 'AveragePooling2D', 'AveragePooling3D', 'GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'GlobalMaxPooling3D', 'GlobalAveragePooling2D', 'GlobalAveragePooling1D', 'GlobalAveragePooling3D',
'LocallyConnected1D', 'LocallyConnected2D', 'RNN', 'SimpleRNN', 'GRU', 'LSTM', 'SimpleRNNCell', 'GRUCell', 'LSTMCell', 'StackedRNNCells', 'CuDNNGRU', 'CuDNNLSTM', 'BatchNormalization', 'Embedding',
'GaussianNoise', 'GaussianDropout', 'AlphaDropout', 'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'Softmax', 'ReLU', 'Bidirectional', 'TimeDistributed', 'ConvLSTM2D', 'ConvLSTM2DCell'],
'merge': ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'Dot'],
'core': ['Dense', 'Activation', 'Dropout', 'Flatten', 'Reshape', 'Permute', 'RepeatVector', 'Lambda', 'ActivityRegularization', 'Masking', 'SpatialDropout1D', 'SpatialDropout2D', 'SpatialDropout3D'],
'convolutional': ['Conv1D', 'Conv2D', 'SeparableConv1D', 'SeparableConv2D', 'DepthwiseConv2D', 'Conv2DTranspose', 'Conv3D', 'Conv3DTranspose', 'Cropping1D', 'Cropping2D', 'Cropping3D', 'UpSampling1D',
'UpSampling2D', 'UpSampling3D', 'ZeroPadding1D', 'ZeroPadding2D', 'ZeroPadding3D'],
'pooling': ['MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D', 'AveragePooling1D', 'AveragePooling2D', 'AveragePooling3D', 'GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'GlobalMaxPooling3D',
'GlobalAveragePooling2D', 'GlobalAveragePooling1D', 'GlobalAveragePooling3D'],
'local': ['LocallyConnected1D', 'LocallyConnected2D'],
'recurrent': ['RNN', 'SimpleRNN', 'GRU', 'LSTM', 'SimpleRNNCell', 'GRUCell', 'LSTMCell', 'StackedRNNCells'],
'cudnn_recurrent': ['CuDNNGRU', 'CuDNNLSTM'],
'normalization': ['BatchNormalization'],
'embeddings': ['Embedding'],
'noise': ['GaussianNoise', 'GaussianDropout', 'AlphaDropout'],
'advanced_activations': ['LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU', 'Softmax', 'ReLU'],
'wrappers': ['Bidirectional', 'TimeDistributed'],
'convolutional_recurrent': ['ConvRNN2D', 'ConvLSTM2D', 'ConvLSTM2DCell'],
'cells': ['SimpleRNNCell', 'GRUCell', 'LSTMCell', 'StackedRNNCells', 'ConvLSTM2DCell']}

layers = _Layers(_GLOBAL_LABELS['layers'])
custom = layers.custom
label = layers.label

try:
    from keras.layers.convolutional_recurrent import ConvRNN2D as _ConvRNN2D
    custom(_ConvRNN2D)
except:
    pass

from keras import layers as _layers

_dir = dir(_layers)
[layers.label(key, func=name) for key, value in _GLOBAL_LABELS.items() for name in value if name in _dir] #ensures native keras objects


