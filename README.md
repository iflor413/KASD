This library is an extension to the serialization and deserialization
functionality in the Keras machine learning API. However this library
is specifically built for the keras functional API.

An Advanced Serialization and Deserialization functionality is offered
by this library along with custom keras object support, where built layers
can be serialized into a dictionary that includes:  
- **'config': (classical serial['config'], dict)**  
- **'class_name': (classical serial['class_name'], str)**  
- **'input': (listed name of input(s), [str])**  
- **'input_shape': (input_shape of tensor,)**  
- **'output_shape': (output_shape of tensor,)**  

, and can be deserialized into a built layer. This library is also capable
of serializing and deserializing a list of build layers into a series
dictionary. This series is in the format {'name': {advanced serial}, ...}, where
'name' is the name of the built layer found in serial['config']['name'].  

Custom keras objects can also be supported natively with keras and with the
advanced serialization and deserialization functionality through the use
of wrappers.  

## Note:
A decorator can be used to register custom keras objects. This
decorator links custom objects to the _GLOBAL_CUSTOM_OBJECTS
from keras.utils.generic_utils, for native support. See below
on how to add custom keras objects.  

**Custom Layer:**
```
>>> from KASD.layers import layers, custom
>>> 
>>> @layers.custom
>>> class layer(...):
>>>     pass
>>>  
>>> #or
>>> 
>>> @custom
>>> class layer(...):
>>>     pass
>>>     
>>> #or
>>> 
>>> @layers
>>> class layer(...):
>>>     pass
```

**Custom Constraint:**
```
>>> from KASD.constraints import constraints, custom
>>> 
>>> @constraints.custom
>>> class constraint(...):
>>>     pass
>>>  
>>> #or
>>> 
>>> @custom
>>> class constraint(...):
>>>     pass
>>>     
>>> #or
>>> 
>>> @constraints
>>> class constraint(...):
>>>     pass
```

**Custom Initializer:**
```
>>> from KASD.initializers import initializers, custom
>>> 
>>> @initializers.custom
>>> class initializer(...):
>>>     pass
>>>  
>>> #or
>>> 
>>> @custom
>>> class initializer(...):
>>>     pass
>>>     
>>> #or
>>> 
>>> @initializers
>>> class initializer(...):
>>>     pass
```

**Custom Regularizer:**
```
>>> from KASD.regularizers import regularizers, custom
>>> 
>>> @regularizers.custom
>>> class regularizer(...):
>>>     pass
>>>  
>>> #or
>>> 
>>> @custom
>>> class regularizer(...):
>>>     pass
>>>     
>>> #or
>>> 
>>> @regularizers
>>> class regularizer(...):
>>>     pass
```

**Custom Activation:**
```
>>> from KASD.activations import activations, custom
>>> 
>>> @activations.custom
>>> def activation(...):
>>>     pass
>>>  
>>> #or
>>> 
>>> @custom
>>> def activation(...):
>>>     pass
>>>     
>>> #or
>>> 
>>> @activations
>>> def activation(...):
>>>     pass
```

## Repository:
https://github.com/iflor413/KASD

## Compatibility:
**Python:** >= 2.7  
**Keras:** 2.0.8, 2.1.2, 2.1.3, 2.1.4, 2.1.5, 2.1.6, 2.2.0, 2.2.2, 2.2.4, 2.3.1  

