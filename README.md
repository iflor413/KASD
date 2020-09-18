This library offers an Advanced Serialization and Deserialization
system with custom keras object support. Built layers (tensors) can
be serialized into a dictionary that includes information used for
deserialized. A collection of tensors can also be serialized into a
series which can be deserialized into a list of inter-connected tensors.
Custom keras objects can also be supported natively with keras and
with the advanced serialization and deserialization functionality.

## Note:
A decorator can be used to register custom keras objects. This
decorator links custom objects to the _GLOBAL_CUSTOM_OBJECTS
from keras.utils.generic_utils._GLOBAL_CUSTOM_OBJECTS for
native support. See below on how to add custom keras objects.

**Custom Layer:**
```
from KASD.layers import layers, custom

@layers.custom
class layer(...):
    pass
 
#or

@custom
class layer(...):
    pass
    
#or

@layers
class layer(...):
    pass
```

**Custom Constraint:**
```
from KASD.constraints import constraints, custom

@constraints.custom
class constraint(...):
    pass
 
#or

@custom
class constraint(...):
    pass
    
#or

@constraints
class constraint(...):
    pass
```

**Custom Initializer:**
```
from KASD.initializers import initializers, custom

@initializers.custom
class initializer(...):
    pass
 
#or

@custom
class initializer(...):
    pass
    
#or

@initializers
class initializer(...):
    pass
```

**Custom Regularizer:**
```
from KASD.regularizers import regularizers, custom

@regularizers.custom
class regularizer(...):
    pass
 
#or

@custom
class regularizer(...):
    pass
    
#or

@regularizers
class regularizer(...):
    pass
```

**Custom Activation:**
```
from KASD.activations import activations, custom

@activations.custom
def activation(...):
    pass
 
#or

@custom
def activation(...):
    pass
    
#or

@activations
def activation(...):
    pass
```

## Repository:
https://github.com/iflor413/KASD

## Compatibility:
**Python:** >= 2.7
**Keras:** 2.0.8, 2.1.2, 2.1.3, 2.1.4, 2.1.5, 2.1.6, 2.2.0, 2.2.2, 2.2.4, 2.3.1

