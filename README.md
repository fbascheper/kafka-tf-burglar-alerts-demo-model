# Build a TensorFlow model for the TF Kafka burglar alerts application 

> Simple storage converters for  [Kafka Connect](https://docs.confluent.io/current/connect/index.html)
``
***

### Contents
This repository contains the code used to build a TensorFlow for Java model for the 
Kafka burglar alerts application. It makes use of 'transfer learning', i.e. it
reuses a previously trained model and re-trains the last set of layers to distinguish burglars
from non burglars.

It is based on [TensorFlow training example](https://www.tensorflow.org/tutorials/images/transfer_learning)

* A Python script 
* A set of training and validation images


### Building the model

````
make init
make test
make run
````

### End result
After training is complete, the ``saved_fine_tuned_model.pb`` is a frozen version of the h5 model and 
can be used by a TensorFlow for Java application.  
