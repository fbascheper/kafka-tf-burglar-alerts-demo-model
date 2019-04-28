# Build a TensorFlow model for a Kafka burglar alerts application 

> This project is used to build a sample model for the [Kafka Streams - TensorFlow burglar alerts application](https://github.com/fbascheper/kafka-streams-tf-burglar-alerts)

***

### Introduction
This repository contains the code used to build a TensorFlow for Java model for the 
Kafka burglar alerts application. It makes use of 'transfer learning', i.e. it
reuses a previously trained model and re-trains the last set of layers to distinguish burglars
from non burglars.

It is based on the [TensorFlow training example](https://www.tensorflow.org/tutorials/images/transfer_learning)
which demonstrates how you to implement transfer learning to create better models.

### Contents 

* A Python script 
* A set of training and validation images


### Building the model

````
make init
make test
make run
````

### End result
After training is complete, the ``saved_fine_tuned_model.pb`` is a frozen version of the ``h5`` model and 
can be used by any TensorFlow for Java application.   
