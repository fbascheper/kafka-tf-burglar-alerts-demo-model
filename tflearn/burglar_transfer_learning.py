#
# Note: based on https://www.tensorflow.org/tutorials/images/transfer_learning
#

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from .util.helpers import plot_results, freeze_session, create_logger, log_tf_model_nodes


def train_model_using_transfer_learning():
    """
    Train a model for burglar alerts using transfer learning.
    """

    # Initialisation

    # this is the relative path where the images are stored
    base_dir = r'input-images/classified-and-converted-using-kafka-storage-converters'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    #
    # Prepare training and validation datasets
    #
    train_burglars_dir = os.path.join(train_dir, 'burglar-alert')
    train_no_burglars_dir = os.path.join(train_dir, 'no-burglar-alert')

    validation_burglars_dir = os.path.join(validation_dir, 'burglar-alert')
    validation_no_burglars_dir = os.path.join(validation_dir, 'no-burglar-alert')

    logger = create_logger()

    logger.info('Using TensorFlow {} and Keras {}'.format(tf.VERSION, tf.keras.__version__))
    logger.info('Total training / validation BURGLAR images = {} / {}'.
                format(len(os.listdir(train_burglars_dir)), len(os.listdir(validation_burglars_dir))))
    logger.info('Total training / validation NOT-BURGLAR images = {} / {}'.
                format(len(os.listdir(train_no_burglars_dir)), len(os.listdir(validation_no_burglars_dir))))

    #
    # Create Image Data Generator with Image Augmentation
    #
    image_width = 659
    image_height = 476

    image_size = 224

    batch_size = 32

    # DO NOT rescale all images by 1./255 and apply image augmentation
    # --> train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_datagen = ImageDataGenerator(rescale=None)
    validation_datagen = ImageDataGenerator(rescale=None)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # Source directory for the training images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # Source directory for the validation images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    label_map = validation_generator.class_indices
    label_map = dict((v, k) for k, v in label_map.items())

    for k, v in label_map.items():
        logger.info("Classification index {} => '{}'".format(k, v))

    #
    # Create the base model from the pre-trained convnets
    #
    IMG_SHAPE = (image_size, image_size, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    #
    # Feature extraction
    #

    # Freeze the convnet
    base_model.trainable = False
    # Let's take a look at the base model architecture
    base_model.summary()

    #
    # Add layers on top of base model
    #
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.1),
        Dense(len(label_map), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print('Number of trainable variables = {0:5d}'.format(len(model.trainable_variables)))

    #
    # Train the model
    #
    epochs = 25
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  workers=4,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  # callbacks=[cp_callback]
                                  )

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    model.summary()
    model.save(r'saved_model.h5')

    # Test fine tuning
    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    fine_tune_at = 150

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

    model.summary()

    history_fine = model.fit_generator(train_generator,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       workers=4,
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps)

    acc += history_fine.history['acc']
    val_acc += history_fine.history['val_acc']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plot_results(acc, loss, val_acc, val_loss, epochs)

    log_tf_model_nodes(logger, "Input", model.inputs)
    log_tf_model_nodes(logger, "Output", model.outputs)

    model.summary()
    model.save(r'saved_fine_tuned_model.h5')

    #
    # See https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    #
    # We need to freeze the session to create a model that's usable in TensorFlow for Java
    # Unfortunately there's a bug in the convert_variables_to_constants method in TF 1.13,
    # so we use the version from intel-analytics / analytics-zoo.
    #
    # See https://github.com/tensorflow/tensorflow/issues/25721
    #
    # Maybe you can try our version of convert_variables_to_constants
    # (See https://github.com/intel-analytics/analytics-zoo/blob/master/pyzoo/zoo/util/tf_graph_util.py#L226 ).
    # This issue should have been fixed there.

    frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])

    tf.train.write_graph(frozen_graph, './', 'saved_fine_tuned_model.pbtxt', as_text=True)
    tf.train.write_graph(frozen_graph, './', 'saved_fine_tuned_model.pb', as_text=False)
