from keras.layers import Lambda, concatenate
from keras import Model
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf

def asign_gpu_model(model, noGpu): 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config)) 

#    if isinstance(gpus, (list, tuple)):
#        num_gpus = len(gpus)
#        target_gpu_ids = gpus
#    else:
#        num_gpus = gpus
#        target_gpu_ids = range(num_gpus)

    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    with tf.device('/gpu:%d' % noGpu):
        with tf.name_scope('replica_%d' % noGpu):
            inputs = []
            # Retrieve a slice of the input.
            for x in model.inputs:
                input_shape = tuple(x.get_shape().as_list())[1:]
                slice_i = Lambda(get_slice,
                                 output_shape=input_shape,
                                 arguments={'i': i,
                                            'parts': num_gpus})(x)
                inputs.append(slice_i)

            # Apply model on slice
            # (creating a model replica on the target device).
            outputs = model(inputs)
            if not isinstance(outputs, list):
                outputs = [outputs]

            # Save the outputs for merging back together later.
            for o in range(len(outputs)):
                all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for name, outputs in zip(model.output_names, all_outputs):
            merged.append(concatenate(outputs,
                                axis=0, name=name))
        return Model(model.inputs, merged)
