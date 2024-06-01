import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam

tf.config.run_functions_eagerly(True)
tf.debugging.set_log_device_placement(False)

def calculate_amount_of_weights(target_model):
    weight_num = 0
    for i in range(len(target_model.model.layers)):
        if 'conv' in target_model.model.layers[i].name or 'out_layer' in target_model.model.layers[i].name or 'dense' in target_model.model.layers[i].name: 
            weights_shape = target_model.model.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            weight_num += no_of_weights
            if target_model.model.layers[i].use_bias:
                weights_shape = target_model.model.layers[i].bias.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                weight_num+=no_of_weights

    return weight_num

"' The optional Task Embedding model '"
def initialize_TE_model(embedding_dim, initialize_with_zero_bias= True):

    #PARAMETERIZE

    input_shape = (1,128,32,1)
    #The task embedding model is a simple nn consisting of 1 2DCONV layer and 2 dense layers
    kernel_size_TE = (5,5)
    num_filters_TE = 8

    #assuming padding is VALID and stride is (1,1,1,1)
    output_size = (input_shape[0], int(((input_shape[1] - kernel_size_TE[0] + 2 * 0) / 1)) + 1, int(((input_shape[2] - kernel_size_TE[1] + 2 * 0) / 1) + 1), num_filters_TE)
    output_flattened = tf.reduce_prod(output_size).numpy()
    
    #dense layers
    dense_layer = [1000, 500]

    # Define the shape of the filter weights [height, width, in_channels, out_channels]
    filter_shape_TE = [kernel_size_TE[0], kernel_size_TE[1], input_shape[-1], num_filters_TE]

    # Initialize filter weights randomly
    filters_TE = tf.Variable(tf.random.truncated_normal(filter_shape_TE, stddev=1.0), trainable=True)
   
    #Do the same for the dense layers
    weights_DL_1 = tf.Variable(tf.random.truncated_normal([output_flattened, dense_layer[0]], stddev=1.0), trainable=True)
    weights_DL_2 = tf.Variable(tf.random.truncated_normal([dense_layer[0], dense_layer[1]], stddev=1.0), trainable=True)
    
    #output layer
    weights_DL_3 = tf.Variable(tf.random.truncated_normal([dense_layer[1], embedding_dim], stddev=1.0), trainable=True)

    if initialize_with_zero_bias:
        bias_TE = tf.Variable(tf.zeros([num_filters_TE]), trainable=True)
        biases_DL_1 = tf.Variable(tf.zeros([dense_layer[0]]), trainable=True)
        biases_DL_2 = tf.Variable(tf.zeros([dense_layer[1]]), trainable=True)
        biases_DL_3 = tf.Variable(tf.zeros([embedding_dim]), trainable=True)
    else:
        bias_TE = tf.Variable(tf.random.truncated_normal([num_filters_TE], stddev=1.0), trainable=True)
        biases_DL_1 = tf.Variable(tf.random.truncated_normal([dense_layer[0]], stddev=1.0), trainable=True)
        biases_DL_2 = tf.Variable(tf.random.truncated_normal([dense_layer[1]], stddev=1.0), trainable=True)
        biases_DL_3 = tf.Variable(tf.random.truncated_normal([embedding_dim], stddev=1.0), trainable=True)
 
    return filters_TE, bias_TE, weights_DL_1, weights_DL_2, weights_DL_3, biases_DL_1, biases_DL_2, biases_DL_3

#----------------just for visualization purposes:-------------------------
def summary_task_embedding_model(embedding_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(128, 32, 1)))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='valid'))
    model.add(tf.keras.layers.Flatten())

    # Add dense layers
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    model.add(tf.keras.layers.Dense(embedding_dim))
    model.summary()
    return model
    
"' The FULL model '"
def create(embedding_dim=100,
           n_chunks=500,
           hnet_hidden_dims=(200, 250),
           inner_net_dims=(200, 250, None),
           l2reg=0,
           dropout_rate = 0.3,
           use_unique_task_embedding = False,
           convolution_layers=None,
           initialize_with_zero_bias = False,
           final_soft_max_layer = False):
    
    def calculate_output_shape_CV_LAYERS(input_shape, convolution_layers, X):
        # Define input shape (ignoring batch size)
        input_height, input_width, input_channels = input_shape[1:]

        # Initialize variables to store output shape after each convolutional layer
        output_height, output_width = input_height, input_width

        # Track whether the first convolutional layer has been encountered
        first_conv_layer = True

        # Iterate through convolutional layers
        for filters, kernel_size, _ in convolution_layers:
            # Calculate output shape after convolution
            output_height = np.floor((output_height - kernel_size[0] + 2 * 0) / 1) + 1
            output_width = np.floor((output_width - kernel_size[1] + 2 * 0) / 1) + 1

            # Check if this is the first convolutional layer
            if first_conv_layer:
                # Apply max pooling after the first convolutional layer
                output_height = np.floor((output_height - 2) / 2) + 1
                output_width = np.floor((output_width - 2) / 2) + 1
                first_conv_layer = False

        # Calculate the total number of features after the last convolutional layer
        num_features = int(output_height * output_width * convolution_layers[-1][0])


        # Calculate the output shape after flattening
        output_shape = (X, num_features)


        return output_shape
    
    
    def calculate_sizes(inner_net_dims, n_classes = 0):
        input_shape = (1,128,32,1) #the shape of 1 melspectogram

        capitol_layer_size = calculate_output_shape_CV_LAYERS(input_shape, convolution_layers, n_chunks)

        inner_net_dims = (capitol_layer_size[1],) + inner_net_dims[:-1] + (n_classes,)

        kernel_shapes_dense_layers = [[x, y] for x, y in zip(inner_net_dims, inner_net_dims[1:])]
        bias_shapes_dense_layers = [[y, ] for x, y in zip(inner_net_dims, inner_net_dims[1:])]

        kernel_shapes_cv = [[kb[1][0], kb[1][1],kb[2], kb[0]] for kb in convolution_layers]
        bias_shapes_cv = [[kb[0]] for kb in convolution_layers]

        weight_shapes_dense = [[k, b] for k, b in zip(kernel_shapes_dense_layers, bias_shapes_dense_layers)]
        weight_shapes_cv = [[k, b] for k, b in zip(kernel_shapes_cv, bias_shapes_cv)]

        weight_shapes_dense = [x for w in weight_shapes_dense for x in w]

        weight_shapes_cv = [x for w in weight_shapes_cv for x in w]

        weight_sizes_dense = [tf.reduce_prod(w) for w in weight_shapes_dense]
        weight_sizes_cv = [tf.reduce_prod(w) for w in weight_shapes_cv]
        weight_num_cv = sum(weight_sizes_cv)
        weight_num_dense= sum(weight_sizes_dense)


        weight_num = sum(weight_sizes_dense) + sum(weight_sizes_cv)

        return weight_num_cv, weight_num_dense, weight_sizes_dense, weight_shapes_dense,weight_sizes_cv, weight_shapes_cv, weight_num

    "' The Target model '"
    def dense_layer(x, w, b, activation_func):
        return activation_func(tf.matmul(x, w) + b)
    
    def softmax_layer(x, w, b):
        return tf.nn.softmax(tf.squeeze(tf.matmul(x, w) + b))
    
    def inner_net(inputs, weights_biases, net_weight_dense, chunks, dropout_rate, final_soft_max_layer, strides=(1, 1), padding='VALID'):
        output = inputs
        for i in range(0, len(weights_biases), 2):
            weight = weights_biases[i]
            bias = weights_biases[i + 1]
            output = tf.nn.conv2d(output, weight, strides=[1, strides[0], strides[1], 1], padding=padding)
            output = tf.nn.bias_add(output, bias)
            output = tf.nn.relu(output)  # Using ReLU activation function
            if i == 0:
                output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            #DropoutLayers, in total 4
            output = tf.nn.dropout(output, rate=dropout_rate)
        
        #flatten output
        output = tf.reshape(output, (1, -1))

        if not final_soft_max_layer:
            for w, b in zip(net_weight_dense[::2], net_weight_dense[1::2]):
                output = dense_layer(output, w, b, activation_func=tf.nn.relu)
            return tf.squeeze(output)
        
        for i, (w, b) in enumerate(zip(net_weight_dense[::2], net_weight_dense[1::2])):
            if i == len(net_weight_dense) // 2 - 1: #net_weight_dense will be twice the number of dense layers (since we have weights and biases for each layer)
                #the last layer
                output = softmax_layer(output, w, b)
            else:
                output = dense_layer(output, w, b, activation_func=tf.nn.relu)

        return output
        
    "' The Hyper model '"
    n_classes = inner_net_dims[-1]
    weight_num_cv, weight_num_dense, weight_sizes_dense, weight_shapes_dense,weight_sizes_cv, weight_shapes_cv, weight_num= calculate_sizes(inner_net_dims, n_classes = n_classes)

    chunk_size = int(np.ceil(weight_num / n_chunks))

    layers = [tf.keras.layers.InputLayer(input_shape=embedding_dim * 2)]
    try:
        layers += [tf.keras.layers.Dense(neurons, activation='relu')
                for neurons in hnet_hidden_dims]
    except TypeError:
        layers += [tf.keras.layers.Dense(hnet_hidden_dims, activation='relu')]
    layers += [tf.keras.layers.Dense(chunk_size, activation='tanh',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))]
    
    hnet = tf.keras.Sequential(layers)
    hnet.build([1, embedding_dim * 2])
    
    class HNet(tf.keras.Model):
        
        def __init__(self, hnet, **kwargs):
            super(HNet, self).__init__(**kwargs)
            self.hnet = hnet
            self.final_soft_max_layer = final_soft_max_layer

            if use_unique_task_embedding:
                self.filters_TE, self.bias_TE, self.weights_DL_1, self.weights_DL_2, self.weights_DL_3, self.biases_DL_1, self.biases_DL_2, self.biases_DL_3 = initialize_TE_model(embedding_dim, initialize_with_zero_bias = initialize_with_zero_bias)
            
            self.n_classes = n_classes
            self.weight_num_cv, self.weight_num_dense, self.weight_sizes_dense, self.weight_shapes_dense,self.weight_sizes_cv, self.weight_shapes_cv, self.weight_num= weight_num_cv, weight_num_dense, weight_sizes_dense, weight_shapes_dense,weight_sizes_cv, weight_shapes_cv, weight_num
            self.chunk_tokens = self.add_weight(shape=[n_chunks, embedding_dim],
                                                trainable=True,
                                                name='chunk_embeddings',
                                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
                             

        def task_embedding_model(self, inputs, strides=(1, 1), padding='VALID'):
            
            conv_output = tf.nn.conv2d(inputs, self.filters_TE, strides=[1, strides[0], strides[1], 1], padding=padding)
            conv_output = tf.nn.bias_add(conv_output, self.bias_TE)
            conv_output = tf.nn.relu(conv_output)
            
            #flatten the output
            total_elements = tf.reduce_prod(tf.shape(conv_output))
            output = tf.reshape(conv_output, [1, total_elements])

            #dense layers, including output layer
            output = tf.matmul(output, self.weights_DL_1)
            output = tf.add(output, self.biases_DL_1)
            output = tf.matmul(output, self.weights_DL_2)
            output = tf.add(output, self.biases_DL_2)
            output = tf.matmul(output,self.weights_DL_3)
            output = tf.add(output, self.biases_DL_3)

            return tf.squeeze(output)
    
        def call(self, inputs, **kwargs):
            task_token, input_data, added_classes = inputs

            #self.hnet.summary()

            #Class-incremental case: add an additional neuron to the final layer of the target network and update the hypernetwork accordingly
            if added_classes:
                self.n_classes = self.n_classes+1

                input_size_last_layer =  self.hnet.layers[-1].kernel.shape[0]
                last_layer_weights, last_layer_biases = self.hnet.layers[-1].get_weights()
                old_chunk_size = int(np.ceil(self.weight_num / n_chunks)) 

                #recalculate the sizes after adding a neuron to the final layer of the target network
                self.weight_num_cv, self.weight_num_dense, self.weight_sizes_dense, self.weight_shapes_dense,self.weight_sizes_cv, self.weight_shapes_cv, self.weight_num= calculate_sizes(inner_net_dims, n_classes = self.n_classes)
                new_chunk_size = int(np.ceil(self.weight_num / n_chunks)) 

                #remove the last layer, by redefining what self.hnet is
                new_layers = self.hnet.layers[:-1]
                self.hnet = tf.keras.Sequential(new_layers)
                self.hnet.build([1, embedding_dim * 2])

                new_dense_layer = tf.keras.layers.Dense(new_chunk_size, activation='tanh',
                                             kernel_regularizer=tf.keras.regularizers.l2(l2reg))
                
                new_weights = np.concatenate([last_layer_weights, np.random.randn(input_size_last_layer, 
                                                                                  new_chunk_size - old_chunk_size)], axis=1)
                
                new_biases = np.concatenate([last_layer_biases, np.random.randn(new_chunk_size - old_chunk_size)], axis=0)
                new_biases = np.reshape(new_biases, (1, new_chunk_size))

                # Add the new last dense layer to the model
                self.hnet.add(new_dense_layer)
                self.hnet.layers[-1].set_weights([new_weights, np.squeeze(new_biases)])

            #task_token is a randomized task embedding. This needs to change if use_unique_task_embedding is true.
            if use_unique_task_embedding:
                task_token= self.task_embedding_model(input_data)
                task_token = task_token/100000 #scale the output of the task embedding model to prevent extreme weights

            task_token = tf.reshape(task_token, [1, embedding_dim])
            task_token = tf.repeat(task_token, n_chunks, axis=0)
            full_token = tf.concat([self.chunk_tokens, task_token], axis=1)

            net_weights = self.hnet(full_token)

            net_weights_flat_cv = tf.reshape(net_weights, (-1,))[:self.weight_num_cv]
            net_weight_flat_dense = tf.reshape(net_weights, (-1,))[self.weight_num_cv:self.weight_num_dense+ self.weight_num_cv]

            net_weight_dense = tf.split(net_weight_flat_dense, self.weight_sizes_dense)
            net_weight_dense = [tf.reshape(w, shape) for w, shape in zip(net_weight_dense, self.weight_shapes_dense)]

            net_weight_cv = tf.split(net_weights_flat_cv, self.weight_sizes_cv)
            net_weight_cv = [tf.reshape(w, shape) for w, shape in zip(net_weight_cv, self.weight_shapes_cv)]
            
            output = inner_net(input_data, net_weight_cv, net_weight_dense, n_chunks, dropout_rate, self.final_soft_max_layer)
            
            
            return output, tf.reshape(net_weights, (-1,))
        
    
    full_model = HNet(hnet)
    return full_model

if __name__ == "__main__":
    embedding_dim = 100
    num_classes = 3
    inner_net_dims = (416, 832, num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    l2reg = 0
    initialize_TE_with_zero_bias = True
    final_soft_max_layer = False

    #using HPO:
    convolution_layers = [(16, (4,4), 1), (64, (5,5), 16), (64, (3,3), 64), (32, (3,3), 64)]
    
    #use either one task embedding for all tasks or train a seperate model (tf.variable vs multiple layers)
    use_unique_task_embedding = True
    task_embedding = tf.random.normal([embedding_dim]) / 10
    task_embedding = tf.Variable(task_embedding, trainable=True)
    
    if use_unique_task_embedding:
        model = create(use_unique_task_embedding = use_unique_task_embedding, 
                       convolution_layers=convolution_layers,
                       inner_net_dims = (200, 300, num_classes),
                       l2reg = l2reg,
                       initialize_with_zero_bias = initialize_TE_with_zero_bias,
                       final_soft_max_layer = final_soft_max_layer)

    else:
        model = create(use_unique_task_embedding = use_unique_task_embedding, inner_net_dims = (200, 300, num_classes), convolution_layers=convolution_layers, l2reg = l2reg,
        initialize_with_zero_bias = initialize_TE_with_zero_bias,
        final_soft_max_layer = final_soft_max_layer)
    
    #with tf.device('/device:GPU:1'): preferable

    with tf.GradientTape() as tape:
        outs, _ = model([task_embedding, np.random.rand(1, 128, 32, 1), True])
        loss = tf.reduce_sum(outs)

    if use_unique_task_embedding:
        trainables = model.trainable_variables
        #trainables = trainables[-6:-1]+trainables[:-6]+ [trainables[-1]]
    else:
        trainables = model.trainable_variables + [task_embedding]

    if not final_soft_max_layer:
        outs = tf.nn.softmax(outs)

    optimizer.build(trainables)

    grads = tape.gradient(loss, trainables)
    print(grads)
    print(outs)
    optimizer.apply_gradients(zip(grads, trainables))




#To Do:

#remove soft max from other models DONE THAT
#update initialize_TE_model in other models DONE THAT
#add soft max in training 
#add: first x-amount of training itterations no class incremental learning DONE THAT

