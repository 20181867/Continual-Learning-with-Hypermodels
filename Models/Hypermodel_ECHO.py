import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
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
 
    return filters_TE, bias_TE, (weights_DL_1, weights_DL_2, weights_DL_3), (biases_DL_1, biases_DL_2, biases_DL_3)

"' The FULL model '"
def create(embedding_dim=100,
           n_chunks=10,
           hnet_hidden_dims=(100, 50,),
           inner_net_dims=(200, 250, None),
           l2reg=0,
           dropout_rate = 0.3,
           use_unique_task_embedding = False,
           convolution_layers=None, 
           depency_preservation_between_chunks = True,
           reservoir_size = 100, spectral_radius = 0.9, sparsity=0.1,
           final_soft_max_layer = False,
           initialize_TE_with_zero_bias = True):

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
    
    "' The Task embedding model '"
    def task_embedding_model(inputs, filters_TE, bias_TE, filters_dl_TE, bias_dl_TE, strides=(1, 1), padding='VALID'):

        conv_output = tf.nn.conv2d(inputs, filters_TE, strides=[1, strides[0], strides[1], 1], padding=padding)
        conv_output = tf.nn.bias_add(conv_output, bias_TE)
        conv_output = tf.nn.relu(conv_output)
        
        #flatten the output
        total_elements = tf.reduce_prod(tf.shape(conv_output))
        output = tf.reshape(conv_output, [1, total_elements])

        #dense layers, including output layer
        for i in range (0,3):
            output = tf.matmul(output, filters_dl_TE[i])
            output = tf.add(output, bias_dl_TE[i])

        return tf.squeeze(output)

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
    
    n_classes = inner_net_dims[-1]

    class HNet(tf.keras.Model):
        
        def __init__(self, reservoir_size, spectral_radius, depency_preservation_between_chunks, sparsity,
                     hnet_hidden_dims, **kwargs):
            super(HNet, self).__init__(**kwargs)
            self.final_soft_max_layer = final_soft_max_layer
            self.chunk_tokens = self.add_weight(shape=[n_chunks, embedding_dim],
                                                trainable=True,
                                                name='chunk_embeddings',
                                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
            
            if use_unique_task_embedding:
                self.filters_TE, self.bias_TE, self.weights_DL_TE, self.biases_DL_TE = initialize_TE_model(embedding_dim, initialize_with_zero_bias = initialize_TE_with_zero_bias)
            

            self.n_classes = n_classes
            self.weight_num_cv, self.weight_num_dense, self.weight_sizes_dense, self.weight_shapes_dense,self.weight_sizes_cv, self.weight_shapes_cv, self.weight_num= calculate_sizes(inner_net_dims, n_classes = self.n_classes)
            self.n_chunks = n_chunks

            "' The Hyper model '"
            self.reservoir_size = reservoir_size
            self.spectral_radius = spectral_radius
            self.sparsity = sparsity
            self.depency_preservation_between_chunks = depency_preservation_between_chunks

            #add one layer as input layer to the reservoir
            self.hnet_hidden_dims = hnet_hidden_dims
            try:
                hnet_hidden_dims = tuple(list(hnet_hidden_dims) + [reservoir_size])
            except TypeError:
                hnet_hidden_dims = (hnet_hidden_dims, reservoir_size)

            HNET_layers = [tf.keras.layers.InputLayer(input_shape=embedding_dim * 2)]
            try:
                HNET_layers += [tf.keras.layers.Dense(neurons, activation='relu')
                        for neurons in hnet_hidden_dims]
            except TypeError:
                HNET_layers += [tf.keras.layers.Dense(hnet_hidden_dims, activation='relu')]
            
            self.first_HNET_part = tf.keras.Sequential(HNET_layers)

            self.reservoir_weights = self.initialize_reservoir_weights()

            self.chunk_size = int(np.ceil(self.weight_num / n_chunks))

            self.output_weights = np.random.rand(reservoir_size, self.chunk_size)

            if not self.depency_preservation_between_chunks:
                #goal: dependency preservation among multiple runs
                self.reservoir_states = tf.zeros(shape=(n_chunks, self.reservoir_size))
            
            #The last layer can unfortunately not be trainable, because gradient computation will not work then
            else:
                self.resize_layer = tf.keras.layers.Dense(units=self.n_chunks*self.reservoir_size,  activation='relu', trainable = False,
                                                          bias_initializer= "zeros")    
                self.resize_layer.build((1, self.reservoir_size))

            "' END Hyper model '"

        def initialize_reservoir_weights(self):
            # Create a random mask with sparsity
            mask = np.random.choice([0, 1], size=(self.reservoir_size, self.reservoir_size),
                                    p=[self.sparsity, 1 - self.sparsity]).astype(float)
            
            # Scale the mask to maintain spectral radius
            mask *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(mask)))
            
            return mask

        def call(self, inputs, **kwargs):
            task_token, input_data, added_classes = inputs

            #Class-incremental case: add an additional neuron to the final layer of the target network and update the hypernetwork accordingly
            if added_classes:
                self.n_classes = self.n_classes+1
                old_weight_num = self.weight_num

                #recalculate the sizes after adding a neuron to the final layer of the target network
                self.weight_num_cv, self.weight_num_dense, self.weight_sizes_dense, self.weight_shapes_dense,self.weight_sizes_cv, self.weight_shapes_cv, self.weight_num= calculate_sizes(inner_net_dims, n_classes = self.n_classes)
                self.weight_num = self.weight_num.numpy()

                extension = np.random.randn(self.reservoir_size, self.weight_num - old_weight_num)
                self.output_weights = tf.concat([self.output_weights, tf.constant(extension, dtype=tf.float32)], axis=1)

            #task_token is a randomized task embedding. This needs to change if use_unique_task_embedding is true.
            if use_unique_task_embedding:
                task_token= task_embedding_model(input_data, self.filters_TE, self.bias_TE, self.weights_DL_TE, self.biases_DL_TE)
                task_token = task_token/100000 #scale the output of the task embedding model to prevent extreme weights

            task_token = tf.reshape(task_token, [1, embedding_dim])
            task_token = tf.repeat(task_token, n_chunks, axis=0)
            full_token = tf.concat([self.chunk_tokens, task_token], axis=1)

            "' The Hyper model '"

            # Assuming inputs is the full token, with shape (n_chunks, embedding_dim*2)
            if self.depency_preservation_between_chunks:
                #goal: dependency preservation only in this run
                batch_size = 1
                num_timesteps = tf.shape(full_token)[0]
            
                # Clean reservoir states after each 'batch' or in this case image
                reservoir_states = tf.zeros(shape=(batch_size, self.reservoir_size))

                # Iterate over timesteps
                for t in range(num_timesteps):
                    #get exactly 1 task & chunk embedding with shape (1, embedding_dim*2)
                    layer_input =  tf.transpose(tf.expand_dims(full_token[t], axis=1))

                    #process this in through the first dense layers
                    layer_input = self.first_HNET_part(layer_input)
                  
                    # Update reservoir states by summing the reservoir states across time to get a single representation.
                    reservoir_states = tf.keras.activations.elu(tf.matmul(layer_input, self.reservoir_weights) 
                                           + reservoir_states)
                    
                # increase size again
                output= tf.reshape(self.resize_layer(reservoir_states), (self.n_chunks, self.reservoir_size))
                
                output = tf.tanh(tf.matmul(output, self.output_weights))

            else:
                #goal: dependency preservation among multiple runs
                batch_size = n_chunks
                num_timesteps = 1

                # Do NOT clean reservoir states after each image
                reservoir_states = self.reservoir_states
                input_echo = self.first_HNET_part(full_token)

                #output has shape (n_chunks, reservoir_size)
                self.reservoir_states = tf.keras.activations.elu(tf.matmul(input_echo, self.reservoir_weights) 
                                           + reservoir_states)

                output = tf.tanh(tf.matmul(self.reservoir_states, self.output_weights))

            "' END Hyper model '"

            net_weights = output 

            net_weights_flat_cv = tf.reshape(net_weights, (-1,))[:self.weight_num_cv]
            net_weight_flat_dense = tf.reshape(net_weights, (-1,))[self.weight_num_cv:self.weight_num_dense+ self.weight_num_cv]

            net_weight_dense = tf.split(net_weight_flat_dense, self.weight_sizes_dense)
            net_weight_dense = [tf.reshape(w, shape) for w, shape in zip(net_weight_dense, self.weight_shapes_dense)]

            net_weight_cv = tf.split(net_weights_flat_cv, self.weight_sizes_cv)
            net_weight_cv = [tf.reshape(w, shape) for w, shape in zip(net_weight_cv, self.weight_shapes_cv)]
            
            output = inner_net(input_data, net_weight_cv, net_weight_dense, n_chunks, dropout_rate,  final_soft_max_layer =  self.final_soft_max_layer)
            return output, tf.reshape(net_weights, (-1,))
    
    full_model = HNet(reservoir_size, spectral_radius, depency_preservation_between_chunks, sparsity,
                     hnet_hidden_dims)
    return full_model

if __name__ == "__main__":
    embedding_dim = 100
    num_classes = 3
    inner_net_dims = (200, 300, num_classes)
    depency_preservation_between_chunks = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    final_soft_max_layer = False
    initialize_TE_with_zero_bias = True

    #using HPO:
    convolution_layers = [(8, (5, 5), 1), (16, (3, 3), 8), (32, (3, 3), 16)]
    
    #use either one task embedding for all tasks or train a seperate model (tf.variable vs multiple layers)
    use_unique_task_embedding = False
    task_embedding = tf.random.normal([embedding_dim]) / 10
    task_embedding = tf.Variable(task_embedding, trainable=True)
    
    if use_unique_task_embedding:
        model = create(use_unique_task_embedding = use_unique_task_embedding, 
                       convolution_layers=convolution_layers,
                       inner_net_dims = (200, 300, num_classes),
                       depency_preservation_between_chunks = depency_preservation_between_chunks,
                       reservoir_size= 10,
                       spectral_radius= 0.9,
                       sparsity= 0.01,
                       final_soft_max_layer = final_soft_max_layer,
                       initialize_TE_with_zero_bias = initialize_TE_with_zero_bias)

    else:
        model = create(use_unique_task_embedding = use_unique_task_embedding, inner_net_dims = (200, 300, num_classes), convolution_layers=convolution_layers,
                       depency_preservation_between_chunks = depency_preservation_between_chunks, reservoir_size= 100,
                       spectral_radius= 0.9, sparsity= 0.01,
                       final_soft_max_layer = final_soft_max_layer,
                       initialize_TE_with_zero_bias = initialize_TE_with_zero_bias)
    
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    y= np.random.randint(0, 3)

    #with tf.device('/device:GPU:1'): preferable

    with tf.GradientTape() as tape:
        outs, _ = model([task_embedding, np.random.rand(1, 128, 32, 1), True])
        loss = loss_fun(y, outs)
        

    if use_unique_task_embedding:
        trainables = model.trainable_variables
    else:
        trainables = model.trainable_variables + [task_embedding]

    if not final_soft_max_layer:
        outs = tf.nn.softmax(outs)

    optimizer.build(trainables)
    grads = tape.gradient(loss, trainables)
    print(grads)
    print(outs)
    #optimizer.apply_gradients(zip(grads, trainables))








