import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU

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
def create_GRU(embedding_dim=100,
           n_chunks=10,
           hnet_hidden_dims=(10, 5,),
           inner_net_dims=(200, 250, None),
           l2reg=0,
           dropout_rate = 0.3,
           use_unique_task_embedding = False,
           convolution_layers=None, 
           depency_preservation_between_chunks = True,
           final_soft_max_layer = False,
           initialize_TE_with_zero_bias = True,
           final_GRU_layer = False):

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
    
    "' The Hyper model '"

    n_classes = inner_net_dims[-1]
    weight_num_cv, weight_num_dense, weight_sizes_dense, weight_shapes_dense,weight_sizes_cv, weight_shapes_cv, weight_num= calculate_sizes(inner_net_dims, n_classes = n_classes)

    chunk_size = int(np.ceil(weight_num / n_chunks))
    weight_num = weight_num.numpy()

    # Define the hypernetwork using GRU layers: input = (batch_size, timesteps, features)
    layers = [tf.keras.layers.InputLayer(input_shape=(None, embedding_dim * 2))]

    try:
        for neurons in hnet_hidden_dims:
            layers.append(GRU(neurons, return_sequences=True))  # Return sequences to pass output to the next GRU layer
    except:
        layers.append(GRU(hnet_hidden_dims, return_sequences=True))


    if not depency_preservation_between_chunks:
        # Add the final GRU layer with chunk_size output units. Unfortunately, there is not enough memory to add a final GRU layer...
        if not final_GRU_layer:
            layers.append(tf.keras.layers.Dense(chunk_size, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg)))
        else:
            layers.append(GRU(chunk_size, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg)))

    else:
        # If dependency must be preserved within chunks, only one batch (of all timestemps) must generate all weights
        if not final_GRU_layer:
            layers.append(tf.keras.layers.Dense(weight_num, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg)))
        else:
            layers.append(GRU(weight_num, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg)))


    hnet = tf.keras.Sequential(layers)
    hnet.build((None, embedding_dim * 2))  # Use None for variable sequence length


    def random_pad(array, pad_width, constant_values=None, parameters=None):
        return np.random.randn(*pad_width)

    class HNet(tf.keras.Model):
        
        def __init__(self, hnet, **kwargs):
            super(HNet, self).__init__(**kwargs)
            self.hnet = hnet
            self.n_classes = n_classes
            self.final_soft_max_layer = final_soft_max_layer
            self.weight_num_cv, self.weight_num_dense, self.weight_sizes_dense, self.weight_shapes_dense,self.weight_sizes_cv, self.weight_shapes_cv, self.weight_num= weight_num_cv, weight_num_dense, weight_sizes_dense, weight_shapes_dense,weight_sizes_cv, weight_shapes_cv, weight_num
            self.chunk_tokens = self.add_weight(shape=[n_chunks, embedding_dim],
                                                trainable=True,
                                                name='chunk_embeddings',
                                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
            
            if use_unique_task_embedding:
                self.filters_TE, self.bias_TE, self.weights_DL_TE, self.biases_DL_TE = initialize_TE_model(embedding_dim, initialize_with_zero_bias = initialize_TE_with_zero_bias)                            


        def call(self, inputs, **kwargs):
            task_token, input_data, added_classes = inputs

            #Class-incremental case: add an additional neuron to the final layer of the target network and update the hypernetwork accordingly
            if added_classes:
                print('-'*15 + 'INCREASING LAST LAYER SIZE' + '-' *15)
                self.n_classes = self.n_classes+1

                try:
                    input_size_last_layer =  hnet_hidden_dims[-1]
                except TypeError:
                    input_size_last_layer =  hnet_hidden_dims
                
                if final_GRU_layer:
                    last_layer_weights, last_layer_recurrent_kernel, last_layer_biases = self.hnet.layers[-1].get_weights()
                else:
                    last_layer_weights, last_layer_biases = self.hnet.layers[-1].get_weights()
                    
                old_chunk_size = int(np.ceil(self.weight_num / n_chunks))
                old_weight_num = self.weight_num

                #recalculate the sizes after adding a neuron to the final layer of the target network
                self.weight_num_cv, self.weight_num_dense, self.weight_sizes_dense, self.weight_shapes_dense,self.weight_sizes_cv, self.weight_shapes_cv, self.weight_num= calculate_sizes(inner_net_dims, n_classes = self.n_classes)
                self.weight_num = self.weight_num.numpy()
                new_chunk_size = int(np.ceil(self.weight_num / n_chunks)) 

                #remove the last layer, by redefining what self.hnet is
                new_layers = self.hnet.layers[:-1]
                self.hnet = tf.keras.Sequential(new_layers)
                self.hnet.build((None,1, embedding_dim * 2)) 

                if not depency_preservation_between_chunks:
                    if final_GRU_layer:
                        # Add the final GRU layer with chunk_size output units
                        new_dense_layer = GRU(new_chunk_size, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg))
                        layers.append(tf.keras.layers.Dense(weight_num, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg)))

                        new_weights = np.concatenate([last_layer_weights, np.random.randn(input_size_last_layer, 
                                                                                    3*new_chunk_size-3*old_chunk_size)], axis=1)
                    

                        new_recurrent_kernel = np.pad(last_layer_recurrent_kernel, [(0, new_chunk_size - old_chunk_size), 
                                                (0, 3*new_chunk_size-3*old_chunk_size)],mode=random_pad)
                        
                        #Each GRU unit has two biases: The reset gate bias and the update gate bias.
                        new_biases = np.concatenate([last_layer_biases, np.random.randn(2, 3*new_chunk_size-3*old_chunk_size)], axis=1)

                        # Add the new last dense layer to the model
                        self.hnet.add(new_dense_layer)
                        self.hnet.layers[-1].set_weights([new_weights, new_recurrent_kernel, np.squeeze(new_biases)])
                    else:
                        new_dense_layer = tf.keras.layers.Dense(new_chunk_size, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg))
                        
                        new_weights = np.concatenate([last_layer_weights, np.random.randn(input_size_last_layer, 
                                                                                        new_chunk_size - old_chunk_size)], axis=1)
                        
                        new_biases = np.concatenate([last_layer_biases, np.random.randn(new_chunk_size - old_chunk_size)], axis=0)
                        new_biases = np.reshape(new_biases, (1, new_chunk_size))

                        # Add the new last dense layer to the model
                        self.hnet.add(new_dense_layer)
                        self.hnet.layers[-1].set_weights([new_weights, np.squeeze(new_biases)])

                else:
                    if final_GRU_layer:
                        # If dependency must be preserved within chunks, only one batch (of all timestemps) must generate all weights
                        new_dense_layer = GRU(self.weight_num, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg))

                        #note that 3 times self.weights because GRY has 3 sets of weights
                        new_weights = np.concatenate([last_layer_weights, np.random.randn(input_size_last_layer, 
                                                                                    3*self.weight_num-3*old_weight_num)], axis=1)

                        # Pad the recurrent_kernel array with zeros to match the new final size
                        new_recurrent_kernel = np.pad(last_layer_recurrent_kernel, [(0, self.weight_num - old_weight_num), 
                                                (0, 3*self.weight_num-3*old_weight_num)],mode=random_pad)

                        #Each GRU unit has two biases: The reset gate bias and the update gate bias.
                        new_biases = np.concatenate([last_layer_biases, np.random.randn(2, 3*self.weight_num-3*old_weight_num)], axis=1)

                        # Add the new last dense layer to the model
                        self.hnet.add(new_dense_layer)
                        self.hnet.layers[-1].set_weights([new_weights, new_recurrent_kernel, np.squeeze(new_biases)])
                    else:
                        new_dense_layer = tf.keras.layers.Dense(self.weight_num, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2reg))
                        
                        new_weights = np.concatenate([last_layer_weights, np.random.randn(input_size_last_layer, 
                                                                                        self.weight_num - old_weight_num)], axis=1)
                        
                        new_biases = np.concatenate([last_layer_biases, np.random.randn(self.weight_num - old_weight_num)], axis=0)

                        # Add the new last dense layer to the model
                        self.hnet.add(new_dense_layer)
                        self.hnet.layers[-1].set_weights([new_weights, np.squeeze(new_biases)])

            #task_token is a randomized task embedding. This needs to change if use_unique_task_embedding is true.
            if use_unique_task_embedding:
                task_token= task_embedding_model(input_data, self.filters_TE, self.bias_TE, self.weights_DL_TE, self.biases_DL_TE)
                task_token = task_token/100000 #scale the output of the task embedding model to prevent extreme weights

            task_token = tf.reshape(task_token, [1, embedding_dim])
            task_token = tf.repeat(task_token, n_chunks, axis=0)
            full_token = tf.concat([self.chunk_tokens, task_token], axis=1)

            if not depency_preservation_between_chunks:
                net_weights = self.hnet(tf.reshape(full_token, [full_token.shape[0],1, full_token.shape[1]]))
            else:
                #depenency preservation between chunks
                net_weights = []
                batch_size = 1
                timesteps = full_token.shape[0]
                features = full_token.shape[1]
                full_token = tf.reshape(full_token, shape=(batch_size, timesteps, features))
                net_weights = self.hnet(full_token)
            
            net_weights_flat_cv = tf.reshape(net_weights, (-1,))[:self.weight_num_cv]
            net_weight_flat_dense = tf.reshape(net_weights, (-1,))[self.weight_num_cv:self.weight_num_dense+ self.weight_num_cv]

            net_weight_dense = tf.split(net_weight_flat_dense, self.weight_sizes_dense)
            net_weight_dense = [tf.reshape(w, shape) for w, shape in zip(net_weight_dense, self.weight_shapes_dense)]

            net_weight_cv = tf.split(net_weights_flat_cv, self.weight_sizes_cv)
            net_weight_cv = [tf.reshape(w, shape) for w, shape in zip(net_weight_cv, self.weight_shapes_cv)]
            
            output = inner_net(input_data, net_weight_cv, net_weight_dense, n_chunks, dropout_rate,  final_soft_max_layer =  self.final_soft_max_layer)
            return output, tf.reshape(net_weights, (-1,))
    
    full_model = HNet(hnet)
    return full_model

if __name__ == "__main__":
    optimizer = tf.keras.optimizers.Adam()
    embedding_dim = 5
    num_classes = 3
    inner_net_dims = (2, num_classes)
    n_chunks = 4
    depency_preservation_between_chunks = False
    hnet_hidden_dims=(10, 5)
    final_soft_max_layer = True
    initialize_TE_with_zero_bias = True

    #using HPO:
    convolution_layers = [(2, (5, 5), 1)]
    
    #use either one task embedding for all tasks or train a seperate model (tf.variable vs multiple layers)
    use_unique_task_embedding = True
    task_embedding = tf.random.normal([embedding_dim]) / 10
    task_embedding = tf.Variable(task_embedding, trainable=True)
    
    if use_unique_task_embedding:
        model = create_GRU(embedding_dim = embedding_dim,
                    hnet_hidden_dims= hnet_hidden_dims,
                    n_chunks = n_chunks,
                    use_unique_task_embedding = use_unique_task_embedding, 
                    convolution_layers=convolution_layers,
                    inner_net_dims = inner_net_dims,
                    depency_preservation_between_chunks = depency_preservation_between_chunks,
                    final_soft_max_layer = final_soft_max_layer,
                    initialize_TE_with_zero_bias = initialize_TE_with_zero_bias
                    )

    else:
        model = create_GRU(embedding_dim = embedding_dim,
                        hnet_hidden_dims= hnet_hidden_dims,
                        n_chunks = n_chunks,
                        use_unique_task_embedding = use_unique_task_embedding, inner_net_dims = inner_net_dims, convolution_layers=convolution_layers,
                        depency_preservation_between_chunks = depency_preservation_between_chunks,
                        final_soft_max_layer = final_soft_max_layer,
                        initialize_TE_with_zero_bias = initialize_TE_with_zero_bias)

    #with tf.device('/device:GPU:1'): preferable

    with tf.GradientTape() as tape:
        outs, _ = model([task_embedding, np.random.rand(1, 128, 32, 1), True])
        loss = tf.reduce_sum(outs)

    if use_unique_task_embedding:
        trainables = model.trainable_variables
    else:
        trainables = model.trainable_variables + [task_embedding]

    if not final_soft_max_layer:
        outs = tf.nn.softmax(outs)
        
    optimizer.build(trainables)
    grads = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(grads, trainables))








