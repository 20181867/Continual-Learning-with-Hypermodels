import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from itertools import islice
import copy
from Training_blanco.training import fill_with_none


" THE REPLAY SPECIFIC FUNTIONS"

def get_updated_trainables(model, validation_accuracy, task_embedding,
                           trainables, optimizer, loss_fun, n_tasks,
                           steps_per_epoch, X_train, y_train,
                           X_test, y_test, max_attempts, use_unique_task_embedding, task_data,
                           memory_regularization, frugal_learning,
                           train_accuracies = [], test_accuracies = [],
                           class_incremental_case = (False, None, None, False),
                           entropy_values = [], task_id =0, final_soft_max_layer = False,
                           resettle_timer = 0, resettle_timer_updater = 0, treshold_CIL = 0, amount_of_classes= 0,
                           which_model = None, embedding_dim = 100, n_chunks = 10, hnet_hidden_dims = (100, 200), inner_net_dims = (50, 100), dropout_rate = 0.3, convolution_layers = [(8, (5, 5), 1), (16, (3, 3), 8), (32, (3, 3), 16)],
                           l2reg = 0, initialize_TE_with_zero_bias = True, match_current_size_CIL = True, predictions = [], training_while_testing = False, chunk_number = (None, None)):

    old_index_TEM = []

    if len(task_data) >0:
        empty_input = tf.zeros((1,128,32,1))
        if not use_unique_task_embedding:
            weights_snapshots = [model([old_task_embedding, empty_input, False])[1] for old_task_embedding in
                            task_data]
        else:
            weights_snapshots = []
            #save the current weights
            weights_backup = model.get_weights().copy()

            #load the old (saved) weights of the Task Embedding Model
            for count, old_trainables in enumerate(task_data):

                current_weights = model.get_weights().copy()

                #locate the Task Embedding Model within the current weights
                if count == 0:
                    old_index_TEM = []

                    #get index of where the task embedding model is, assuming the task embedding model of:
                            # [8, (5, 5), 1] 2D convolutional layer
                            # (1000, 500) dense layer 
                    for index, weights in enumerate(current_weights):
                        this_weight = np.array(weights)
                        if this_weight.shape == (5, 5, 1, 8):
                            old_index_TEM.append(index)
                            old_index_TEM.append(index + 1) #for the filter bias
                        if this_weight.shape == (1000, 500):
                            #take one above this for the first dense layer between convolutional and hidden dense layer
                            old_index_TEM.append(index -1)
                            old_index_TEM.append(index)
                            old_index_TEM.append(index+1)
                            old_index_TEM.append(index+2)
                            old_index_TEM.append(index+3)
                            old_index_TEM.append(index+4)
                #assumption that the order of convolutional layers vs dense layers of the task embedding model did not change
                for count, index in enumerate(old_index_TEM):
                    current_weights[index] = copy.deepcopy(old_trainables[count])
                

                model.set_weights(current_weights)

                #run the model with these old weights of the Task Embedding Model
                weights_snapshots.append(model([task_embedding, empty_input, False])[1])
            
            #reset the model, including the task embedding model, to its current state
            model.set_weights(weights_backup)
    else:
        weights_snapshots = []
        empty_input = None

    
    if validation_accuracy is None:
        final_accuracy, train_accuracies,test_accuracies,predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number = train_epoch(model, task_embedding, trainables, optimizer, 
                                weights_snapshots, task_data, memory_regularization, frugal_learning,
                                n_tasks, empty_input,
                                loss_fun = loss_fun,
                                steps_per_epoch= steps_per_epoch,
                                X_train = X_train,
                                y_train = y_train,
                                X_test= X_test,
                                y_test = y_test,
                                train_accuracies = train_accuracies,
                                test_accuracies = test_accuracies,
                                class_incremental_case = class_incremental_case,
                                entropy_values = entropy_values,
                                use_unique_task_embedding = use_unique_task_embedding,
                                task_id = task_id,
                                final_soft_max_layer = final_soft_max_layer,
                                resettle_timer = resettle_timer,
                                resettle_timer_updater = resettle_timer_updater,
                                treshold_CIL = treshold_CIL,
                                amount_of_classes = amount_of_classes,
                                match_current_size_CIL = match_current_size_CIL, 
                                predictions = predictions,
                                training_while_testing = training_while_testing, chunk_number = chunk_number)
    else:
        for j in range (max_attempts):
            final_accuracy, train_accuracies,test_accuracies,predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number = train_epoch(model, task_embedding, trainables, optimizer, 
                                    weights_snapshots, task_data, memory_regularization, frugal_learning,
                                    n_tasks, empty_input,
                                    loss_fun = loss_fun,
                                    steps_per_epoch= steps_per_epoch,
                                    X_train = X_train,
                                    y_train = y_train,
                                    X_test= X_test,
                                    y_test = y_test,
                                    train_accuracies = train_accuracies,
                                    test_accuracies = test_accuracies,
                                    class_incremental_case = class_incremental_case,
                                    entropy_values = entropy_values,
                                    use_unique_task_embedding = use_unique_task_embedding,
                                    task_id = task_id,
                                    final_soft_max_layer = final_soft_max_layer,
                                    resettle_timer = resettle_timer,
                                    resettle_timer_updater = resettle_timer_updater,
                                    treshold_CIL = treshold_CIL,
                                    amount_of_classes = amount_of_classes,
                                    match_current_size_CIL = match_current_size_CIL, 
                                    predictions = predictions,
                                    training_while_testing = training_while_testing, chunk_number = chunk_number)
            if final_accuracy > validation_accuracy:
                break
            elif j == (max_attempts-1):
                print(f'Validation accuracy cannot be reached.')
                break
            else:
                #throw away results from the last run
                train_accuracies = train_accuracies[:-len(X_train)]
                test_accuracies = test_accuracies[:-len(X_test)]
                print(f'Validation accuracy not reached. Repeating Task...')

    if not use_unique_task_embedding:
        return task_embedding, train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number
    else:
        old_index_TEM = []
        #return only task embedding model trainables
        for index, weights in enumerate(trainables):
            this_weight = np.array(weights)
            if this_weight.shape == (5, 5, 1, 8):
                old_index_TEM.append(index)
                old_index_TEM.append(index + 1) #for the filter bias
            if this_weight.shape == (1000, 500):
                #take one above this for the first dense layer between convolutional and hidden dense layer
                old_index_TEM.append(index -1)
                old_index_TEM.append(index)
                old_index_TEM.append(index+1)
                old_index_TEM.append(index+2)
                old_index_TEM.append(index+3)
                old_index_TEM.append(index+4)

        old_trainables = copy.deepcopy([trainables[old_index_TEM[0]]] + [trainables[old_index_TEM[1]]]+ [trainables[old_index_TEM[2]]]+ [trainables[old_index_TEM[3]]]+ [trainables[old_index_TEM[4]]]+ [trainables[old_index_TEM[5]]]+ [trainables[old_index_TEM[6]]]+ [trainables[old_index_TEM[7]]])

        return old_trainables, train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number


#function developed for specifically for Class Incremental Learning
def check_and_correct_old_weight_sizes(old_weights, new_weights):

    if len(old_weights) != len(new_weights):
        additional_values = len(new_weights) - len(old_weights)
        old_weights = tf.concat([old_weights, tf.random.normal((additional_values, ))], axis=0)

    return old_weights



" THE (MODIFIED) TRAINING FUNTIONS"

@tf.function
def train_step(task_embedding, x, y, model, loss_fun, trainables, optimizer, class_incremental_case, use_unique_task_embedding,
               entropy_values, increase_classes = False,  increase_classes_allowed = False, final_soft_max_layer = False, treshold_CIL = 0, chunk_number = (None, None)):

    needed_aid = False

    with tf.GradientTape() as tape:
        outs, _ = model([task_embedding, x, increase_classes])
        try:
            loss = loss_fun(y, outs)
        except:
            zeros_tensor = tf.zeros(int(y) - int(tf.size(outs).numpy() -1 ))
            corrected_outs = tf.concat([outs, zeros_tensor], axis=0)
            loss = loss_fun(y, corrected_outs)
            needed_aid = True

        if model.losses:
            loss += tf.add_n(model.losses)

    #if we are in the class incremental learning senario
    if class_incremental_case[0] and increase_classes_allowed:

        global CIL_memory_buffer #
        epsilon = 1e-10  # Small epsilon value to prevent zero probabilities
        if not final_soft_max_layer:
            probabilities = tf.nn.softmax(np.squeeze(outs)) + epsilon
        else:
            probabilities = outs+ epsilon
        
        if len(probabilities.shape) == 1:
            entropy = (-tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=0))/ len(probabilities)
        else:
            entropy = (-tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=1))/ len(probabilities[1])

        entropy_values.append(entropy.numpy())
        #calculate similarity with previously stored te's
        x_for_CIL = np.reshape(x, [-1])

        similarity = 0
        for element in CIL_memory_buffer:
            try:
                if element == 0:
                    break
            except:
                similarity = (similarity + tf.reduce_sum(element * x_for_CIL) / (tf.norm(element)* tf.norm(x_for_CIL)))/2

        if class_incremental_case[3][0]: #if exponential moving average
            smoothed_entropy_values = exponential_moving_average(entropy_values, class_incremental_case[3][1])
            if similarity == 0:
                total_score = smoothed_entropy_values[-1]
            else:                
                try:
                    total_score = tf.minimum(class_incremental_case[5]* chunk_number[1], 1)* similarity + tf.maximum(1- (class_incremental_case[5]* chunk_number[1]), 0)* smoothed_entropy_values[-1]
                except:
                    total_score = tf.minimum(class_incremental_case[5]* chunk_number[1], 1)* similarity.numpy() + tf.maximum(1- (class_incremental_case[5]* chunk_number[1]), 0)* smoothed_entropy_values[-1]
        else:
            if similarity == 0:
                total_score = entropy
            else:
                #Score(x) = min(λc, 1)Prob(x) + max(1 − λc, 0)Ent(x)
                try:
                    total_score = tf.minimum(class_incremental_case[5]* chunk_number[1], 1)* similarity + tf.maximum(1- (class_incremental_case[5]* chunk_number[1]), 0)* entropy
                except:
                    total_score = tf.minimum(class_incremental_case[5]* chunk_number[1], 1)* similarity.numpy() + tf.maximum(1- (class_incremental_case[5]* chunk_number[1]), 0)* entropy


        if total_score> treshold_CIL: # a high entropy is encountered (higher then treshold)
            new_class_detected = True
        else:
            new_class_detected = False

        #report to researcher
        if not final_soft_max_layer:
            print(' ')
            print('Due to these probabilities: {}'.format(tf.round(np.squeeze(tf.nn.softmax(np.squeeze(outs))* 100)/100)))
            print('The entropy is: {}'.format(round(entropy.numpy(), 2)))
            print('The similarity score is: {}'.format(similarity))
            print('Therefore the score is: {} \n'.format(total_score))

        else:
            print(' ')
            print('Due to these probabilities: {}'.format((tf.round(outs * 100) / 100)))
            print('The entropy is: {}'.format(round(entropy.numpy(), 2)))
            print('The similarity score is: {}'.format(similarity))
            print('Therefore the score is: {} \n'.format(total_score))

        #if indeed a new class is detected and the model is allowed to adjust to this
        if new_class_detected and increase_classes_allowed:
            #check is buffer should be cleaned before adding to it, by counting empty space left:
            cleaning = True
            for count, element in enumerate(CIL_memory_buffer):
                try:
                    if element == 0:
                        CIL_memory_buffer[count] = x_for_CIL
                        cleaning = False
                        break
                except:
                    continue
            if cleaning:
                print("New class detected! Adjusting model...")
                CIL_memory_buffer = [0 for i in range(0,class_incremental_case[6])]
                CIL_memory_buffer[0] = x_for_CIL

                #recursion
                outs, entropy_values, optimizer, increase_classes, trainables, chunk_number = train_step(task_embedding, x, y, model, loss_fun, trainables, optimizer, (False, class_incremental_case[1], class_incremental_case[2], (False, 0), class_incremental_case[4]), use_unique_task_embedding,
                    entropy_values, increase_classes = True,  increase_classes_allowed = True, final_soft_max_layer = final_soft_max_layer, treshold_CIL = treshold_CIL, chunk_number= chunk_number)
                
                return outs, entropy_values, optimizer, increase_classes, trainables, chunk_number
        
    #if adding a class, you must redefine the trainables and the optimizer
    if increase_classes:
        if use_unique_task_embedding:
            trainables = model.trainable_variables
        else:
            trainables = model.trainable_variables + [task_embedding]

        optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
        optimizer.build(trainables)


    grads = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(grads, trainables))

    if needed_aid:
        outs = corrected_outs
    return outs, entropy_values, optimizer, increase_classes, trainables, chunk_number


@tf.function
def test_step(model, task_embedding, x, y):
    outs, _ = model([task_embedding,x, False])
    return outs

def exponential_moving_average(entropies, alpha=0.2):
    """Compute the exponential moving average over entropy values."""
    ema = [entropies[0]]
    for i in range(1, len(entropies)):
        ema.append(alpha * entropies[i] + (1 - alpha) * ema[-1])
    return ema



def create_my_model(which_model, embedding_dim, n_chunks, hnet_hidden_dims,
                    inner_net_dims, dropout_rate, use_unique_task_embedding, 
                    convolution_layers, l2reg, initialize_TE_with_zero_bias,
                    final_soft_max_layer):
    
    if which_model[0] == 'Dense':
        from Models.Hypermodel import create
        model = create(embedding_dim = embedding_dim,
                        n_chunks = n_chunks,
                        hnet_hidden_dims = hnet_hidden_dims,
                        inner_net_dims = inner_net_dims,
                        dropout_rate = dropout_rate,
                        use_unique_task_embedding = use_unique_task_embedding, 
                        convolution_layers=convolution_layers,
                        l2reg=l2reg,
                        initialize_with_zero_bias = initialize_TE_with_zero_bias,
                        final_soft_max_layer = final_soft_max_layer)
    elif which_model[0] == 'DRNN':
        #('DRNN', depency_preservation_between_chunks, dilation_rate)
        from Models.Hypermodel_DRNN import create
        model = create(embedding_dim=embedding_dim,
                        n_chunks=n_chunks,
                        hnet_hidden_dims=hnet_hidden_dims,
                        inner_net_dims=inner_net_dims,
                        #l2reg L2 regularization, also known as weight decay, penalizes large weights in the model by adding a term to the loss function that is proportional to the square of the weight values. This helps prevent overfitting by discouraging overly complex models.
                        l2reg=l2reg,
                        dropout_rate = dropout_rate,
                        use_unique_task_embedding = use_unique_task_embedding,
                        convolution_layers=convolution_layers, 
                        dilation_rate = which_model[2],
                        depency_preservation_between_chunks = which_model[1],
                        final_soft_max_layer = final_soft_max_layer,
                        initialize_TE_with_zero_bias = initialize_TE_with_zero_bias)
    elif which_model[0] == 'ECHO':
        #('ECHO', depency_preservation_between_chunks, reservoir_size, spectral_radius, sparsity)
        from Models.Hypermodel_ECHO import create
        model = create(embedding_dim=embedding_dim,
                        n_chunks=n_chunks,
                        hnet_hidden_dims=hnet_hidden_dims,
                        inner_net_dims=inner_net_dims,
                        l2reg=l2reg,
                        dropout_rate = dropout_rate,
                        use_unique_task_embedding = use_unique_task_embedding,
                        convolution_layers=convolution_layers, 
                        depency_preservation_between_chunks = which_model[1],
                        reservoir_size = which_model[2], spectral_radius = which_model[3], sparsity=which_model[4],
                        final_soft_max_layer = final_soft_max_layer,
                        initialize_TE_with_zero_bias = initialize_TE_with_zero_bias)
    elif which_model[0] == 'GRU':
        #('GRU', depency_preservation_between_chunks)
        from Models.Hypermodel_GRU import create_GRU
        model = create_GRU(embedding_dim=embedding_dim,
                        n_chunks=n_chunks,
                        hnet_hidden_dims=hnet_hidden_dims,
                        inner_net_dims=inner_net_dims,
                        l2reg=l2reg,
                        dropout_rate = dropout_rate,
                        use_unique_task_embedding = use_unique_task_embedding,
                        convolution_layers=convolution_layers, 
                        depency_preservation_between_chunks = which_model[1],
                        final_soft_max_layer = final_soft_max_layer,
                        initialize_TE_with_zero_bias = initialize_TE_with_zero_bias)
    elif which_model[0] == 'LSTM':
        #('GRU', depency_preservation_between_chunks)
        from Models.Hypermodel_LSTM import create_LSTM
        model = create_LSTM(embedding_dim=embedding_dim,
                            n_chunks=n_chunks,
                            hnet_hidden_dims=hnet_hidden_dims,
                            inner_net_dims=inner_net_dims,
                            l2reg=l2reg,
                            dropout_rate = dropout_rate,
                            use_unique_task_embedding = use_unique_task_embedding,
                            convolution_layers=convolution_layers, 
                            depency_preservation_between_chunks = which_model[1],
                            final_soft_max_layer = final_soft_max_layer,
                            initialize_TE_with_zero_bias = initialize_TE_with_zero_bias)
    else:
        raise ValueError('Please enter a correct model (Dense, DRNN, ECHO, GRU or LSTM)')

    return model



def run(X_train = [np.random.rand(10, 128, 32, 1)]*3, 
            y_train = [np.random.randint(0, 3, size=(10))]*3,
            X_test = [np.random.rand(5, 128, 32, 1)]*3,
            y_test = [np.random.randint(0, 3, size=(5))]*3,
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), 
            embedding_dim = 100, 
            num_classes =3,
            l2reg = 0,
            inner_net_dims = (200, 300),
            loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            convolution_layers = [(8, (5, 5), 1), (16, (3, 3), 8), (32, (3, 3), 16)],
            use_unique_task_embedding = False,
            hnet_hidden_dims = (100, 50),
            dropout_rate = 0.3,
            n_chunks = 10,
            max_attempts = 3,
            steps_per_epoch = 5,
            validation_accuracy = -0.1,
            memory_regularization = 0.6, #how much do you let the past influence the current predictions
            frugal_learning = (True, 2, 10), #(Do you want to sample when n_tasks reaches a certain amount?, If True, what amount?, If True, many samples?)
            class_incremental_case = (True, 0, 5, (False, 0.1), (4, 0.3), 0.4, 3),
            initialize_TE_with_zero_bias = True,
            final_soft_max_layer = True,
            training_while_testing = False,
            which_model = ('DRNN', False, 100, 0.9, 0.1),
            match_current_size_CIL = False): #If True, the weight snapshots are increased in size to match the new weights if classes were added in the mean time. If False, the new weights are decreased, with slicing to match the weight snapshots.



    #setup data collection
    train_accuracies = []
    test_accuracies = []
    predictions = []
    amount_of_classes = [(num_classes, 0, 0)]

    #for the class incremental learning scenario
    entropy_values = []
    resettle_timer, resettle_timer_updater = class_incremental_case[4][0], class_incremental_case[4][0]
    treshold_CIL = class_incremental_case[1]
    added_a_class = False
    chunk_number = (False, -class_incremental_case[2])
    global CIL_memory_buffer
    CIL_memory_buffer = [0 for i in range(0, class_incremental_case[6])]


    try:
        inner_net_dims = inner_net_dims + (num_classes,)
    except TypeError:
        inner_net_dims = (inner_net_dims, num_classes)
        
    n_tasks = len(X_train)
    task_embeddings = [tf.Variable(tf.random.normal([embedding_dim], stddev=1) / 10, trainable=True) for _ in range(n_tasks)]

    #create model
    model = create_my_model(which_model, embedding_dim, n_chunks, hnet_hidden_dims,
                    inner_net_dims, dropout_rate, use_unique_task_embedding, 
                    convolution_layers, l2reg, initialize_TE_with_zero_bias,
                    final_soft_max_layer)

    if use_unique_task_embedding:
        trainables = model.trainable_variables
    else:
        trainables = model.trainable_variables + task_embeddings
    
    optimizer.build(trainables)

    #memory 
    task_data = []

    for i in range(0, n_tasks):
        print('RUNNING TASK {} \n'.format(i+1)+ '-'*30)

        #spread out testing times, with the assumption we have more training than test data
        X_test[i] = fill_with_none(X_test[i], X_train[i][:steps_per_epoch])
        y_test[i] = fill_with_none(y_test[i], y_train[i][:steps_per_epoch])

        #get a unique random task embedding for every task
        task_embedding = task_embeddings[i]

        #task id is the task number
        task_id = int(i)

        if use_unique_task_embedding:
            old_task_embedding_model, train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number  = get_updated_trainables(model, validation_accuracy, task_embedding,
                           trainables, optimizer, loss_fun, n_tasks,
                           steps_per_epoch, X_train[i], y_train[i],
                           X_test[i], y_test[i], max_attempts, use_unique_task_embedding,
                           task_data, memory_regularization, frugal_learning,
                           train_accuracies = train_accuracies,
                           test_accuracies = test_accuracies,
                           class_incremental_case = class_incremental_case,
                           entropy_values = entropy_values,
                           task_id = task_id,
                           final_soft_max_layer = final_soft_max_layer,
                           resettle_timer = resettle_timer,
                           resettle_timer_updater = resettle_timer_updater,
                           treshold_CIL = treshold_CIL,
                           amount_of_classes = amount_of_classes,
                           which_model = which_model, embedding_dim = embedding_dim, n_chunks = n_chunks, hnet_hidden_dims = hnet_hidden_dims, inner_net_dims = inner_net_dims, dropout_rate = dropout_rate,
                           convolution_layers = convolution_layers, l2reg = l2reg, initialize_TE_with_zero_bias = initialize_TE_with_zero_bias, match_current_size_CIL = match_current_size_CIL, predictions = predictions, training_while_testing = training_while_testing, chunk_number =chunk_number)
            task_data.append(old_task_embedding_model)
            
            
        else:
            trainables = model.trainable_variables + [task_embedding]
            if class_incremental_case[0] and added_a_class:
                trainables = model.trainable_variables + task_embeddings
                optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
                optimizer.build(trainables)
                trainables = model.trainable_variables + [task_embedding]
            old_task_embedding, train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number = get_updated_trainables(model, validation_accuracy, task_embedding,
                           trainables, optimizer, loss_fun, n_tasks,
                           steps_per_epoch, X_train[i], y_train[i],
                           X_test[i], y_test[i], max_attempts, use_unique_task_embedding,
                           task_data, memory_regularization, frugal_learning,
                           train_accuracies = train_accuracies,
                           test_accuracies = test_accuracies,
                           class_incremental_case = class_incremental_case,
                           entropy_values = entropy_values,
                           task_id = task_id,
                           final_soft_max_layer = final_soft_max_layer,
                           resettle_timer = resettle_timer,
                           resettle_timer_updater = resettle_timer_updater,
                           treshold_CIL = treshold_CIL,
                           amount_of_classes = amount_of_classes,
                           which_model = which_model, embedding_dim = embedding_dim, n_chunks = n_chunks, hnet_hidden_dims = hnet_hidden_dims, inner_net_dims = inner_net_dims, dropout_rate = dropout_rate,
                           convolution_layers = convolution_layers, l2reg = l2reg, initialize_TE_with_zero_bias = initialize_TE_with_zero_bias, match_current_size_CIL =match_current_size_CIL, predictions = predictions, training_while_testing = training_while_testing, chunk_number = chunk_number)
            task_data.append(old_task_embedding)

    if class_incremental_case[0]:
        return train_accuracies, test_accuracies, predictions, amount_of_classes
    else:
        return train_accuracies, test_accuracies, predictions, None
        
    
def train_epoch(model,
                task_embedding,
                trainables, 
                optimizer,
                weights_snapshots,
                task_data, 
                memory_regularization,
                frugal_learning,
                n_tasks,
                empty_input,
                loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                steps_per_epoch = 5,
                X_train = np.random.rand(10, 128, 32, 1), 
                y_train = np.random.randint(0, 3, size=(10)),
                X_test=np.random.rand(5, 128, 32, 1),
                y_test = np.random.randint(0, 3, size=(5)),
                train_accuracies = [],
                test_accuracies = [],
                predictions = [],
                class_incremental_case = (False, None, None, False),
                entropy_values = [],
                use_unique_task_embedding = False,
                task_id = 0,
                final_soft_max_layer = False,
                resettle_timer = 0,
                resettle_timer_updater = 0,
                treshold_CIL = 0,
                amount_of_classes = [],
                match_current_size_CIL = False,
                training_while_testing = False,
                chunk_number = (None, None)):

    Accu_train = tf.metrics.SparseCategoricalAccuracy()
    Loss_train = tf.metrics.SparseCategoricalCrossentropy()

    Accu_test = tf.metrics.SparseCategoricalAccuracy()
    Loss_test = tf.metrics.SparseCategoricalCrossentropy()

    train_data = list(zip(X_train, y_train))
    tbar = tqdm(islice(train_data, steps_per_epoch),
            total=steps_per_epoch,
            ascii=True)
    
    #assumption that every task has the same amount of images. This is only used for class incremental learning.
    if class_incremental_case[0]:
        images_processed_estimation = task_id* steps_per_epoch
    increase_classes_allowed = False #for the initial class incremental timer
    added_a_class = False

    predictions_training = []
    predictions_testing = []

    ''' MAIN TRAINING LOOP'''
    for epoch, (x, y) in enumerate(tbar):

        if class_incremental_case[0]:
            # Resettle time handling, if the resettle time has run out or no additional classes have been found yet
            if (resettle_timer_updater == 0) or (resettle_timer_updater == resettle_timer):
                resettle_timer_updater = resettle_timer
                increase_classes_allowed_resettle = True
            elif (resettle_timer_updater + 1) == resettle_timer:
                increase_classes_allowed_resettle = False
            else:
                increase_classes_allowed_resettle = False

            # Initial timer handling    
            images_processed_estimation = images_processed_estimation+1
            if ((class_incremental_case[2] -images_processed_estimation) < 0) and (increase_classes_allowed_resettle):
                increase_classes_allowed = True
            else:
                increase_classes_allowed = False

        outs, entropy_values, optimizer, increase_classes, trainables, chunk_number = train_step(task_embedding, x.reshape(1, 128, 32, 1), y, model, 
                            loss_fun, trainables, optimizer, class_incremental_case, use_unique_task_embedding, entropy_values, increase_classes = False,
                            increase_classes_allowed = increase_classes_allowed, final_soft_max_layer = final_soft_max_layer,
                            treshold_CIL = treshold_CIL, chunk_number = chunk_number)

        # Resettle time handling,
        if (class_incremental_case[0]) and (resettle_timer_updater > 0) and (increase_classes):
            resettle_timer_updater -= 1
        elif (class_incremental_case[0]) and (resettle_timer_updater != resettle_timer):
            resettle_timer_updater -= 1
        
        # Increase treshold for class incremental learning after finding one class
        if (class_incremental_case[0]) and increase_classes:
            treshold_CIL += class_incremental_case[4][1]
            added_a_class = True
            amount_of_classes.append((len(outs.numpy()), task_id+1, epoch))
            chunk_number = (True, -resettle_timer_updater)
            if class_incremental_case[4][1] != 0:
                print('\n the treshold for adding a new class just increased and is now {}'.format(round(treshold_CIL, 2)))
        elif (class_incremental_case[0]) and (not increase_classes):
            chunk_number = (chunk_number[0], chunk_number[1]+1)
        
        if memory_regularization>0 and len(task_data) >0 and len(weights_snapshots)>0:
            with tf.GradientTape() as tape:
                loss = tf.constant(0, dtype=tf.float32)

                # for reading the code: see first the ELSE statement. The IF statement does not contain explanations.
                if frugal_learning[0] and len(task_data) >= frugal_learning[1]:
                    #for frugal learning, pick a subset when the amount of tasks gets high.
                    #this way, regularization over many tasks is prevented.
                    print('-'*30 + 'FRUGAL LEARNING INTERVENTION' + '-'*30)
                    samples = tf.random.uniform(shape=(frugal_learning[2],), minval=0, maxval=len(task_data), dtype=tf.int32)
                    samples = tf.unique(samples)[0]
                    num_samples = tf.cast(len(samples), tf.float32)
                    for count, sample_idx in enumerate(samples):
                        if use_unique_task_embedding:
                            old_model_weights = task_data[sample_idx]
                            weight_snapshot = weights_snapshots[sample_idx]
                            weights_backup = model.get_weights().copy()
                            current_weights = model.get_weights().copy()
                            if count == 0: 
                                index_TEM = []
                                for index, weights in enumerate(current_weights):
                                    this_weight = np.array(weights)
                                    if this_weight.shape == (5, 5, 1, 8):
                                        index_TEM.append(index)
                                        index_TEM.append(index + 1)
                                    if this_weight.shape == (1000, 500):
                                        index_TEM.append(index -1)
                                        index_TEM.append(index)
                                        index_TEM.append(index+1)
                                        index_TEM.append(index+2)
                                        index_TEM.append(index+3)
                                        index_TEM.append(index+4)
                            for count_, index in enumerate(index_TEM):
                                current_weights[index] = copy.deepcopy(old_model_weights[count_])
                            model.set_weights(current_weights)
                            _, weights = model([task_embedding, empty_input, False])
                            model.set_weights(weights_backup)
                        else:
                            old_token = tf.gather(task_data, sample_idx)
                            weight_snapshot = tf.gather(weights_snapshots, sample_idx)
                            _, weights = model([old_token, empty_input, False])
                        if class_incremental_case[0]:
                            if match_current_size_CIL:
                                weight_snapshot = check_and_correct_old_weight_sizes(weight_snapshot, weights)
                            else:
                                weights = weights[:len(weight_snapshot)]
                        l2 = tf.reduce_sum(tf.square(weights - weight_snapshot))
                        loss += memory_regularization * l2 / num_samples
                else:
                    if use_unique_task_embedding:
                        weights_backup = model.get_weights().copy()

                        # weights_snapshots = the weights that the PREVIOUS VERSION of the hypermodel would give to the targetnetwork, given the OLD Task Embedding Model.
                        for count, (old_task_embedding_model_weights, weight_snapshot) in enumerate(zip(task_data, weights_snapshots)): 
                            
                            # load the old task embedding model with the new hypermodel
                            current_weights = model.get_weights().copy()

                            #locate where the current Task Embedding Model exists 
                            if count == 0:
                                index_TEM = []
                                for index, weights in enumerate(current_weights):
                                    this_weight = np.array(weights)
                                    if this_weight.shape == (5, 5, 1, 8):
                                        index_TEM.append(index)
                                        index_TEM.append(index + 1) 
                                    if this_weight.shape == (1000, 500):
                                        index_TEM.append(index -1)
                                        index_TEM.append(index)
                                        index_TEM.append(index+1)
                                        index_TEM.append(index+2)
                                        index_TEM.append(index+3)
                                        index_TEM.append(index+4)

                            #we make one assumption: the convolutional layer is always underneath or above the dense layers, but this ordering does not change.
                            for count_, index in enumerate(index_TEM):
                                current_weights[index] = copy.deepcopy(old_task_embedding_model_weights[count_])

                            model.set_weights(current_weights)

                            # weights = the weights that the CURRENT VERSION of the hypernetwork would give to the targetnetwork, given the OLD Task Embedding Model
                            _, weights = model([task_embedding, empty_input, False]) #the input, task embedding, remains the same (untrainable) random variable

                            # calculate l2 given:
                                    # the weights that CURRENT VERSION of the hypermodel would give to the targetnetwork
                                    # minus the weights that the PERVIOUS VERSION of the hypermodel would give to the targetnetwork
                                    # GIVEN the same (old) target network model
                            
                            #.......... if the CURRENT VERSION of the hypermodel is updated by increasing the amount of classes, the weights of the weight snapshot differ in size! .......
                            if class_incremental_case[0]:
                                if match_current_size_CIL:
                                    #increase the old weights if necessary by adding random values to match the new weights
                                    weight_snapshot = check_and_correct_old_weight_sizes(weight_snapshot, weights)
                                else:
                                    #Decrease the new weights if necessary by adding random values to match the new weights
                                    weights = weights[:len(weight_snapshot)]

                            l2 = tf.reduce_sum(tf.square(weights - weight_snapshot))
                            loss += memory_regularization * l2 / len(task_data)
        
                        #reset the model, including the task embedding model, to its current state
                        model.set_weights(weights_backup)
                    else: 
                        #weights_snapshots = the weights that the PREVIOUS VERSION of the hypermodel would give to the targetnetwork, given the OLD task embedding.
                        for old_token, weight_snapshot in zip(task_data, weights_snapshots):

                            # weights = the weights that the CURRENT VERSION of the hypernetwork would give to the targetnetwork, given the OLD task embedding.
                            _, weights = model([old_token, empty_input, False])

                            #.......... if the CURRENT VERSION of the hypermodel is updated by increasing the amount of classes, the weights of the weight snapshot differ in size! .......
                            if class_incremental_case[0]:
                                if match_current_size_CIL:
                                    #increase the old weights if necessary by adding random values to match the new weights
                                    weight_snapshot = check_and_correct_old_weight_sizes(weight_snapshot, weights)
                                else:
                                    #Decrease the new weights if necessary by adding random values to match the new weights
                                    weights = weights[:len(weight_snapshot)]
                            
                            l2 = tf.reduce_sum(tf.square(weights - weight_snapshot))
                            loss += memory_regularization * l2 / len(task_data)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #update stats
        guessed_class = tf.argmax(outs).numpy()
        predictions_training.append(guessed_class)

        Accu_train.update_state(y, outs)
        Loss_train.update_state(y, outs)
        train_accuracies.append(Accu_train.result().numpy())

        if training_while_testing:
            if X_test[epoch] is not None:
                outs = test_step(model, task_embedding, x.reshape(1, 128, 32, 1), y_test[epoch])

                if class_incremental_case[0]: # we potentially miss a class
                    if int(y_test[epoch]) > (tf.size(outs).numpy() -1): # we definitely miss a class
                        #say that the probability assigned to this class was zero
                        zeros_tensor = tf.zeros(int(y_test[epoch]) - int(tf.size(outs).numpy() -1 ))
                        outs = tf.concat([outs, zeros_tensor], axis=0)

                #update stats
                guessed_class = tf.argmax(outs).numpy()
                Accu_test.update_state(y_test[epoch], outs)
                Loss_test.update_state(y_test[epoch], outs)
                test_accuracies.append(Accu_test.result().numpy())
                predictions_testing.append(guessed_class)
            else:
                test_accuracies.append(None)

    print(f'\n TRAIN: accuracy {Accu_train.result():6.3f}, loss {Loss_train.result():6.3f}')

    if training_while_testing:
        print(f' VALID: accuracy {Accu_test.result():6.3f}, loss {Loss_test.result():6.3f} \n')
        predictions.append((predictions_training, predictions_testing))
        return Accu_test.result(), train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number


    test_data = list(zip(X_test, y_test))
    for x, y in test_data:
        if x is None:
            test_accuracies.append(None)
            continue

        outs = test_step(model, task_embedding, x.reshape(1, 128, 32, 1), y)

        if class_incremental_case[0]: # we potentially miss a class
            if int(y) > (tf.size(outs).numpy() -1): # we definitely miss a class
                #say that the probability assigned to this class was zero
                zeros_tensor = tf.zeros(int(y) - int(tf.size(outs).numpy() -1 ))
                outs = tf.concat([outs, zeros_tensor], axis=0)

        #update stats
        guessed_class = tf.argmax(outs).numpy()
        Accu_test.update_state(y, outs)
        Loss_test.update_state(y, outs)
        test_accuracies.append(Accu_test.result().numpy())
        predictions_testing.append(guessed_class)

    print(f' VALID: accuracy {Accu_test.result():6.3f}, loss {Loss_test.result():6.3f}')
    predictions.append((predictions_training, predictions_testing))
    return Accu_test.result(), train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number
    

if __name__ == "__main__":
    run()





