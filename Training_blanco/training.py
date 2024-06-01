import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from itertools import islice
import matplotlib.pyplot as plt
import os
import time
import copy


@tf.function
def train_step(task_embedding, x, y, model, loss_fun, trainables, optimizer, class_incremental_case, use_unique_task_embedding,
               entropy_values, increase_classes = False,  increase_classes_allowed = False, final_soft_max_layer = False, treshold_CIL = 0, chunk_number = (False, 0)):

    needed_aid = False

    with tf.GradientTape() as tape:
        outs, _ = model([task_embedding, x, increase_classes])
        try:
            loss = loss_fun(y, outs)
        except: #occurs if the true label falls outside the prediction range
            #punish the model heavily by saying it had ZERO probability towards these or this new class(es)
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
                entropy_values, increase_classes = True,  increase_classes_allowed = True, final_soft_max_layer = final_soft_max_layer, treshold_CIL = treshold_CIL, chunk_number = chunk_number)
                return outs, entropy_values, optimizer, increase_classes, trainables, chunk_number
        
    #if adding a class, you must redefine the trainables and the optimizer
    if increase_classes:
        if use_unique_task_embedding:
            trainables = model.trainable_variables
        else:
            trainables = model.trainable_variables + [task_embedding]

        optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
        optimizer.build(trainables)

    grads = tape.gradient(loss, trainables) #you must update trainables when adding a class (increasing the model size)!
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
            X_test = [np.random.rand(2, 128, 32, 1)]*3,
            y_test = [np.random.randint(0, 3, size=(2))]*3,
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005), 
            embedding_dim = 100, #amount of values in chunk and task embedding
            num_classes =3, #initial number of classes
            l2reg=0.00001, # adding a penalty term to the loss function that is proportional to the square of the magnitude of the weights in the network. high value = reducing the complexity of the model = prevents overfitting.
            inner_net_dims = (200, 300), #amount of neurons in hidden layers of the innernet
            loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), #if from_logits is True, we expect NO soft max layer
            convolution_layers = [(8, (5, 5), 1), (16, (3, 3), 8), (32, (3, 3), 16)], # amount of filters, (kernel size), input dim
            use_unique_task_embedding = True, #if True, use (and train) a task embedding model to develop taskembeddings. If false, use random task embeddings, unique for each task.
            hnet_hidden_dims = (100, 50,), #amount of neurons in hidden layers of the hypernet
            dropout_rate = 0.3, #after every convolutional layer, there is a dropout layer. Higher dropout prevents overfitting
            n_chunks = 10, #the amount of 'chunks'. Every chunk is a part of the innernet weights. Determines repetition of the hypernet.
            max_attempts = 2, #if you use validation accuracy, detemines how many repetitions of a task is maximally allowed before accepting a certain accuracy score.
            epochs_per_task = 6, # use len(X_train) to test every image in a task. Determines amount of epochs that are running per task.
            validation_accuracy = None, # The minimum accuracy value you want to have before heading to the next task. Influences amount of task repetitions. You may also set this to None, or a negative value.
            class_incremental_case = (True, 0.35, 4, (True, 0.9), (0, 0.02), 0.4, 3),
                                        #True = class incremental case
                                        #if True, what is the entropy treshold? 
                                        #if True, after how many epochs/images are we considering adding classes? 
                                        #If True, Do you want to use Exponential Moving Average (EMA) as entropy measurement (= set to True), Alpha: values of alpha give more weight to recent observations, making the EMA more responsive to changes in the data.
                                        #If True, Resettle time = time after class is added to NOT add another class. Set to -1 if you dont want to use it. Entropy value to add to treshhold after a class is added.
                                        #If True, the trade offparameter. 0.4 is found as the best tradeoff parameter between probability and entropy by Da-Wei Zhou.
                                        #If True, the size of the buffer memory. If the model detects a new instance as a novelty, it stores the instance temporally in the buffer. See Da-Wei Zhou.
            initialize_TE_with_zero_bias = True, # Do you want to initialize the Task Embedding Model with zero biases (True), or random biases (False)?
            final_soft_max_layer = True, # If soft max is true, the loss function expects NO 'from_logits'. Determines if the last layer in the innernet is a(n indirectly) trainable softmax layer.
            training_while_testing = True,
            which_model = ('Dense', False, 100, 0.9, 0.1)): 
                                        # Dense, DRNN, ECHO, GRU or LSTM
                                        # If not Dense, would you like Depency Preservation between Chunks or between tasks?
                                        # If DRNN, what is the distilation rate? The dilation rate determines the spacing between the kernel elements. A dilation rate of 1 means standard convolution where every element in the input is considered. If ECHO, the: reservoir_size. The number of units in the reservoir layer. 
                                        # If ECHO, the spectral_radius. A high spectral radius means that signals are amplified more as they pass through the reservoir, potentially leading to faster dynamics and richer representations.
                                        # If ECHO, the sparsity. A high sparsity value means that a larger portion of the reservoir weight matrix is zero, resulting in fewer connections between reservoir units. This can lead to simpler dynamics and potentially better generalization.
    
    
    #setup data collection
    train_accuracies = []
    test_accuracies = []
    predictions = []

    #amount of classes = (amount of classes, in which task, in which epoch)
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
    
    #if not class_incremental_case[0]:
    optimizer.build(trainables) # Build optimizer with all trainable variables
    
    for i in range(0, n_tasks):
        print('RUNNING TASK {} \n'.format(i+1)+ '-'*30)

        #spread out testing times, with the assumption we have more training than test data
        X_test[i] = fill_with_none(X_test[i], X_train[i][:epochs_per_task])
        y_test[i] = fill_with_none(y_test[i], y_train[i][:epochs_per_task])
        
        #get a unique random task embedding for every task
        task_embedding = task_embeddings[i]

        #task id is the task number
        task_id = int(i)

        #this task embedding is only trainable if the task embedding model is not used
        if not use_unique_task_embedding:
            trainables = model.trainable_variables + [task_embedding]
            if class_incremental_case[0] and added_a_class:
                trainables = model.trainable_variables + task_embeddings
                optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
                optimizer.build(trainables)
                trainables = model.trainable_variables + [task_embedding]
        
        if validation_accuracy is None:
            final_accuracy, train_accuracies,test_accuracies,predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number  = train_epoch(model, task_embedding, trainables, optimizer, 
                        loss_fun = loss_fun,
                        epochs_per_task= epochs_per_task,
                        X_train = X_train[i],
                        y_train = y_train[i],
                        X_test= X_test[i],
                        y_test = y_test[i], 
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
                        predictions = predictions,
                        training_while_testing= training_while_testing,
                        chunk_number =chunk_number)
        else:
            for j in range (0, max_attempts):
                final_accuracy, train_accuracies,test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater,trainables, treshold_CIL, added_a_class, chunk_number = train_epoch(model, task_embedding, trainables, optimizer, 
                    loss_fun = loss_fun,
                    epochs_per_task= epochs_per_task,
                    X_train = X_train[i],
                    y_train = y_train[i],
                    X_test= X_test[i],
                    y_test = y_test[i],
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
                    predictions = predictions,
                    training_while_testing = training_while_testing,
                    chunk_number =chunk_number)

                if final_accuracy > validation_accuracy:
                    break
                elif j == (max_attempts-1):
                    print(f'Validation accuracy cannot be reached.')
                    break
                else:
                    #throw away results from the last run
                    train_accuracies = train_accuracies[:-len(X_train[i])]
                    test_accuracies = test_accuracies[:-len(X_test[i])]
                    print(f'Validation accuracy not reached. Repeating Task...')
    
    if class_incremental_case[0]:
        return train_accuracies, test_accuracies, predictions, amount_of_classes
    else:
        return train_accuracies, test_accuracies, predictions, None

def train_epoch(model,
                task_embedding,
                trainables, 
                optimizer,
                loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                epochs_per_task = 10,
                X_train = np.random.rand(10, 128, 32, 1), 
                y_train = np.random.randint(0, 3, size=(10)),
                X_test=np.random.rand(5, 128, 32, 1),
                y_test = np.random.randint(0, 3, size=(5)),
                train_accuracies = [],
                test_accuracies = [],
                predictions = [],
                class_incremental_case = (False, None, None, False, None),
                entropy_values = [],
                use_unique_task_embedding = False,
                task_id = 0,
                final_soft_max_layer = False,
                resettle_timer = 0,
                resettle_timer_updater = 0,
                treshold_CIL = 0,
                amount_of_classes = [],
                training_while_testing = False,
                chunk_number = (False, 0)):

    Accu_train = tf.metrics.SparseCategoricalAccuracy()
    Loss_train = tf.metrics.SparseCategoricalCrossentropy()

    Accu_test = tf.metrics.SparseCategoricalAccuracy()
    Loss_test = tf.metrics.SparseCategoricalCrossentropy()

    train_data = list(zip(X_train, y_train))
    tbar = tqdm(islice(train_data, epochs_per_task),
            total=epochs_per_task,
            ascii=True)
    
    #assumption that every task has the same amount of images. This is only used for class incremental learning.
    if class_incremental_case[0]:
        images_processed_estimation = task_id* epochs_per_task
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

    print(f' VALID: accuracy {Accu_test.result():6.3f}, loss {Loss_test.result():6.3f} \n')

    predictions.append((predictions_training, predictions_testing))
    return Accu_test.result(), train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, chunk_number















def visualize_accuracy(train_accuracies, test_accuracies, task_sizes, increase_class_positions, n_classes, title, DI):
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111)
    ax.set_position([0.1,0.1,0.5,0.8])
    
    # Plot training accuracy
    ax.plot(train_accuracies, label='Training Accuracy', marker='o')
    # Plot testing accuracy if available
    num_test_accuracies = len(test_accuracies)
    test_indices = np.arange(num_test_accuracies)
    test_indices_with_values = [i for i, acc in enumerate(test_accuracies) if acc is not None]  # Indices with non-None values
    test_values = [test_accuracies[i] for i in test_indices_with_values]  # Non-None test accuracy values
    ax.plot(test_indices_with_values, test_values, label='Testing Accuracy', marker='o', linestyle='-')
    
    # Add task boundaries
    task_boundaries = task_sizes
    for count, boundary in enumerate(task_boundaries):
        if not DI:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.4)
        else:
            if ((count-1) % 3 == 0):
                ax.axvline(x=boundary, color='purple', linestyle='-', linewidth=3)
            if ((count-1) % 3 == 1):
                ax.axvline(x=boundary, color='purple', linestyle='--', linewidth=2)
    
    # Add red lines and labels
    for count, position in enumerate(increase_class_positions):
        if count ==0:
            ax.axvline(x=position, color='red', linewidth=2 , linestyle='dotted')
        else:
            ax.axvline(x=position, color='red', linewidth=2)
        ax.text(position, -0.1, f'num classes: {n_classes[count]}                          ', rotation=90, verticalalignment='top', horizontalalignment='center', color='green')
    
    ax.set_xlabel('1 bar = 1 image')
    ax.set_ylabel('Accuracy')
    ax.set_title(r'$\bf{Accuracy\ Over\ Tasks}$' + '\n' + str(title), fontsize=16)
    ax.legend()
    ax.grid(True)

    # Adjust x-axis labels if test accuracies are fewer
    xticks = list(range(num_test_accuracies)) + task_boundaries
    
    # Rotate the task labels vertically
    if not DI:
        task_labels = [f'Task {i+1}' for i in range(len(task_sizes))]
    else:
        task_labels = []
        for i in range(0,len(task_sizes)):
            if ((i-1) % 3 == 0):
                task_labels.append(f'Domain shift at Task {i+1}')
            elif ((i-1) % 3 == 1):
                task_labels.append(f'Domain shift return at Task {i+1}')
            else:
                task_labels.append(f'Task {i+1}')

    ax.set_xticks(xticks)
    ax.set_xticklabels([''] * num_test_accuracies + task_labels, rotation=90)

    #plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0, wspace = 0.5, hspace= 0.5)
    
    # Return the figure
    return fig

    

# To expand the testing array to match the training array in length
# Spread values out as much as possible, with a bias towards testing at the end vs. the beginning of a task
def fill_with_none(short_array, long_array):
    if len(short_array) >= len(long_array):
        return short_array

    filled_array = [None] * len(long_array)
    spacing = len(long_array) // (len(short_array) - 1)

    for i, value in enumerate(short_array):
        if (i * spacing)< len(filled_array):
            filled_array[i * spacing] = value
        else:
            for k, j in enumerate(reversed(filled_array)):
                if j is None:
                    filled_array[len(filled_array) - 1 - k] = value
                    break

    #tackle this unwanted behaviour (no test at the end of a task) by reversing the order
    if filled_array[len(filled_array) -1] is None:
        return [filled_array[i] for i in range(len(filled_array) - 1, -1, -1)]
    
    return filled_array


def run_visualization(results, epochs_per_task, amount_of_tasks, file_name, DI):
    train_accuracies, test_accuracies, predictions, amount_of_classes = results[0], results[1], results[2], results[3]
    task_sizes = [epochs_per_task * i for i in range(0, amount_of_tasks)]

    if amount_of_classes is None:
        fig = visualize_accuracy(train_accuracies, test_accuracies, task_sizes, [], [], file_name, DI)
    else:
        increased_class = []
        n_classes = []
        for class_list in amount_of_classes:
            increased_class.append((class_list[1]-1)* epochs_per_task + class_list[2])
            n_classes.append(class_list[0])

        try:
            increased_class[0] = 0 #the first 'class increase' happens when initializing the model
        except:
            increased_class = 0
        fig = visualize_accuracy(train_accuracies, test_accuracies, task_sizes, increased_class, n_classes, file_name, DI)
    
    # Change this absolute path: the absolute path to the results folder
    path = os.path.abspath(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code\Results')

    #save figure in folder 'Results'
    existing_files = os.listdir(path)
    # Generate a unique filename using timestamp, if the image already exists 
    filename = f"{file_name}.jpg"
    if filename in existing_files:
        # If file with the same name exists, add a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{file_name}_{timestamp}.jpg"

    filepath = os.path.join(path, filename)

    fig.tight_layout(pad=3.0)


    # Save the figure
    fig.savefig(filepath, bbox_inches='tight', dpi=150)

if __name__ == "__main__":
    train_accuracies, test_accuracies, predictions, amount_of_classes = run()
    epochs_per_task= 6
    amount_of_tasks= 3

    results = (train_accuracies, test_accuracies, predictions, amount_of_classes)
    run_visualization(results, epochs_per_task,amount_of_tasks, 'test', False)
