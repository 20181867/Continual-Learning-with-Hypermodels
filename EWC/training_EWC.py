import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from itertools import islice
from Training_blanco.training import fill_with_none
import copy


" THE EWC SPECIFIC FUNTIONS"
def initialize_fisher_matrices(trainables):
    fisher_matrices = [tf.Variable(tf.zeros_like(w)) for w in trainables]
    return fisher_matrices


def calculate_ewc_loss(original_weights, fisher_matrices, trainables):

    try:
        ewc_loss = tf.constant(0.0)
        for weights, original_weight, fisher_matrix in zip(trainables, original_weights, fisher_matrices):
            ewc_loss += tf.reduce_sum(fisher_matrix * tf.square(weights - original_weight))
    except: #in case two dimensions were changed during adding a class, e.g. in GRU and LSTM with the recurrent gates
        ewc_loss = tf.constant(0.0)
        fisher_matrices = increase_fisher_matrices_size(fisher_matrices, trainables)
        for weights, original_weight, fisher_matrix in zip(trainables, original_weights, fisher_matrices):
            ewc_loss += tf.reduce_sum(fisher_matrix * tf.square(weights - original_weight))     

    return ewc_loss, fisher_matrices


def update_fisher_matrices(trainables, fisher_matrices, model, loss, train_data, task_embedding):
    for x,y in train_data:
        with tf.GradientTape() as tape:
            predictions, _ = model([task_embedding, x.reshape(1, 128, 32, 1), False])
            try:
                loss_value = loss(y, predictions)
            except:
                zeros_tensor = tf.zeros(int(y) - int(tf.size(predictions).numpy() -1 ))
                corrected_outs = tf.concat([predictions, zeros_tensor], axis=0)
                loss_value = loss(y, corrected_outs)

        gradients = tape.gradient(loss_value, trainables)

        for count, (fisher_matrix, gradient) in enumerate(zip(fisher_matrices, gradients)):
            try:
                fisher_matrices[count] = fisher_matrix + (tf.square(gradient)/len(trainables))
            except ValueError:
                print('WARNING: GRADIENT MAY BE NONE')
                continue
    return fisher_matrices

#function developed for Class Incremental Learning
def increase_fisher_matrices_size(fisher_matrices, trainables):
    #in DRNN, the task embedding model might 'flip to the front' in terms of the list of trainable variables:
    #to check this, we use the fact that the task embedding model is of fixex shape and size. 
    #Note: although it never occurs with these models, we also check if the task embedding model flips to the front
    if (((np.array(fisher_matrices[1]).shape != np.array(trainables[1]).shape) and (np.array(trainables[1]).shape == (1000,500))) or ((np.array(fisher_matrices[-5]).shape != np.array(trainables[-5]).shape) and (np.array(trainables[-5]).shape == (1000,500)))):
        starting_index = 0
        for i, array in enumerate(fisher_matrices):
            if array.shape == (1000, 500):
                starting_index = i
        task_embedding_model = fisher_matrices[(starting_index-1): starting_index+5]
        try:
            fisher_matrices = task_embedding_model + fisher_matrices[:(starting_index-1)] + fisher_matrices[starting_index+5:]
        except:
            try:
                fisher_matrices = task_embedding_model + fisher_matrices[:(starting_index-1)]
            except:
                fisher_matrices = task_embedding_model + fisher_matrices[starting_index+5:]


    new_fisher_matrices = []
    for count, (old_trainable, new_trainable) in enumerate(zip(fisher_matrices, trainables)):
        if old_trainable.shape != new_trainable.shape:
            #the kernel values
            if len(old_trainable.shape)== 2 and len(new_trainable.shape) ==2:
                #check if the first or the second value deviates. It is always the first value, except.... for GRU and LSTM!

                if old_trainable.shape[0] != new_trainable.shape[0]:
                    additional_values = new_trainable.shape[0] - old_trainable.shape[0]
                    new_fisher_matrices.append(tf.concat([fisher_matrices[count], tf.random.normal((additional_values, old_trainable.shape[1]))], axis=0))
                else:
                    additional_values = new_trainable.shape[1] - old_trainable.shape[1]
                    new_fisher_matrices.append(tf.concat([fisher_matrices[count], tf.random.normal((old_trainable.shape[0], additional_values))], axis=1))
            #the bias values
            elif len(old_trainable.shape)== 1 and len(new_trainable.shape) ==1:
                additional_values = new_trainable.shape[0] - old_trainable.shape[0]
                new_fisher_matrices.append(tf.concat([fisher_matrices[count], tf.random.normal((additional_values,))], axis=0))
        else:
            new_fisher_matrices.append(fisher_matrices[count])

    return new_fisher_matrices



" THE (MODIFIED) TRAINING FUNTIONS"
@tf.function
def train_step(task_embedding, x, y, model, loss_fun, trainables, optimizer, fisher_matrices, ewc_lambda,
               class_incremental_case, use_unique_task_embedding, entropy_values, increase_classes = False,  increase_classes_allowed = False, final_soft_max_layer = False, treshold_CIL = 0, original_weights = [], chunk_number = (None, None)):

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

        #if a new class is detected, the fisher matrices must be remade 
        if increase_classes:

            if use_unique_task_embedding:
                trainables = model.trainable_variables
                original_weights = copy.deepcopy(model.trainable_weights)
            else:
                trainables = model.trainable_variables+ [task_embedding]
                original_weights = copy.deepcopy(model.trainable_weights) + [copy.deepcopy(task_embedding)]


            fisher_matrices = copy.deepcopy(increase_fisher_matrices_size(fisher_matrices, trainables))

        ewc_loss, fisher_matrices = calculate_ewc_loss(original_weights, fisher_matrices, trainables)
        loss+= ewc_lambda* ewc_loss
        if model.losses:
            loss += tf.add_n(model.losses)

    #if we are in the class incremental learning senario
    if class_incremental_case[0] and increase_classes_allowed:

        global CIL_memory_buffer
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
                outs, entropy_values, optimizer, increase_classes, trainables, fisher_matrices, original_weights, chunk_number = train_step(task_embedding, x, y, model, loss_fun, trainables, optimizer, 
                                                                                                            fisher_matrices, ewc_lambda, (False, class_incremental_case[1], class_incremental_case[2], (False, 0), class_incremental_case[4]), 
                                                                                                            use_unique_task_embedding, entropy_values, increase_classes = True, increase_classes_allowed= True, final_soft_max_layer = final_soft_max_layer, treshold_CIL = treshold_CIL, original_weights = original_weights, chunk_number = chunk_number)
                return outs, entropy_values, optimizer, increase_classes, trainables, fisher_matrices, original_weights, chunk_number
        

    grads = tape.gradient(loss, trainables)
    try:
        optimizer.apply_gradients(zip(grads, trainables))
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
        optimizer.build(trainables)
        optimizer.apply_gradients(zip(grads, trainables))

    if needed_aid:
        outs = corrected_outs
    
    return outs, entropy_values, optimizer, increase_classes, trainables, fisher_matrices, original_weights, chunk_number


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
            embedding_dim = 99, 
            num_classes =3,
            l2reg = 0, 
            inner_net_dims = (2),
            loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
            convolution_layers = [(4, (5, 5), 1)],
            use_unique_task_embedding = False,
            hnet_hidden_dims = (1),
            dropout_rate = 0.1,
            n_chunks = 51,
            max_attempts = 5,
            ewc_lambda = 0.1,
            steps_per_epoch = 6,
            validation_accuracy = -0.1,
            class_incremental_case = (True, 0.2, 3, (False, 0.1), (2, 0.5), 0.4, 4),
            initialize_TE_with_zero_bias = True,
            final_soft_max_layer = True, #if soft max is true, the loss function expects NO 'from_logits'
            training_while_testing = True,
            which_model = ('DRNN', True, 100, 0.9, 0.1)):


    #setup data collection
    train_accuracies = []
    test_accuracies = []
    predictions = []
    amount_of_classes = [(num_classes, 0, 0)]

    #setup for the class incremental learning scenario
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
        #model.original_weights = [tf.Variable(w) for w in model.get_weights()]
    else:
        trainables = model.trainable_variables + task_embeddings
    

    optimizer.build(trainables)
    fisher_matrices_OG = initialize_fisher_matrices(trainables)

    n_tasks= len(X_train)

    for i in range(0, n_tasks):
        print('RUNNING TASK {} \n'.format(i+1)+ '-'*30)

        #spread out testing times, with the assumption we have more training than test data
        X_test[i] = fill_with_none(X_test[i], X_train[i][:steps_per_epoch])
        y_test[i] = fill_with_none(y_test[i], y_train[i][:steps_per_epoch])

        #get a unique random task embedding for every task
        task_embedding = task_embeddings[i]
    
        #task id is the task number
        task_id = int(i)

        #get the weights of the current (old) model
        original_weights = copy.deepcopy(model.trainable_weights)
        
        #this task embedding is only trainable if the task embedding model is not used
        if not use_unique_task_embedding:
            if added_a_class:
                all_trainables = model.trainable_variables + task_embeddings
                optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
                optimizer.build(all_trainables) 
            if i == 0: #the first itteration, no task_embedding is added yet
                trainables = model.trainable_variables + [task_embedding]
                original_weights = copy.deepcopy(model.trainable_weights) + [copy.deepcopy(task_embedding)]
                model_fisher_matrices = fisher_matrices_OG[:-n_tasks]
                fisher_matrices = model_fisher_matrices + [fisher_matrices_OG[len(model_fisher_matrices)+i]]
            else: #reuse information from the previous tasks, but with another task embedding
                trainables = trainables[:-1] + [task_embedding]
                original_weights = copy.deepcopy(model.trainable_weights) + [copy.deepcopy(task_embedding)]
                fisher_matrices = fisher_matrices[:-1] + [fisher_matrices_OG[len(model_fisher_matrices)+i]]
        else:
            if i == 0:
                fisher_matrices = fisher_matrices_OG
        
        if validation_accuracy is None:
            final_accuracy, train_accuracies,test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater,trainables, treshold_CIL, added_a_class, fisher_matrices,chunk_number = train_epoch(model, task_embedding, trainables, optimizer, 
                        fisher_matrices, ewc_lambda,
                        loss_fun = loss_fun,
                        steps_per_epoch= steps_per_epoch,
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
                        predictions =predictions,
                        original_weights = original_weights,
                        training_while_testing = training_while_testing,
                        chunk_number =chunk_number)
        else:
            for j in range (max_attempts):
                final_accuracy, train_accuracies,test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater,trainables, treshold_CIL, added_a_class, fisher_matrices,chunk_number = train_epoch(model, task_embedding, trainables, optimizer, 
                    fisher_matrices, ewc_lambda,
                    loss_fun = loss_fun,
                    steps_per_epoch= steps_per_epoch,
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
                    original_weights = original_weights,
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
                fisher_matrices,
                ewc_lambda,
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
                original_weights = [],
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

        outs, entropy_values, optimizer, increase_classes, trainables, fisher_matrices, original_weights, chunk_number = train_step(task_embedding, x.reshape(1, 128, 32, 1), y, model, 
                            loss_fun, trainables, optimizer, fisher_matrices, ewc_lambda, class_incremental_case, use_unique_task_embedding, entropy_values, increase_classes = False,
                            increase_classes_allowed = increase_classes_allowed, final_soft_max_layer = final_soft_max_layer,
                            treshold_CIL = treshold_CIL, original_weights = original_weights, chunk_number = chunk_number)

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

    #update fisher matrix
    fisher_matrices = update_fisher_matrices(trainables, fisher_matrices, model, loss_fun, train_data, task_embedding)
    print(f'\n TRAIN: accuracy {Accu_train.result():6.3f}, loss {Loss_train.result():6.3f}')

    if training_while_testing:
        print(f' VALID: accuracy {Accu_test.result():6.3f}, loss {Loss_test.result():6.3f} \n')
        predictions.append((predictions_training, predictions_testing))
        return Accu_test.result(), train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, fisher_matrices, chunk_number

    test_data = list(zip(X_test, y_test))
    for x, y in test_data:
        if x is None:
            test_accuracies.append(None)
            continue
        
        outs = test_step(model, task_embedding, x.reshape(1, 128, 32, 1), y)
        if class_incremental_case[0]:
            if int(y) > (tf.size(outs).numpy() -1):
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
    return Accu_test.result(), train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, fisher_matrices, chunk_number
    

if __name__ == "__main__":
    run()






