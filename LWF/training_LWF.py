import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from itertools import islice
from Training_blanco.training import fill_with_none

" THE LWF SPECIFIC FUNTIONS"
def distillation_loss(teacher_outputs, student_outputs, temperature, final_soft_max_layer):
    if not final_soft_max_layer:
        teacher_probs = tf.nn.softmax(teacher_outputs / temperature)
        student_probs = tf.nn.softmax(student_outputs / temperature)
    else:
        teacher_probs = teacher_outputs / temperature
        student_probs = student_outputs / temperature
    loss = tf.keras.losses.KLDivergence()(teacher_probs, student_probs)
    return loss

def distill_knowledge(teacher_model, teacher_model_trainable_variables, student_model_trainable_variables, distillation_temperature, unique_task_embedding):
    new_teacher_model_trainable_variables= []
    for teacher_var, student_var in zip(teacher_model_trainable_variables, student_model_trainable_variables):
        new_teacher_model_trainable_variables.append(teacher_var * (1 - distillation_temperature) + student_var * distillation_temperature)
    
    if unique_task_embedding:
        teacher_model.set_weights(new_teacher_model_trainable_variables)
        return new_teacher_model_trainable_variables
    else:
        teacher_model.set_weights(new_teacher_model_trainable_variables[:-1])
        task_embedding_teacher = new_teacher_model_trainable_variables[-1]
        return new_teacher_model_trainable_variables, task_embedding_teacher



" THE (MODIFIED) TRAINING FUNTIONS"
@tf.function
def train_step(task_embedding, x, y, model, loss_fun, trainables, 
               optimizer, 
               temperature, class_incremental_case, use_unique_task_embedding,
               entropy_values, increase_classes = False,  increase_classes_allowed = False, final_soft_max_layer = False, treshold_CIL = 0,
               teacher_model = None, task_embedding_t = [], chunk_number = (None, None)):

    needed_aid = False

    teacher_model_outputs, _ = teacher_model([task_embedding_t, x.reshape(1, 128, 32, 1), increase_classes])
    
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
        if teacher_model_outputs is not None:
            dist_loss = distillation_loss(teacher_model_outputs, outs, temperature, final_soft_max_layer)
            loss += dist_loss

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
                outs, entropy_values, optimizer, increase_classes, trainables, teacher_model_trainables, chunk_number = train_step(task_embedding, x, y, model, loss_fun, trainables, optimizer, temperature, (False, class_incremental_case[1], class_incremental_case[2], (False, 0), class_incremental_case[4]), use_unique_task_embedding,
                    entropy_values, increase_classes = True,  increase_classes_allowed = True, final_soft_max_layer = final_soft_max_layer, treshold_CIL = treshold_CIL, teacher_model = teacher_model, task_embedding_t = task_embedding_t, chunk_number = chunk_number)
                return outs, entropy_values, optimizer, increase_classes, trainables, teacher_model_trainables, chunk_number
        
    #if adding a class, you must redefine the trainables and the optimizer
    if increase_classes:
        if use_unique_task_embedding:
            trainables = model.trainable_variables
        else:
            trainables = model.trainable_variables + [task_embedding]
            teacher_model_trainables = teacher_model.trainable_variables + [task_embedding_t]
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
        optimizer.build(trainables)

    if use_unique_task_embedding:
        teacher_model_trainables = teacher_model.trainable_variables
    else:
        teacher_model_trainables = teacher_model.trainable_variables + [task_embedding_t]

    grads = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(grads, trainables))
    if needed_aid:
        outs = corrected_outs
    return outs, entropy_values, optimizer, increase_classes, trainables, teacher_model_trainables, chunk_number


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
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), 
            embedding_dim = 100, 
            num_classes =3,
            l2reg = 0, 
            inner_net_dims = (200, 300),
            loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            convolution_layers = [(8, (5, 5), 1), (16, (3, 3), 8), (32, (3, 3), 16)],
            use_unique_task_embedding = True,
            hnet_hidden_dims = (100, 50,),
            dropout_rate = 0.3,
            n_chunks = 10,
            max_attempts = 3,
            steps_per_epoch = 5,
            validation_accuracy = -1.0,
            update_every_epoch = False, 
            distillation_temperature=3,
            class_incremental_case = (True, 0, 4, (True, 0.1), (4, 0.1), 0.4, 5),
            initialize_TE_with_zero_bias = True,
            final_soft_max_layer = True,
            training_while_testing = False,
            which_model = ('Dense', False, 100, 0.9, 0.1)):

    # Check for GPU availability
    gpu_available = tf.config.list_physical_devices('GPU')
    if gpu_available:
        print("GPU is available")
        # Configure TensorFlow to use GPU memory growth
        for gpu in gpu_available:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set TensorFlow to use GPU by default
        tf.config.set_visible_devices(gpu_available, 'GPU')

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


    #initialize student model
    try:
        inner_net_dims = inner_net_dims + (num_classes,)
    except TypeError:
        inner_net_dims = (inner_net_dims, num_classes)
    n_tasks = len(X_train)
    task_embeddings_student = [tf.Variable(tf.random.normal([embedding_dim], stddev=1) / 10, trainable=True) for _ in range(n_tasks)]

    student_model = create_my_model(which_model, embedding_dim, n_chunks, hnet_hidden_dims,
                    inner_net_dims, dropout_rate, use_unique_task_embedding, 
                    convolution_layers, l2reg, initialize_TE_with_zero_bias,
                    final_soft_max_layer)
    
    if use_unique_task_embedding:
        student_model_trainables = student_model.trainable_variables
    else:
        student_model_trainables = student_model.trainable_variables + task_embeddings_student

    optimizer.build(student_model_trainables)

    #initialize teacher model
    task_embeddings_teacher = [tf.Variable(tf.random.normal([embedding_dim], stddev=1) / 10, trainable=True) for _ in range(n_tasks)]

    teacher_model = create_my_model(which_model, embedding_dim, n_chunks, hnet_hidden_dims,
                    inner_net_dims, dropout_rate, use_unique_task_embedding, 
                    convolution_layers, l2reg, initialize_TE_with_zero_bias,
                    final_soft_max_layer)

    if use_unique_task_embedding:
        teacher_model_trainables = teacher_model.trainable_variables

    else:
        teacher_model_trainables = teacher_model.trainable_variables + task_embeddings_teacher
        

    #run tasks
    for i in range(0, n_tasks):
        print('RUNNING TASK {} \n'.format(i+1)+ '-'*30)

        #spread out testing times, with the assumption we have more training than test data
        X_test[i] = fill_with_none(X_test[i], X_train[i][:steps_per_epoch])
        y_test[i] = fill_with_none(y_test[i], y_train[i][:steps_per_epoch])

        #get a unique random task embedding for every task
        task_embedding_student = task_embeddings_student[i]
        task_embedding_teacher = task_embeddings_teacher[i]

        #task id is the task number
        task_id = int(i)

        #this task embedding is only trainable if the task embedding model is not used
        if not use_unique_task_embedding:
            student_model_trainables = student_model.trainable_variables + [task_embedding_student]
            teacher_model_trainables = teacher_model.trainable_variables + [task_embedding_teacher]
            if class_incremental_case[0] and added_a_class:
                student_model_trainables = student_model.trainable_variables + task_embeddings_student
                optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.learning_rate)
                optimizer.build(student_model_trainables)
                student_model_trainables = student_model.trainable_variables + [task_embedding_student]

        if validation_accuracy is None:
            final_accuracy, train_accuracies,test_accuracies,predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, student_model_trainables, treshold_CIL, added_a_class, teacher_model_trainables_CI, chunk_number = train_epoch(student_model, task_embedding_student, student_model_trainables, optimizer, teacher_model, task_embedding_teacher, distillation_temperature,
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
                        predictions = predictions,
                        training_while_testing = training_while_testing,
                        chunk_number =chunk_number)
            
            if class_incremental_case[0]:
                teacher_model_trainables = teacher_model_trainables_CI

            if update_every_epoch:
                if use_unique_task_embedding:
                    student_model_trainables = student_model.trainable_variables
                    teacher_model_trainables = distill_knowledge(teacher_model, teacher_model_trainables, student_model_trainables, distillation_temperature, use_unique_task_embedding)

                else:
                    student_model_trainables = student_model.trainable_variables + [task_embedding_student]
                    teacher_model_trainables, task_embedding_teacher = distill_knowledge(teacher_model, teacher_model_trainables, student_model_trainables, distillation_temperature, use_unique_task_embedding)
                

        else:
            for j in range (max_attempts):
                final_accuracy, train_accuracies,test_accuracies,predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, student_model_trainables, treshold_CIL, added_a_class, teacher_model_trainables_CI, chunk_number = train_epoch(student_model, task_embedding_student, student_model_trainables, optimizer, teacher_model, task_embedding_teacher, distillation_temperature,
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
                    predictions = predictions,
                    training_while_testing = training_while_testing,
                    chunk_number =chunk_number)
                

                if class_incremental_case[0]:
                    teacher_model_trainables = teacher_model_trainables_CI
                    
                if update_every_epoch:
                    if use_unique_task_embedding:
                        student_model_trainables = student_model.trainable_variables
                        teacher_model_trainables = distill_knowledge(teacher_model, teacher_model_trainables, student_model_trainables, distillation_temperature, use_unique_task_embedding)

                    else:
                        student_model_trainables = student_model.trainable_variables + [task_embedding_student]
                        teacher_model_trainables, task_embedding_teacher = distill_knowledge(teacher_model, teacher_model_trainables, student_model_trainables, distillation_temperature, use_unique_task_embedding)
                    
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

        if not update_every_epoch: #but rather, update only once per task; for cases you use validation accuracy
            if use_unique_task_embedding:
                student_model_trainables = student_model.trainable_variables
                teacher_model_trainables = distill_knowledge(teacher_model, teacher_model_trainables, student_model_trainables, distillation_temperature, use_unique_task_embedding)

            else:
                student_model_trainables = student_model.trainable_variables + [task_embedding_student]
                teacher_model_trainables, task_embedding_teacher = distill_knowledge(teacher_model, teacher_model_trainables, student_model_trainables, distillation_temperature, use_unique_task_embedding)

    if class_incremental_case[0]:
        return train_accuracies, test_accuracies, predictions, amount_of_classes
    else:
        return train_accuracies, test_accuracies, predictions, None
    
def train_epoch(model,
                task_embedding,
                trainables, 
                optimizer,
                teacher_model,
                task_embedding_t,
                temperature,
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

        #teacher_outs, _ = teacher_model([task_embedding_t, x.reshape(1, 128, 32, 1)])
        outs, entropy_values, optimizer, increase_classes, trainables, teacher_model_trainables, chunk_number = train_step(task_embedding, x.reshape(1, 128, 32, 1), y, model, 
                            loss_fun, trainables, optimizer,
                            temperature, class_incremental_case, use_unique_task_embedding, entropy_values, increase_classes = False,
                            increase_classes_allowed = increase_classes_allowed, final_soft_max_layer = final_soft_max_layer,
                            treshold_CIL = treshold_CIL, teacher_model = teacher_model, task_embedding_t = task_embedding_t, chunk_number = chunk_number)
        
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
        return Accu_test.result(), train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, teacher_model_trainables, chunk_number

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
    return Accu_test.result(), train_accuracies, test_accuracies, predictions, amount_of_classes, entropy_values, optimizer, resettle_timer_updater, trainables, treshold_CIL, added_a_class, teacher_model_trainables, chunk_number

if __name__ == "__main__":
    run()







