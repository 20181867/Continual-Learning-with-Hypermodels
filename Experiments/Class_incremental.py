import random
import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import tensorflow as tf
import numpy as np
from Experiments.learning_functions import run_learning_function
import pandas as pd
import yaml
import os
import json
from Training_blanco.training import run_visualization
from Preprocessing.load_data import load_the_data

if __name__ == "__main__":
        
    '''       HYPERPARAMETERS        '''

    configs = yaml.safe_load_all(open('class_incremental.yaml', 'r'))
    for config in configs:
        print("RUNNING EXPERIMENT {}".format(config['run_name']))

        optimizer = eval(config['Optimizer_and_Learning_Rate'])
        loss_fun = eval(config['Loss_function'])
        epochs_per_tasks = int(config['Number_of_epochs_per_task'])
        embedding_dim = int(config['Embedding_dimension_chunk_and_task_embeddings'])
        use_unique_task_embedding = config['Use_a_seperate_task_embedding_model']
        initialize_TE_with_zero_bias = config['Initialize_task_embedding_model_with_zero_bias_(True)_or_random_bias_(False)']
        inner_net_dims = eval(config['Target_network_dimension'])
        convolution_layers = eval(config['Convolutional_layers'])
        final_soft_max_layer = config['A_final_trainable_soft_max_layer_in_the_target_network']
        dropout_rate = float(config['Dropout_rate_in_target_network'])
        n_chunks = int(config['Number_of_chunks'])
        hnet_hidden_dims = eval(config['Hypernetwork_dimension'])
        num_classes = int(config['Number_of_initial_classes'])
        class_incremental_case = eval(config['Class_Incremental_Case'])
        l2reg = float(config['L2_regularization_strenght'])
        validation_accuracy = float(config['Validation_accuracy'])
        max_attempts = int(config['Max_attempts_when_using_validation_accuracy'])
        sr = int(config['Sampel_rate_sound_data'])
        test_data_per_task = int(config['Test_images_per_task'])
        testing_while_training = config['Testing_while_training']

        which_learningfunction = eval(config['Which_learningfunction'])
        which_model = eval(config['Which_Model'])


        # Hyperparameter that remained from the previous research (see report):
        preprocessing = 'MEL'

        # Parameters for Class Incremental Learning
        additional_classes = int(config['Additional_classes_per_task'])
        amount_of_tasks = int(config['Amount_of_tasks'])
        dilution_fraction = float(eval(config['Dilution_factor']))

        # Parameters for running experiments
        append_results = config['Add_results_to_previous_experiment_results']
        visualize_results = config['Visualize_results']


        '''       BEGIN CODE        '''
        
        hyperparameters = (optimizer, embedding_dim, num_classes, l2reg, inner_net_dims, loss_fun, convolution_layers, use_unique_task_embedding, hnet_hidden_dims, dropout_rate, n_chunks, max_attempts, epochs_per_tasks, validation_accuracy, class_incremental_case, initialize_TE_with_zero_bias, final_soft_max_layer, testing_while_training)

        classes = [ 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
                        'house', 'left', 'marvel', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                        'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

        # Change this absolute path: the absolute path to the results folder
        parent_folder_path = r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code\Results'
        file_path = os.path.join(parent_folder_path, 'experiments_results_CIL.csv')
        
        if not append_results:
            if os.path.exists(file_path):
                os.remove(file_path)

        def get_task_data (classes, amount_of_tasks, amount_of_initial_classes, addition_classes_per_task, sr, preprocessing, epochs_per_tasks, test_data_per_task):
            randomizer = random.randint(0, len(classes)-amount_of_initial_classes-addition_classes_per_task*(amount_of_tasks-1)-1)
            X_train, X_test, y_train, y_test = [], [], [], []
            all_classes = {}
            conversions = []
            for i in range(0, amount_of_tasks):
                if i == 0:
                    task_classes = classes[randomizer:amount_of_initial_classes+randomizer]
                    randomizer += len(task_classes)
                else:
                    task_classes = classes[randomizer:addition_classes_per_task+randomizer]
                    randomizer += len(task_classes)
                
                all_classes['Task ' + str(i) + ' had these additional classes included:'] = task_classes
                X_train_this_task, X_test_this_task, y_train_this_task, y_test_this_task, conversion_this_task, _ = load_the_data(preprocessing, task_classes,sr, False)
                X_train_this_task = np.expand_dims(X_train_this_task[:epochs_per_tasks], axis=-1)
                y_train_this_task = y_train_this_task[:epochs_per_tasks]
                X_test_this_task = np.expand_dims(X_test_this_task[:test_data_per_task], axis=-1)
                y_test_this_task = y_test_this_task[:test_data_per_task]

                X_train.append(X_train_this_task)
                X_test.append(X_test_this_task)
                y_train.append(y_train_this_task)
                y_test.append(y_test_this_task)
                conversions.append(conversion_this_task)
            
            return X_train, y_train, X_test, y_test, conversions, all_classes
            
        X_train, y_train, X_test, y_test, conversions, all_classes = get_task_data(classes, amount_of_tasks, num_classes,additional_classes, sr, preprocessing, epochs_per_tasks, test_data_per_task)

        #x_train_backup = copy.deepcopy(X_train)
        #X_test_backup = copy.deepcopy(X_test)

        # Labels of new classes are now ALSO 0, 1, 2 ect. when they should be starting like 3,4,5
        def increase_label_values(y_labels):
            min_label_value = 0
            for count, label_values in enumerate(y_labels):
                for count2, value in enumerate(label_values):
                    y_labels[count][count2] += min_label_value
                min_label_value = max(label_values) +1 # +1 because classification starts at 0
            return y_labels

        y_train = increase_label_values(y_train)
        y_test = increase_label_values(y_test)

        #y_test_backup = copy.deepcopy(y_test)
        #y_train_backup = copy.deepcopy(y_train)

        def preprocess_CI(x_data, y_data, dilution_factor):

            prev_task_data_x_train = []
            prev_task_data_y_train = []  
            
            y_data = list(y_data)
            for count, (x_image, y_label) in enumerate(zip(x_data, y_data)):
            #count = the task id
                if count == 0:
                    #in the first task id, we have no dilution
                    prev_task_data_x_train.append(x_image)
                    prev_task_data_y_train.append(y_label)
                else:
                    dilution_factor = int(len(x_image) * dilution_fraction)
                    #pick random images to replace with old data:
                    replace_indices = np.random.choice(np.arange(len(x_image)), size=dilution_factor, replace=False)
                    
                    for replace_index in replace_indices:
                        #pick randomly which old data:
                        which_task = np.random.choice(np.arange(len(prev_task_data_x_train)))
                        which_image = np.random.randint(0, len(prev_task_data_x_train[which_task])-1)
                        
                        x_image[replace_index] = prev_task_data_x_train[which_task][which_image]
                        y_data[count][replace_index] = prev_task_data_y_train[which_task][which_image]

                    prev_task_data_x_train.append(x_image)
                    prev_task_data_y_train.append(y_data[count])
            
            return x_data, y_data


        X_train_ready, y_train_ready = preprocess_CI(X_train, y_train, dilution_fraction)
        X_test_ready, y_test_ready = preprocess_CI(X_test, y_test, dilution_fraction)

        print('TRAINING: Running {} tasks, each containing {} images'.format(len(X_train), len(X_train[0])))

        print('TESTING: Running {} tasks, each containing {} images'.format(len(X_test), len(X_test[0])))


        train_metrics, test_metrics, raw_results  = run_learning_function(which_model = which_model,
                            which_learningfunction = which_learningfunction,
                            hyperparameters = hyperparameters,
                            X_train= X_train,
                            X_test= X_test,
                            y_train= y_train,
                            y_test = y_test)

        def safe_data(train_metrics, test_metrics, hyperparameters, list_of_classes):
            hyperparameters = {
                'Name Experiment': config['run_name'],
                'Optimizer': hyperparameters[0],
                'Loss function': hyperparameters[5],
                'Number of epochs per task': hyperparameters[12],
                'Learning_rate': optimizer.learning_rate.numpy(),
                'Embedding dimension chunk- and task-embeddings': hyperparameters[1],
                'Use a seperate task embedding model': hyperparameters[7],
                'Initialize task embedding model with zero bias (True), or random bias (False)': hyperparameters[15],
                'Target network dimension': hyperparameters[4],
                'Convolutional layers': hyperparameters[6],
                'A final trainable soft max layer in the target network?': hyperparameters[16],
                'Dropout rate in target network': hyperparameters[9],
                'Number of \'chunks\'': hyperparameters[10],
                'Hypernetwork dimension': hyperparameters[8],
                'Number (initial) classes': hyperparameters[2],
                'Class Incremental Case': hyperparameters[14],
                'Testing while training': hyperparameters[17],
                'L2 regularization strenght': hyperparameters[3],
                'Validation accuracy': hyperparameters[12],
                'Max attempts when using validation accuracy': hyperparameters[11]
            }

            # Change this absolute path: the absolute path to the results folder
            parent_folder_path = r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code\Results'
            file_path = os.path.join(parent_folder_path, 'experiments_results_CIL.csv')

            if os.path.isfile(file_path):
                # Read existing DataFrame
                results_df = pd.read_csv(file_path)
                new_rows = pd.DataFrame({'Hyperparameters': [hyperparameters, hyperparameters], 'Results': [train_metrics, test_metrics], 'Additional classes': [list_of_classes,list_of_classes]})
                results_df = pd.concat([results_df, new_rows], ignore_index=True)
                           
            else:
                results_df = pd.DataFrame(columns=['Hyperparameters', 'Results', 'Additional classes'])
                results_df.loc[0] = [hyperparameters, train_metrics, list_of_classes]
                results_df.loc[1] = [hyperparameters, test_metrics, list_of_classes]

            results_df.to_csv(file_path, index=False)
        
        safe_data(train_metrics, test_metrics, hyperparameters, json.dumps(all_classes))

        if visualize_results:
            run_visualization(raw_results, epochs_per_tasks, amount_of_tasks, str(config['run_name']), False)

        print("END EXPERIMENT: {}".format(config['run_name']))