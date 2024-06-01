import random
import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
from Preprocessing.load_data import load_the_data
import tensorflow as tf
import numpy as np
import copy
from Experiments.learning_functions import run_learning_function
import pandas as pd
import yaml
import os
import json
from Training_blanco.training import run_visualization

if __name__ == "__main__":
        
    '''       HYPERPARAMETERS        '''

    configs = yaml.safe_load_all(open('domain_incremental.yaml', 'r'))
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
        num_classes = int(config['Number_of_classes'])
        class_incremental_case = eval(config['Class_Incremental_Case'])
        l2reg = float(config['L2_regularization_strenght'])
        validation_accuracy = float(config['Validation_accuracy'])
        max_attempts = int(config['Max_attempts_when_using_validation_accuracy'])
        sr = int(config['Sampel_rate_sound_data'])
        test_data_per_task = int(config['Test_images_per_task'])
        testing_while_training = config['Testing_while_training']

        which_learningfunction = eval(config['Which_learningfunction'])
        which_model = eval(config['Which_Model'])

        # Hyperparameter that remained from the previous experiments, see report:
        preprocessing = 'MEL'

        # Parameters for Task Incremental Learning
        n_domain_shifts = int(config['Amount_of_domain_shifts'])

        # Parameters for running experiments
        append_results = config['Add_results_to_previous_experiment_results']
        visualize_results = config['Visualize_results']


        '''       BEGIN CODE        '''
        
        hyperparameters = (optimizer, embedding_dim, num_classes, l2reg, inner_net_dims, loss_fun, convolution_layers, use_unique_task_embedding, hnet_hidden_dims, dropout_rate, n_chunks, max_attempts, epochs_per_tasks, validation_accuracy, class_incremental_case, initialize_TE_with_zero_bias, final_soft_max_layer, testing_while_training)

        #domain incremental learning consists of domain shifts. Every domain shift block consists of 3 tasks:
        #           the first task from the original (non noisy) domain
        #           the second task from the noisy domain
        #           the third task from the original (non noisy) domain
        amount_of_tasks = n_domain_shifts*3


        classes = [ 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
                        'house', 'left', 'marvel', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                        'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

        # Change this absolute path: the absolute path to the results folder
        parent_folder_path = r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code\Results'
        file_path = os.path.join(parent_folder_path, 'experiments_results_DI.csv')
        
        if not append_results:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        def get_task_data (classes, amount_of_tasks, num_classes,  sr, preprocessing, epochs_per_tasks, test_data_per_task):
            X_train, X_test, y_train, y_test = [], [], [], []
            conversions = []
            selected_classes = random.sample(classes, num_classes)
            list_of_classes = {'Every task  has these classes:': selected_classes}

            #loop over every domain shift
            for i in range(0, int(amount_of_tasks/3)):
            
                #loop over every task within this domain shift block
                for j in range(0, 3):
                    #get task data from original domain
                    if j == 0 or j == 2:
                        X_train_this_task, X_test_this_task, y_train_this_task, y_test_this_task, conversion_this_task, _ = load_the_data(preprocessing, selected_classes,sr, False)
                        X_train_this_task = np.expand_dims(X_train_this_task[:epochs_per_tasks], axis=-1)
                        y_train_this_task = y_train_this_task[:epochs_per_tasks]
                        X_test_this_task = np.expand_dims(X_test_this_task[:test_data_per_task], axis=-1)
                        y_test_this_task = y_test_this_task[:test_data_per_task]

                        X_train.append(X_train_this_task)
                        X_test.append(X_test_this_task)
                        y_train.append(y_train_this_task)
                        y_test.append(y_test_this_task)
                        conversions.append(conversion_this_task)
                    #get task data from alternative (noisy) domain
                    elif j == 1:
                        X_train_this_task, X_test_this_task, y_train_this_task, y_test_this_task, conversion_this_task, _ = load_the_data(preprocessing, selected_classes,sr, True)
                        X_train_this_task = np.expand_dims(X_train_this_task[:epochs_per_tasks], axis=-1)
                        y_train_this_task = y_train_this_task[:epochs_per_tasks]
                        X_test_this_task = np.expand_dims(X_test_this_task[:test_data_per_task], axis=-1)
                        y_test_this_task = y_test_this_task[:test_data_per_task]

                        X_train.append(X_train_this_task)
                        X_test.append(X_test_this_task)
                        y_train.append(y_train_this_task)
                        y_test.append(y_test_this_task)
                        conversions.append(conversion_this_task)

            return X_train, y_train, X_test, y_test, conversions, list_of_classes
        

        X_train, y_train, X_test, y_test, conversions, list_of_classes = get_task_data(classes, amount_of_tasks, num_classes, sr, preprocessing, epochs_per_tasks, test_data_per_task)

        print('TRAINING: Running {} tasks, each containing {} images'.format(len(X_train), len(X_train[0])))

        print('TESTING: Running {} tasks, each containing {} images'.format(len(X_test), len(X_test[0])))


        train_metrics, test_metrics, raw_results = run_learning_function(which_model = which_model,
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
                'Max attempts when using validation accuracy': hyperparameters[11],
            }

            # Change this absolute path: the absolute path to the results folder
            parent_folder_path = r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code\Results'
            file_path = os.path.join(parent_folder_path, 'experiments_results_DI.csv')

            if os.path.isfile(file_path):
                # Read existing DataFrame
                results_df = pd.read_csv(file_path)
                new_rows = pd.DataFrame({'Hyperparameters': [hyperparameters, hyperparameters], 'Results': [train_metrics, test_metrics], 'Classes in task': [list_of_classes,list_of_classes]})
                results_df = pd.concat([results_df, new_rows], ignore_index=True)
                           
            else:
                results_df = pd.DataFrame(columns=['Hyperparameters', 'Results', 'Classes in task'])
                results_df.loc[0] = [hyperparameters, train_metrics, list_of_classes]
                results_df.loc[1] = [hyperparameters, test_metrics, list_of_classes]

            results_df.to_csv(file_path, index=False)
        
        safe_data(train_metrics, test_metrics, hyperparameters, json.dumps(list_of_classes))

        if visualize_results:
            run_visualization(raw_results, epochs_per_tasks, amount_of_tasks, str(config['run_name']), True)

        print("END EXPERIMENT: {}".format(config['run_name']))
