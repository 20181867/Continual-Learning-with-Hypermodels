import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_continual_learning(train_accuracies, test_accuracies, predictions, true_labels_train, true_labels_test):
    train_metrics = {}
    test_metrics = {}
    # Flatten the predictions and true labels for ease of calculation
    train_predictions_flat = [pred for sublist in [task[0] for task in predictions] for pred in sublist]
    test_predictions_flat = [pred for sublist in [task[1] for task in predictions] for pred in sublist]
    true_labels_train_flat = [label for sublist in true_labels_train for label in sublist]
    true_labels_test_flat = [label for sublist in true_labels_test for label in sublist]
    

    # Train Accuracy
    train_accuracy = np.mean(train_accuracies)
    test_accuracy = np.mean([acc for acc in test_accuracies if acc is not None])

    train_metrics['accuracy'] = train_accuracy
    test_metrics['accuracy'] = test_accuracy

    train_precision = precision_score(true_labels_train_flat, train_predictions_flat, average='macro')

    train_recall = recall_score(true_labels_train_flat, train_predictions_flat, average='macro')

    train_f1 = f1_score(true_labels_train_flat, train_predictions_flat, average='macro')


    train_metrics['precision'] = train_precision
    train_metrics['recall'] = train_recall
    train_metrics['f1_score'] = train_f1

    # Test Precision, Recall, F1-score

    test_precision = precision_score([x for x in true_labels_test_flat if x is not None], test_predictions_flat, average='macro')

    test_recall = recall_score([x for x in true_labels_test_flat if x is not None], test_predictions_flat, average='macro')

    test_f1 = f1_score([x for x in true_labels_test_flat if x is not None], test_predictions_flat, average='macro')

    test_metrics['precision'] = test_precision
    test_metrics['recall'] = test_recall
    test_metrics['f1_score'] = test_f1

    # Confusion matrix
    train_confusion_matrix = confusion_matrix(true_labels_train_flat, train_predictions_flat)

    test_confusion_matrix = confusion_matrix([x for x in true_labels_test_flat if x is not None], test_predictions_flat)

    train_metrics['confusion_matrix'] = train_confusion_matrix
    test_metrics['confusion_matrix'] = test_confusion_matrix

    #forgetting measure:
    divided_array_train = np.array_split(np.array(train_accuracies), len(predictions))
    divided_array_test = np.array_split(np.array(test_accuracies), len(predictions))

    def forgetting_measure(divided_array_of_task_accuracies):
        divided_array = divided_array_of_task_accuracies

        forgetting_measurement = []
        for count, task_data in enumerate(divided_array):
            #filter out None values -> they are excluded from accuracy measurement
            task_data = np.array([x for x in task_data if x is not None])

            if count == 0:
                current_accuracy = np.mean(task_data)
                forgetting_measurement.append(current_accuracy)
            else:
                current_accuracy = np.mean(task_data)
                if previous_accuracy == 0:
                    forgetting_measurement.append(0)
                else:
                    #forgetting_measure = (old_task_performance - new_task_performance) / old_task_performance
                    forgetting_measure = (previous_accuracy - current_accuracy) / previous_accuracy
                    forgetting_measurement.append(forgetting_measure)
            
            previous_accuracy = current_accuracy
        
        return forgetting_measurement
        
    
    train_metrics['forgetting_measure'] = forgetting_measure(divided_array_train)
    test_metrics['forgetting_measure'] = forgetting_measure(divided_array_test)

    return train_metrics, test_metrics


def run_learning_function(which_model, which_learningfunction, hyperparameters, X_train, y_train, X_test, y_test):
    if which_learningfunction[0].upper() == 'BLANCO':
        from Training_blanco.training import run
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], epochs_per_task = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17], which_model= which_model)
    elif which_learningfunction[0].upper() == 'REPLAY':
        from Replay.training_replay import run
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], steps_per_epoch = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17], which_model= which_model,
            #replay specific parameter(s):
            memory_regularization = which_learningfunction[1],
            frugal_learning = which_learningfunction[2],
            match_current_size_CIL = which_learningfunction[3])        
    elif which_learningfunction[0].upper() == 'MAS':
        from MAS.training_MAS import run
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], steps_per_epoch = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17], which_model= which_model,
            #replay specific parameter(s):
            regularization_strenght = which_learningfunction[1],
            momentum = which_learningfunction[2])         
    elif which_learningfunction[0].upper() == 'LWF':
        from LWF.training_LWF import run
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], steps_per_epoch = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17], which_model= which_model,
            #LWF specific parameter(s):
            update_every_epoch = which_learningfunction[1],
            distillation_temperature = which_learningfunction[2])
    elif which_learningfunction[0].upper() == 'GEM':
        from GEM.training_GEM import run 
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], steps_per_epoch = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17], which_model= which_model,
            #GEM specific parameter(s):
            storage_capacity = which_learningfunction[1],
            Average_GEM = which_learningfunction[2],
            margin = which_learningfunction[3])
    elif which_learningfunction[0].upper() == 'AGEM':     
        from AGEM.training_AGEM import run
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], steps_per_epoch = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17], which_model= which_model,
            #AGEM specific parameter(s):
            storage_capacity = which_learningfunction[1],
            Average_GEM = which_learningfunction[2],
            margin = which_learningfunction[3])
    elif which_learningfunction[0].upper() == 'EWC':     
        from EWC.training_EWC import run
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], steps_per_epoch = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17],which_model= which_model,
            #EWC specific parameter(s):
            ewc_lambda = which_learningfunction[1])
    elif which_learningfunction[0].upper() == 'HNET_RGEM':     
        from HNET_RGEM.training_HNET_RGEM import run
        train_accuracies, test_accuracies, predictions, amount_of_classes = run(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, optimizer = hyperparameters[0], embedding_dim = hyperparameters[1],
            num_classes = hyperparameters[2], l2reg = hyperparameters[3], inner_net_dims = hyperparameters[4], loss_fun =hyperparameters[5],convolution_layers= hyperparameters[6], use_unique_task_embedding = hyperparameters[7],
            hnet_hidden_dims = hyperparameters[8], dropout_rate = hyperparameters[9], n_chunks = hyperparameters[10], max_attempts = hyperparameters[11], steps_per_epoch = hyperparameters[12], validation_accuracy = hyperparameters[13],
            class_incremental_case = hyperparameters[14], initialize_TE_with_zero_bias = hyperparameters[15], final_soft_max_layer = hyperparameters[16], training_while_testing = hyperparameters[17],which_model= which_model,
            #HNET_RGEM specific parameter(s):
            storage_capacity = which_learningfunction[1])
    else:
        raise ValueError("Please enter a correct learning function. Options are: Blanco, Replay, MAS, LWF, GEM, AGEM, EWC, HNET_RGEM")
        
    train_metrics, test_metrics = evaluate_continual_learning(train_accuracies, test_accuracies, predictions, y_train, y_test)
    raw_results = (train_accuracies, test_accuracies, predictions, amount_of_classes)
    return train_metrics, test_metrics, raw_results

if __name__ == "__main__":
    raise ValueError('this script is not executable')