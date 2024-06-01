import nni

"'HPO'"

search_space_mel_spec = {
    "filter_size_c1" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},
    "filter_size_c2" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},
    "filter_size_c3" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},
    "filter_size_c4" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},

    "kernel_size_c1" : {'_type': 'choice', '_value': [3,4,5]},
    "kernel_size_c2" : {'_type': 'choice', '_value': [3,4,5]},
    "kernel_size_c3" : {'_type': 'choice', '_value': [3,4,5]},
    "kernel_size_c4" : {'_type': 'choice', '_value': [3,4,5]},

    "numb_units_d1" : {'_type': 'choice', '_value': [100, 256, 416, 832]},
    "numb_units_d2" : {'_type': 'choice', '_value': [100, 256, 416, 832]},

    "learning_rate" : {'_type': 'uniform', '_value': [0.0001, 0.001]}
}

search_space_waveform = {
    "filter_size_c1" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},
    "filter_size_c2" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},
    "filter_size_c3" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},
    "filter_size_c4" : {'_type': 'choice', '_value': [8, 16, 32, 64, 128]},

    "kernel_size_c1" : {'_type': 'choice', '_value': [3,4,5]},
    "kernel_size_c2" : {'_type': 'choice', '_value': [3,4,5]},
    "kernel_size_c3" : {'_type': 'choice', '_value': [3,4,5]},
    "kernel_size_c4" : {'_type': 'choice', '_value': [3,4,5]},

    "numb_units_d1" : {'_type': 'choice', '_value': [100, 256, 416, 832]},
    "numb_units_d2" : {'_type': 'choice', '_value': [100, 256, 416, 832]},

    "learning_rate" : {'_type': 'uniform', '_value': [0.0001, 0.001]}
}




#config the experiment
experiment = nni.Experiment('local')
experiment.config.trial_command = 'python data_processing.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space_mel_spec

#how many trails
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 10

#pick algorithm
experiment.config.tuner.name= 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

#run experiment
experiment.run(8078)
