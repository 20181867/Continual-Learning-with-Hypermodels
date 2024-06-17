This code contains the research tool: everything you need to replicate the experiments from the master thesis:

    "Continual Learning with Hypernetworks"

...and much more! The research tool allows you to consider many possible (frugal learning) scenarios and (dis)prove your own hypothesis!

In total, there are 35 different combinations of hypermodels AND learning functions to try out!
With over 25 hyperparameters to tune (e.g. the learning rate and whether or not to add a seperate task embedding model), there is a lot to discover!

To prevent dependency issues, this certainly works:

tensorflow version: 2.12.0
pandas version: 2.0.1
yaml version: 6.0.1
nni version: 3.0
matplotlib version: 3.4.3
sklearn version: 1.1.2
numpy version: 1.26.4
tqdm version: 4.64.1

The code is generated using Python 3.10.

Setup:

1. download the SSC from:
    https://www.kaggle.com/datasets/jbuchner/synthetic-speech-commands-dataset
    (2GB)
2. Unzip the folder named 'data', containing the folders 'augmented_dataset' and 'augmented_dataset_verynoisy'
3. Add this unzipped folder in the (initially empty) folder 'data'
4. Check the dependencies
5. Use a suitable editor, for example, VSC

How does it work:

1. Open the Experiments folder.
2. Open the .yaml file for the scenario of choice (e.g. Task_incremental.yaml).
3. Adjust the parameters for the experiment. Make sure to follow the guidelines as written in the class_incremental.yaml file.
4. Additional experiments may be added, by separating two sets of hyperparameters with a horizontal line: --- 
5. Save the .yaml adjustments!
6. Run the .py file for the specific scenario (e.g. Task_incremental.py).
7. Open the Results folder.
8. Check your results in the .csv and .jpg file!

For a demonstration video, see:

https://youtu.be/dDXlg9s5q_s

For questions regarding the research tool:

contact me at yvarrooijackers@gmail.com

See the report "Continual Learning with Hypernetworks". If there are still questions, contact y.rooijackers@student.tue.nl
