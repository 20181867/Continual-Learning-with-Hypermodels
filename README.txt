This code contains the research tool: everything you need to replicate the experiments from the master thesis:

    "Continual Learning with Hypernetworks"

...and more! The research tool allows you to consider many possible (frugal learning) scenarios. 
In total, there are 7*5 = 35 different combinations of models AND learning functions to
try out! With over 25 hyperparameters to tune (e.g. the learning rate and whether or not to add a seperate task embedding model), there is a lot 
to discover.

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

How does it work:

1. Check the dependencies, unzip the files and load a suitable editor.
2. Open the Experiments folder.
3. Open the .yaml file for the scenario of choice (e.g. Task_incremental.yaml).
4. Adjust the parameters for the experiment. Make sure to follow the guidelines as written in the class_incremental.yaml file.
5. Additional experiments may be added, by separating two sets of hyperparameters with a horizontal line: --- 
6. Save the .yaml adjustments!
7. Run the .py file for the specific scenario (e.g. Task_incremental.py).
8. Open the Results folder.
9. Check your results in the .csv and .jpg file!

For questions regarding the research tool:

See the report "Continual Learning with Hypernetworks". If there are still questions, contact y.rooijackers@student.tue.nl
