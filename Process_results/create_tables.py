import pandas as pd
import re
from IPython.display import display


'''SHOW RESULT TABLES'''

#what metric do you want? Choice between 'accuracy', 'precision', 'recall', 'f1_score' and 'forgetting_measure'
metric = 'forgetting_measure'

#'test' or 'train' data?
test_or_train = 'test'

#how many chunks? Choice between '300', '500', '700' or 'average':
chunks = '700' 

#which scenario? Choice between 'CIL', 'DIL', or 'TIL'
scenario = 'TIL' 


#NOTE: make sure the experimental results are saved within the same folder using the orginal name.






















'''Start Code '''
title = 'default'

# Define the columns and rows
columns = ['Dense', 'LSTM', 'GRU', 'ECHO', 'DRNN', 'Average']
rows = ['Blanco', 'EWC', 'LWF', 'MAS', 'GEM', 'A-GEM', 'Replay', 'Average']


# Read the CSV file
try:
    if scenario == 'CIL':
        df = pd.read_csv('experiments_results_CIL.csv')
    elif scenario == 'DIL':
        df = pd.read_csv('experiments_results_DI.csv')
    elif scenario == 'TIL':
        df = pd.read_csv('experiments_results_TI.csv')
except:
    raise FileNotFoundError("Please check the name and location of the result file. This should be within the same folder and using the original name: for example: 'experiments_results_TI.csv'.")




test_df_300_chunk = pd.DataFrame(index=rows, columns=columns)
test_df_500_chunk = pd.DataFrame(index=rows, columns=columns)
test_df_700_chunk = pd.DataFrame(index=rows, columns=columns)
train_df_300_chunk = pd.DataFrame(index=rows, columns=columns)
train_df_500_chunk = pd.DataFrame(index=rows, columns=columns)
train_df_700_chunk = pd.DataFrame(index=rows, columns=columns)

test_df_average = pd.DataFrame(index=rows, columns=columns)
train_df_average = pd.DataFrame(index=rows, columns=columns)


#make place for a 'tem' and a NOT 'tem' value
for dataframe in [test_df_300_chunk, test_df_500_chunk, test_df_700_chunk, train_df_300_chunk, train_df_500_chunk, train_df_700_chunk, test_df_average, train_df_average]:
    for row in rows:
        for col in columns:
            dataframe.at[row, col] = (None, None)



count = 0
#seperate into different datasets
for index, row in df.iterrows():


    for text_chunk in ['300 chunks', '500 chunks', '700 chunks']:

        #seperate into chunks:
        if text_chunk in str(row['Hyperparameters']):

            #seperate into train and test set
            if index % 2 == 0: #train data

                for tem_text in ['no TEM', 'with TEM']:

                    #seperate into TEM and no TEM
                    if tem_text in str(row['Hyperparameters']):

                        for row_text in ['Blanco', 'EWC', 'LWF', 'MAS', 'GEM', 'AGEM', 'REPLAY']:

                            #seperate into rows
                            if row_text in str(row['Hyperparameters']):


                                for column_text in ['Dense', 'LSTM', 'GRU', 'ECHO', 'DRNN']:


                                    #seperate into columns:
                                    if column_text in str(row['Hyperparameters']):


                                        #rename labels:
                                        if row_text == 'AGEM':
                                            neat_text = 'A-GEM'
                                        elif row_text == 'REPLAY':
                                            neat_text = 'Replay'
                                        else:
                                            neat_text = row_text

                                        #get data
                                        if metric == 'forgetting_measure':
                                            match = re.search(r"forgetting_measure': \[([^\]]+)\]", row['Results'])
                                            forgetting_measure = [float(num) for num in match.group(1).split(',')]
                                            result = (forgetting_measure[1]+ forgetting_measure[2])/2
                                        elif metric == 'accuracy':
                                            match = re.search(r"'accuracy':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))
                                        elif metric == 'recall':
                                            match = re.search(r"'recall':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))
                                        elif metric == 'precision':
                                            match = re.search(r"'precision':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))
                                        elif metric == 'f1_score':
                                            match = re.search(r"'f1_score':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))

                                        #get the right value from the right table 
                                        if text_chunk == '300 chunks':
                                            current_value = train_df_300_chunk.at[neat_text, column_text]
                                        elif text_chunk == '500 chunks':
                                            current_value = train_df_500_chunk.at[neat_text, column_text]
                                        elif text_chunk == '700 chunks':
                                            current_value = train_df_700_chunk.at[neat_text, column_text]

                                        # the first item in the tuple is without TEM, the second is with TEM
                                        if tem_text == 'no TEM':
                                            if current_value[0] is None:
                                                new_value = (round(result, 3) , current_value[1])
                                            else:
                                                new_value = ( round((current_value[0] + result)/2, 3) , current_value[1])
                                        if tem_text == 'with TEM':
                                            if current_value[1] is None:
                                                new_value = (current_value[0], round(result, 3))
                                            else:
                                                new_value = (current_value[0], round((current_value[1] + result)/2, 3))

                                        #change the right value from the right table 
                                        if text_chunk == '300 chunks':
                                            train_df_300_chunk.at[neat_text, column_text] = new_value
                                        elif text_chunk == '500 chunks':
                                            train_df_500_chunk.at[neat_text, column_text] = new_value
                                        elif text_chunk == '700 chunks':
                                            train_df_700_chunk.at[neat_text, column_text] = new_value
        
            else: #test data
                for tem_text in ['no TEM', 'with TEM']:

                    #seperate into TEM and no TEM
                    if tem_text in str(row['Hyperparameters']):

                        for row_text in ['Blanco', 'EWC', 'LWF', 'MAS', 'GEM', 'AGEM', 'REPLAY']:

                            #seperate into rows
                            if row_text in str(row['Hyperparameters']):


                                for column_text in ['Dense', 'LSTM', 'GRU', 'ECHO', 'DRNN']:


                                    #seperate into columns:
                                    if column_text in str(row['Hyperparameters']):


                                        #rename labels:
                                        if row_text == 'AGEM':
                                            neat_text = 'A-GEM'
                                        elif row_text == 'REPLAY':
                                            neat_text = 'Replay'
                                        else:
                                            neat_text = row_text

                                        #get data
                                        if metric == 'forgetting_measure':
                                            match = re.search(r"forgetting_measure': \[([^\]]+)\]", row['Results'])
                                            forgetting_measure = [float(num) for num in match.group(1).split(',')]
                                            result = (forgetting_measure[1]+ forgetting_measure[2])/2
                                        elif metric == 'accuracy':
                                            match = re.search(r"'accuracy':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))
                                        elif metric == 'recall':
                                            match = re.search(r"'recall':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))
                                        elif metric == 'precision':
                                            match = re.search(r"'precision':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))
                                        elif metric == 'f1_score':
                                            match = re.search(r"'f1_score':\s*([\d.]+)", row['Results'])
                                            result = float(match.group(1))

                                        #get the right value from the right table 
                                        if text_chunk == '300 chunks':
                                            current_value = test_df_300_chunk.at[neat_text, column_text]
                                        elif text_chunk == '500 chunks':
                                            current_value = test_df_500_chunk.at[neat_text, column_text]
                                        elif text_chunk == '700 chunks':
                                            current_value = test_df_700_chunk.at[neat_text, column_text]

                                        # the first item in the tuple is without TEM, the second is with TEM
                                        if tem_text == 'no TEM':
                                            if current_value[0] is None:
                                                new_value = (round(result, 3) , current_value[1])
                                            else:
                                                new_value = ( round((current_value[0] + result)/2, 3) , current_value[1])
                                        if tem_text == 'with TEM':
                                            if current_value[1] is None:
                                                new_value = (current_value[0], round(result, 3))
                                            else:
                                                new_value = (current_value[0], round((current_value[1] + result)/2, 3))

                                        #change the right value from the right table 
                                        if text_chunk == '300 chunks':
                                            test_df_300_chunk.at[neat_text, column_text] = new_value
                                        elif text_chunk == '500 chunks':
                                            test_df_500_chunk.at[neat_text, column_text] = new_value
                                        elif text_chunk == '700 chunks':
                                            test_df_700_chunk.at[neat_text, column_text] = new_value


    #seperate into train and test set
    if index % 2 == 0: #train data

        for tem_text in ['no TEM', 'with TEM']:

            #seperate into TEM and no TEM
            if tem_text in str(row['Hyperparameters']):

                for row_text in ['Blanco', 'EWC', 'LWF', 'MAS', 'GEM', 'AGEM', 'REPLAY']:

                    #seperate into rows
                    if row_text in str(row['Hyperparameters']):


                        for column_text in ['Dense', 'LSTM', 'GRU', 'ECHO', 'DRNN']:


                            #seperate into columns:
                            if column_text in str(row['Hyperparameters']):


                                #rename labels:
                                if row_text == 'AGEM':
                                    neat_text = 'A-GEM'
                                elif row_text == 'REPLAY':
                                    neat_text = 'Replay'
                                else:
                                    neat_text = row_text

                                #get data
                                if metric == 'forgetting_measure':
                                    match = re.search(r"forgetting_measure': \[([^\]]+)\]", row['Results'])
                                    forgetting_measure = [float(num) for num in match.group(1).split(',')]
                                    result = (forgetting_measure[1]+ forgetting_measure[2])/2
                                elif metric == 'accuracy':
                                    match = re.search(r"'accuracy':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))
                                elif metric == 'recall':
                                    match = re.search(r"'recall':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))
                                elif metric == 'precision':
                                    match = re.search(r"'precision':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))
                                elif metric == 'f1_score':
                                    match = re.search(r"'f1_score':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))

                                #get the right value from the right table 
                                current_value = train_df_average.at[neat_text, column_text]

                                # the first item in the tuple is without TEM, the second is with TEM
                                if tem_text == 'no TEM':
                                    if current_value[0] is None:
                                        new_value = (round(result, 3) , current_value[1])
                                    else:
                                        new_value = ( round((current_value[0] + result)/2, 3) , current_value[1])
                                if tem_text == 'with TEM':
                                    if current_value[1] is None:
                                        new_value = (current_value[0], round(result, 3))
                                    else:
                                        new_value = (current_value[0], round((current_value[1] + result)/2, 3))

                                #change the right value from the right table 
                                train_df_average.at[neat_text, column_text] = new_value



    #seperate into train and test set
    else: #test data

        for tem_text in ['no TEM', 'with TEM']:

            #seperate into TEM and no TEM
            if tem_text in str(row['Hyperparameters']):

                for row_text in ['Blanco', 'EWC', 'LWF', 'MAS', 'GEM', 'AGEM', 'REPLAY']:

                    #seperate into rows
                    if row_text in str(row['Hyperparameters']):


                        for column_text in ['Dense', 'LSTM', 'GRU', 'ECHO', 'DRNN']:


                            #seperate into columns:
                            if column_text in str(row['Hyperparameters']):


                                #rename labels:
                                if row_text == 'AGEM':
                                    neat_text = 'A-GEM'
                                elif row_text == 'REPLAY':
                                    neat_text = 'Replay'
                                else:
                                    neat_text = row_text

                                #get data
                                if metric == 'forgetting_measure':
                                    match = re.search(r"forgetting_measure': \[([^\]]+)\]", row['Results'])
                                    forgetting_measure = [float(num) for num in match.group(1).split(',')]
                                    result = (forgetting_measure[1]+ forgetting_measure[2])/2
                                elif metric == 'accuracy':
                                    match = re.search(r"'accuracy':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))
                                elif metric == 'recall':
                                    match = re.search(r"'recall':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))
                                elif metric == 'precision':
                                    match = re.search(r"'precision':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))
                                elif metric == 'f1_score':
                                    match = re.search(r"'f1_score':\s*([\d.]+)", row['Results'])
                                    result = float(match.group(1))

                                #get the right value from the right table 
                                current_value = test_df_average.at[neat_text, column_text]

                                # the first item in the tuple is without TEM, the second is with TEM
                                if tem_text == 'no TEM':
                                    if current_value[0] is None:
                                        new_value = (round(result, 3) , current_value[1])
                                    else:
                                        new_value = ( round((current_value[0] + result)/2, 3) , current_value[1])
                                if tem_text == 'with TEM':
                                    if current_value[1] is None:
                                        new_value = (current_value[0], round(result, 3))
                                    else:
                                        new_value = (current_value[0], round((current_value[1] + result)/2, 3))

                                #change the right value from the right table 
                                test_df_average.at[neat_text, column_text] = new_value

#calculate and add averages
average_Dense = (0, 0)
average_LSTM = (0, 0)
average_GRU = (0, 0)
average_ECHO = (0, 0)
average_DRNN = (0, 0)

average_Blanco = (0, 0)
average_EWC = (0, 0)
average_LWF = (0, 0)
average_MAS = (0, 0)
average_GEM = (0, 0)
average_AGEM = (0, 0)
average_Replay = (0, 0)

for dataframe in [test_df_300_chunk, test_df_500_chunk, test_df_700_chunk, train_df_300_chunk, train_df_500_chunk, train_df_700_chunk, test_df_average, train_df_average]:
    
    for index, row in dataframe.iterrows():

        for column, value in row.items():

            try:
                if column == 'Dense':
                    average_Dense = (round((average_Dense[0] + value[0])/2,2), round((average_Dense[1]+ value[1])/2,2))
                    dataframe.at['Average', column] = average_Dense
                elif column == 'LSTM':
                    average_LSTM = (round((average_LSTM[0] + value[0])/2,2), round((average_LSTM[1]+ value[1])/2,2))
                    dataframe.at['Average', column] = average_LSTM
                elif column == 'GRU':
                    average_GRU = (round((average_GRU[0] + value[0])/2,2), round((average_GRU[1]+ value[1])/2,2))
                    dataframe.at['Average', column] = average_GRU
                elif column == 'ECHO':
                    average_ECHO = (round((average_ECHO[0] + value[0])/2,2), round((average_ECHO[1]+ value[1])/2,2))
                    dataframe.at['Average', column] = average_ECHO
                elif column == 'DRNN':
                    average_DRNN = (round((average_DRNN[0] + value[0])/2,2), round((average_DRNN[1]+ value[1])/2,2))
                    dataframe.at['Average', column] = average_DRNN
                                                                            
                if index == 'Blanco':
                    average_Blanco = (round((average_Blanco[0]+ value[0])/2,2), round((average_Blanco[1]+ value[1])/2,2))
                    dataframe.at[index, 'Average'] = average_Blanco
                elif index == 'EWC':
                    average_EWC = (round((average_EWC[0]+ value[0])/2,2), round((average_EWC[1]+ value[1])/2,2))
                    dataframe.at[index, 'Average'] = average_EWC
                elif index == 'LWF':
                    average_LWF = (round((average_LWF[0]+ value[0])/2,2), round((average_LWF[1]+ value[1])/2,2))
                    dataframe.at[index, 'Average'] = average_LWF
                elif index == 'MAS':
                    average_MAS = (round((average_MAS[0]+ value[0])/2,2), round((average_MAS[1]+ value[1])/2,2))
                    dataframe.at[index, 'Average'] = average_MAS
                elif index == 'GEM':
                    average_GEM = (round((average_GEM[0]+ value[0])/2,2), round((average_GEM[1]+ value[1])/2,2))
                    dataframe.at[index, 'Average'] = average_GEM
                elif index == 'A-GEM':
                    average_AGEM = (round((average_AGEM[0]+ value[0])/2,2), round((average_AGEM[1]+ value[1])/2,2))
                    dataframe.at[index, 'Average'] = average_AGEM
                elif index == 'Replay':
                    average_Replay = (round((average_Replay[0]+ value[0])/2,2), round((average_Replay[1]+ value[1])/2,2))
                    dataframe.at[index, 'Average'] = average_Replay
            except TypeError: #occurs if None value
                continue
    dataframe.at['Average', 'Average'] = ('no', 'meaning')        




#print and save wanted results
if test_or_train == 'train':
    if chunks == '300':
        display(train_df_300_chunk)
        selected_dataset = train_df_300_chunk
    elif chunks == '500':
        display(train_df_500_chunk)
        selected_dataset = train_df_500_chunk
    elif chunks == '700':
        display(train_df_700_chunk)
        selected_dataset = train_df_700_chunk
    elif chunks == 'average':
        display(train_df_average)
        selected_dataset = train_df_average
    else:
        raise ValueError('please define a correct amount of chunks (e.g. \'300\' or \'average\')')
elif test_or_train == 'test':
    if chunks == '300':
        display(test_df_300_chunk)
        selected_dataset = test_df_300_chunk
    elif chunks == '500':
        display(test_df_500_chunk)
        selected_dataset = test_df_500_chunk
    elif chunks == '700':
        display(test_df_700_chunk)
        selected_dataset = test_df_700_chunk
    elif chunks == 'average':
        display(test_df_average)
        selected_dataset = test_df_average
    else:
        raise ValueError('please define a correct amount of chunks (e.g. \'300\' or \'average\')')
else:
    raise ValueError('please define whether to use the train or test results. For example: \'train\' ')
        

if title == 'default':

    if metric == 'f1_score':
        metric_text = 'F1 Score'
    elif metric == 'forgetting_measure':
        metric_text = 'Forgetting Measure'
    elif metric == 'accuracy':
        metric_text = 'Accuracy'
    elif metric == 'precision':
        metric_text = 'Precision'
    elif metric == 'recall':
        metric_text = 'Recall'
    
    if chunks != 'average':
        title_text = metric_text + ' for ' + scenario
        subtitle_text = 'with ' + chunks + ' Chunks'
    else:
        title_text = metric_text + ' for ' + scenario
        subtitle_text = 'as average of 300, 500 and 700 chunks'

# Function to format tuples with a thinner dotted dark blue line
def format_tuple(t):
    return f"""
    <div style='line-height: 0.3; font-size: 8px;'>
        <div style='line-height: 1;'>
            {t[0]}<br>
        </div>
        <span style='display: inline-block; width: 80%; border-bottom: 1px dotted darkblue; vertical-align: middle;'></span><br>
        <div style='line-height: 1.5;'>
            {t[1]}
        </div>
    </div>
    """
# Apply the formatting function to each element in the DataFrame
formatted_df = selected_dataset.applymap(format_tuple)

# Add title and caption using pandas styling
styled_df = formatted_df.style.set_caption(title_text) \
                             .set_table_styles([
                                 # Style for the caption
                                 {'selector': 'caption',
                                  'props': [('caption-side', 'top'),
                                            ('font-size', '20px'),
                                            ('font-weight', 'bold'),
                                            ('text-align', 'center'),
                                            ('margin-bottom', '20px'),
                                            ('text-decoration', 'underline')]},
                                 # Style for the table, adding borders
                                 {'selector': '',
                                  'props': [('border', '3px solid #8a90a6'),  # Dark navy blue
                                            ('border-collapse', 'collapse'),
                                            ('font-family', 'Latin Modern Roman, serif'),
                                            ('font-size', '12px'),
                                            ('color', 'black')]},
                                 # Style for table cells, adding borders and compactness
                                 {'selector': 'td, th',
                                  'props': [('border', '2px solid #c1cae8'),  # Lighter blue
                                            ('padding', '3px'),  # Reduced padding for compactness
                                            ('text-align', 'center'),
                                            ('font-family', 'Latin Modern Roman, serif'),
                                            ('font-size', '12px'),
                                            ('color', 'black'),
                                            ('font-weight', 'normal')]}
                             ])

# Additional text
additional_text = "Additional text anywhere on the HTML file."

# HTML content with additional text
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results</title>
    <style>
        .table-text {{
            font-size: 10px;
            font-weight: normal;
            text-align: center;
            margin-top: -28%; /* Adjust margin to push the text down */
            margin-left: -71%; /* Adjust margin to push the text down */
            position: relative; /* Set position to relative */
            z-index: 1; /* Ensure the text appears above other content */
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div> {styled_df.to_html(escape=False)} </div>
    <div class="table-text">{subtitle_text}</div>
</body>
</html>
"""

html_file = "result_table.html"
with open(html_file, "w") as file:
    file.write(html_content)

import webbrowser

# Open the HTML file in the default web browser
webbrowser.open("result_table.html")



