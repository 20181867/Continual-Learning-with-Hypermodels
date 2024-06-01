import librosa as lr 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class load_data:
    def __init__(self, DI):
        #replace this with the absolute path to your data
        if DI:
            self.DI = True
            self.dataset= r"C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code\dataset\data\augmented_dataset_verynoisy\augmented_dataset_verynoisy"
        else:
            self.DI = False
            self.dataset= r"C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code\dataset\data\augmented_dataset\augmented_dataset"

    def count(self, print_):
        size=[]
        for file in os.listdir(self.dataset):
            size.append(len(os.listdir(os.path.join(self.dataset,file))))
        if print_:
            print(pd.DataFrame(size,columns=['Number Of Samples'],index=os.listdir(self.dataset)))  
        else:
            return pd.DataFrame(size,columns=['Number Of Samples'],index=os.listdir(self.dataset))

    def plot_data_distribution(self):
        total_sum = self.count(False)
        plt.figure(figsize=(10,10))
        plt.pie(x='Number Of Samples',labels=os.listdir(self.dataset),autopct ='%1.1f%%',data=total_sum)
        plt.title('Distribution Of Data In Train',fontsize=20)
        plt.show()

    def load(self, subset: bool, subset_list: list, sr): 
        data=[]
        label=[]
        sample=[]
        if subset:
            if self.DI:
                print("loading {} words from NOISY data".format(len(subset_list)))
            else: 
                print("loading {} words".format(len(subset_list)))
        else:
            print("loading 30 words")

        for file in os.listdir(self.dataset):
            if subset:
                if file not in subset_list:
                    continue
            path_=os.path.join(self.dataset,file)
            print("loading {}".format(file))
            for fil in tqdm(os.listdir(path_)):
                data_contain,sample_rate=lr.load(os.path.join(path_,fil) ,sr=sr)
                data.append(data_contain)
                sample.append(sample_rate)
                label.append(file)
        return data,label,sample
    
    def forge_dataframe(self, label, sample):
        df=pd.DataFrame()
        df['Label'],df['sample']=label,sample
        return df
    
    def Mel(self, data, sr):
        mel_spec = lr.feature.melspectrogram(y=data, sr=sr)
        return mel_spec
    
    def preproces_data_with_Mel(self, data, sr, label):
        Mel_data = []
        print("processing {} samples/items ('{}it') into mel spectograms".format(len(data), len(data)))
        for count, index in tqdm(enumerate (data)):
            Mel_data.append((label[count], self.Mel(index, sr[count])))
        return Mel_data
        
    def plot_mel(self, mel_spec,label,sr, conversion):
        # Convert to decibel scale
        mel_spec_db = lr.power_to_db(mel_spec, ref=np.max)
        # Visualize Mel-spectrogram
        plt.figure(figsize=(10, 4))
        lr.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        print(conversion)
        plt.title('Mel-spectrogram of ' + str(label))

    def convert_labels(self, labels):
        conversion={}
        x=0
        for i in pd.unique(labels):
            conversion[i]=x
            x+=1
        for count, label in enumerate(labels):
            labels[count] = conversion.get(label)
        return labels, conversion
        

"'Load Data'"
def load_the_data(preprocessing, subsetlist, sr, DI):
    data_loader = load_data(DI)

    if subsetlist is not None:
        data,label,sample= data_loader.load(True, subsetlist, sr)
    else:
        data,label,sample= data_loader.load(False, subsetlist, sr)
    
    #dataframe = data_loader.forge_dataframe(label, sample)
    label, conversion = data_loader.convert_labels(label)

    if preprocessing == 'MEL':
        Mel_data = data_loader.preproces_data_with_Mel(data, sample, label)

        #for plotting MEL spectograms:
        #data_loader.plot_mel(Mel_data[0][1],Mel_data[0][0],sr, conversion)

    "'Split Data'"
    #for waveform
    if preprocessing != 'MEL':
        data=np.array(data).reshape(-1,16000,1)
        label=np.array(label)
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=44, shuffle =True)

        return X_train, X_test, y_train, y_test, conversion


    #for Mel spectogram
    if preprocessing == 'MEL':
        labels_mel = []
        data_mel = []
        for item in Mel_data:
            labels_mel.append(item[0])
            data_mel.append(item[1])

        data_mel = np.array(data_mel)

        #input shape of one image
        input_shape = data_mel.shape
        input_shape = (input_shape[1], input_shape[2], 1) #128, 32, 1    

        X_train_mel, X_test_mel, y_train_mel, y_test_mel = train_test_split(data_mel, labels_mel, test_size=0.1, random_state=44, shuffle =True)
        return X_train_mel, X_test_mel, y_train_mel, y_test_mel, conversion, input_shape

