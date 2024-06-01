import sys
sys.path.append(r'C:\Users\20181867\Documents\Schoolbestanden\Thesis\Code')
import numpy as np
import keras
import tensorflow as tf
import nni 
from Preprocessing.load_data import load_the_data
from tensorflow.keras.optimizers import Adam


class SendMetrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 'self.model' is automatically set by Keras during training
        nni.report_intermediate_result(logs['val_accuracy'])


class model_waveform:
    def __init__(self, conversion, sr, params):
        num_class=len(conversion.keys())
        self.model=keras.Sequential()
        #block 1
        self.model.add(keras.layers.Conv1D(filters=params['filter_size_c1'], kernel_size=params['kernel_size_c1'],activation=tf.nn.relu,input_shape=(sr,1)))
        self.model.add(keras.layers.MaxPooling1D(3))
        self.model.add(keras.layers.Dropout(.3))
        #block 2
        self.model.add(keras.layers.Conv1D(filters=params['filter_size_c2'], kernel_size=params['kernel_size_c2'],activation=tf.nn.relu))
        self.model.add(keras.layers.MaxPooling1D(3))
        self.model.add(keras.layers.Dropout(.3))
        #block 3
        self.model.add(keras.layers.Conv1D(filters=params['filter_size_c3'], kernel_size=params['kernel_size_c3'],activation=tf.nn.relu))
        self.model.add(keras.layers.MaxPooling1D(3))
        self.model.add(keras.layers.Dropout(.3))
        #block 4
        self.model.add(keras.layers.Conv1D(filters=params['filter_size_c4'], kernel_size=params['kernel_size_c4'],activation=tf.nn.relu))
        self.model.add(keras.layers.MaxPooling1D(3))
        self.model.add(keras.layers.Dropout(.3))
        self.model.add(keras.layers.Flatten())
        #dense layers
        self.model.add(keras.layers.Dense(params['numb_units_d1'],activation=tf.nn.relu))
        self.model.add(keras.layers.Dropout(.3))
        self.model.add(keras.layers.Dense(params['numb_units_d2'],activation=tf.nn.relu))
        self.model.add(keras.layers.Dropout(.3))
        self.model.add(keras.layers.Dense(num_class,activation=tf.nn.softmax))

    def summary(self):
        print(self.model.summary())
    
    def compile(self, optimizer, loss, metric):
        self.model.compile(optimizer =optimizer,loss=loss,metrics=metric)
    
    def model_fit(self, x_train, y_train, epochs, batch_size, X_test, y_test):
        send_metrics_callback = SendMetrics()
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[send_metrics_callback])
        
    def save_model(self, name, ):
        self.model.save(str(name)+'.h5', include_optimizer=True)

    def evaluate(self, X_test_mel, y_test_mel):
        loss,acc = self.model.evaluate(X_test_mel,y_test_mel)
        return loss,acc


class model_mel_spectogram:
    def __init__(self, conversion, input_shape, params):
        num_class=len(conversion.keys())
        self.model = keras.Sequential()

        # Convolutional blocks
        self.model.add(keras.layers.Conv2D(filters = params['filter_size_c1'], kernel_size=params['kernel_size_c1'], activation=tf.nn.relu, input_shape=input_shape))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Dropout(0.3))

        self.model.add(keras.layers.Conv2D(filters=params['filter_size_c2'], kernel_size= params['kernel_size_c2'], activation=tf.nn.relu))
        dropoutlayer_2 = keras.layers.Dropout(0.3)
        self.model.add(dropoutlayer_2)

        self.model.add(keras.layers.Conv2D(filters=params['filter_size_c3'], kernel_size=params['kernel_size_c3'], activation=tf.nn.relu))
        self.model.add(keras.layers.Dropout(0.3))

        self.model.add(keras.layers.Conv2D(filters=params['filter_size_c4'], kernel_size=params['kernel_size_c4'], activation=tf.nn.relu))
        self.model.add(keras.layers.Dropout(0.3))

        # Flatten layer
        self.model.add(keras.layers.Flatten())

        # Dense layers
        self.model.add(keras.layers.Dense(params['numb_units_d1'], activation=tf.nn.relu))
        self.model.add(keras.layers.Dropout(0.3))

        self.model.add(keras.layers.Dense(params['numb_units_d2'], activation=tf.nn.relu))
        self.model.add(keras.layers.Dropout(0.3))

        # Output layer
        self.model.add(keras.layers.Dense(num_class, activation=tf.nn.softmax))


    def summary(self):
        print(self.model.summary())
    
    def compile(self, optimizer, loss, metric):
        self.model.compile(optimizer =optimizer,loss=loss,metrics=metric)
    
    def model_fit(self, x_train, y_train, epochs, batch_size, X_test, y_test):

        send_metrics_callback = SendMetrics()
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[send_metrics_callback])
        
    def save_model(self, name):
        self.model.save(str(name)+'.h5', include_optimizer=True)

    def evaluate(self, X_test_mel, y_test_mel):
        loss,acc = self.model.evaluate(X_test_mel,y_test_mel)
        return loss,acc




"'Train Models'"
def run(params, use_presaved_model, preprocessing, batch_size, sr, subsetlist):
    #waveform model
    if preprocessing != 'MEL':
        if not use_presaved_model:
            X_train, X_test, y_train, y_test, conversion = load_the_data(preprocessing, subsetlist, 16000, False)
            
            wavefrom_model = model_waveform(conversion, sr, params)
            optimizer, loss, metric, epochs = Adam(learning_rate=params['learning_rate']), 'sparse_categorical_crossentropy', ['accuracy'], 10
            wavefrom_model.compile(optimizer, loss, metric)
            wavefrom_model.model_fit(X_train, y_train, epochs, batch_size, X_test, y_test)
            wavefrom_model.save_model('wavefrom_model')

            loss,acc= wavefrom_model.evaluate(X_test,y_test)
            nni.report_final_result(acc)

        else:
            wavefrom_model = tf.keras.models.load_model('wavefrom_model.h5')

    #melspec model
    if preprocessing == 'MEL':
        if not use_presaved_model:
            X_train, X_test, y_train, y_test, conversion, input_shape = load_the_data(preprocessing, subsetlist, 16000, False)

            melspec_model = model_mel_spectogram(conversion, input_shape, params)

            learning_rate = params['learning_rate']
            optimizer, loss, metric, epochs = Adam(learning_rate=learning_rate), 'sparse_categorical_crossentropy', ['accuracy'], 10
            melspec_model.compile(optimizer, loss, metric)
            
            #preparing shapes
            X_train, X_test = np.array(X_train), np.array(X_test)
            X_train, X_test= X_train.reshape((len(X_train), 128, 32, 1)), X_test.reshape((len(X_test), 128, 32, 1))
            y_train, y_test = np.array(y_train), np.array(y_test)

            melspec_model.model_fit(X_train, y_train, epochs, batch_size, X_test, y_test)
            melspec_model.save_model('melspec_model')

            loss,acc= melspec_model.evaluate(X_test,y_test)
            nni.report_final_result(acc)

        else:
            melspec_model = tf.keras.models.load_model('melspec_model.h5')

if __name__ == '__main__':
    preprocessing = 'MEL'
    batch_size = 200
    use_presaved_model = False
    subsetlist = ['bed', 'cat', 'bird']
    sr = 16000
    try:
        params = nni.get_next_parameters()
        run(params, use_presaved_model, preprocessing, batch_size, sr, subsetlist)
    except Exception:
        raise


