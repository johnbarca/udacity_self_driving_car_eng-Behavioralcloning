
import os
import cv2
import csv
import numpy as np
import sklearn
import keras
from keras.models import Sequential, Model
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, BatchNormalization, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

# use external GPU, if available
os.environ["CUDA_VISIBLE_DEVICES"]="0";  


# call for plot the lossing during training, only used in Jupyter version
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()


data_dir = 'data/'

from sklearn.model_selection import train_test_split

samples = []

with open(data_dir + 'driving_log.csv') as csv_file:        
    reader = csv.reader(csv_file)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

 
def generator(samples, batch_size=32, measurement_offset=0):
    num_samples = len(samples)
    

    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                center_cam = data_dir + './IMG/' + batch_sample[0].split('/')[-1]
                left_cam =  data_dir + './IMG/' + batch_sample[1].split('/')[-1]
                right_cam =  data_dir + './IMG/' + batch_sample[2].split('/')[-1]

                img = cv2.imread(center_cam)
		#convert back to R
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                measurement = float(batch_sample[3])
                #remove anything that comes from turning round off the track
                if measurement > 0.95: 
                    continue
                
                images.append(img)
                measurements.append(measurement)

                img = cv2.flip(img, 1)
                images.append(img)
                measurements.append(-measurement)
               
                
                # left cam
                img = cv2.imread(left_cam)   
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                measurements.append(measurement + measurement_offset)

                ## flip the image
                img = cv2.flip(img, 1)
                images.append(img)
                measurements.append(-(measurement + measurement_offset))

                # right cam
                img = cv2.imread(right_cam)   
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                measurements.append(measurement - measurement_offset)

                 ## flip the image
                img = cv2.flip(img, 1)
                images.append(img)
                measurements.append(-(measurement - measurement_offset))
                
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size = 64

# try and find the best camera offset parameter
measurement_offset = 0.04

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size, measurement_offset=measurement_offset)
validation_generator = generator(validation_samples, batch_size=batch_size,  measurement_offset=measurement_offset)





def create_model():
    model = Sequential()

    #crop image to lose trees, etc, from top
    model.add(Cropping2D(cropping=((60, 10), (0, 0)), input_shape=(160, 320, 3)))
    #normalize
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    
    #large filtes and strides, no pooling
    model.add(Conv2D(filters = 24, kernel_size = (8, 8), strides = (4, 4), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 36, kernel_size = (5, 5), strides = (2, 2), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 48, kernel_size = (3, 3), strides = (1, 1), activation = 'relu',  padding = 'same'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3),  activation = 'relu'))
    # big reduction in model complexity, without apparent loss of generality
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3),  activation = 'relu'))

    # add dropout to prevent overfitting
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(rate = 0.2))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(1, activation='linear'))    

    return model

#compile the model
model= create_model()
model.compile(loss='mse', optimizer='adam')
model.summary()

# set up early stopping based on val_loss and model checkpointing
earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = True, mode = 'min')
checkpoint = ModelCheckpoint('models/model-measurement-' + str(measurement_offset) + '-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5', verbose=True, monitor='val_loss', save_best_only=True, mode='auto')  


model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/batch_size), 
            epochs=20, callbacks=[earlyStopping, checkpoint], verbose = True)


model.save('model' + str(measurement_offset) + '.h5')







