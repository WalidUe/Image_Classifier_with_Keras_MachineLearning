from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm

model = Sequential()#Create a model (sequential type) to be able to add layer in order
#Adding the first convolution to output a feature map
#filters: output 32 features
#kernel_size: 3x3 kernel (filter matrix) used to calculate output features
#input_shape: each image is 32x32x3
#activation: relu activation for each of the operations as it produces the best results
#padding: adds padding to the input values in the kernel to make sure that the mas value is 3
model.add(Conv2D(filter=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same',
                 kernel_constraint=maxnorm(3)))
#Add the max pool layer to decrease the image size from 32x32 to 16x16
#pool_size: finds the max value in each 2x2 section of the input
model.add(MaxPooling2D(pool_size=(2, 2)))
#Flatten layer converts a matrix into a 1 dimensional array
model.add(Flatten())
#First dense layer to create the actual prediction network
#units: 512 neurons at this layer, increase for greater accuracy, decrease for faster train speed
#activation: relu because it works so well
model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
#Dropout layer to ignore some neurons during training which improves model reliability
#rate:0.5 means half neurons dropped
model.add(Dropout(rate=0.5))##drop neurons by half
#Final dense layer used to produce output for each of our 10 categories
#units: 10 categories so 10 output units
#activation: softmax because we are calculating probabilities for each of the 10 categories (not as clear as 0 or 1)
model.add(Dense(units=10, activation='softmax'))##since we only have 10 categories






