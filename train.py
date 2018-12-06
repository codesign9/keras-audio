from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

config.max_len = 11
config.buckets = 20


# Save data to array file first
save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels=["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 300#200#50#49#50
config.batch_size = 50#100#200#100#50#100
config.cnn_dropout = 0.2

num_classes = 3

X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)
print(X_test.shape)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
print(y_train_hot.shape)

model = Sequential()
#model.add(Reshape((28,28,1), input_shape=(28,28))) # change from 28x28 into 28x28x1
model.add(Dropout(config.cnn_dropout))
#model.add(Conv2D(32, (3,3), padding='same', activation='relu')) #32
model.add(Conv2D(64, (3,3), padding='same', activation='relu')) #32
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(config.cnn_dropout))
#model.add(Conv2D(12, (3,3), padding='same', activation='relu')) #32
model.add(Conv2D(32, (3,3), padding='same', activation='relu')) #32
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(config.cnn_dropout))
#model.add(Conv2D(12, (3,3), padding='same', activation='relu')) #32
model.add(Conv2D(12, (3,3), padding='same', activation='relu')) #32
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(config.cnn_dropout))
#model.add(Conv2D(12, (3,3), padding='same', activation='relu')) #32
#model.add(Conv2D(4, (3,3), padding='same', activation='relu')) #32
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(config.cnn_dropout))
#model.add(Conv2D(4, (3,3), padding='same', activation='relu')) #32
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten(input_shape=(config.buckets, config.max_len, channels)))
#model.add(Dense(num_classes*10))
model.add(Dense(num_classes*5, activation='relu'))
#model.add(Dense(num_classes*3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
model.fit(X_train, y_train_hot, batch_size=config.batch_size, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])