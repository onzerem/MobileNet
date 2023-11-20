import tensorflow
import numpy as np

from mnet import MobileNet

if __name__ == "__main__":
    print("Testing")
    # load data (cifar10)
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.cifar10.load_data()
    
    # scale images
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    model = MobileNet(input_shape=(32,32,3), n_classes=10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')  

    model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.2)  
    model.evaluate(x_test, y_test, batch_size=64)
    
