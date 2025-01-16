import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test, y_test)=cifar10.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

def build_model(input_shape=(32,32,3),num_classes=10):
    model=Sequential([
        layers.Conv2D(16,(3,3),activation='relu',input_shape=input_shape),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(32,(3,3),activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(num_classes,activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model=build_model(input_shape=(32,32,3),num_classes=10)
history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test),batch_size=64)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot()

loss_val,acc_val=model.evaluate(x_test,y_test)
print(f'Accuracy={acc_val: .4f}')
