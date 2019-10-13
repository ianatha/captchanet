import os
import numpy as np
from PIL import Image
dataset = []
labels = []
for filename in os.listdir("secondary_dataset"):
    if filename.endswith(".jpg"):
        img_path = os.path.join("secondary_dataset", filename)
        im = Image.open(img_path).convert('L')
        dataset.append(np.asarray(im, dtype=np.float32))
        label = filename.replace(".jpg", "").split("-")[0]
        labels.append(label)
        print(label)
        
Image.fromarray(np.array(dataset[1], dtype=np.uint8))
print(labels)
ch2index = {}
index2ch = {}
index = 0
for label in labels:
    for ch in label:
        if ch not in ch2index.keys():
            ch2index[ch] = index
            index2ch[index] = ch
            index += 1
print("total char type:", len(ch2index))
y0 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []

for label in labels:
    y0.append(ch2index[label[0]])
    y1.append(ch2index[label[1]])
    y2.append(ch2index[label[2]])
    y3.append(ch2index[label[3]])
    y4.append(ch2index[label[4]])
    y5.append(ch2index[label[5]])

print(len(y0))
print(len(y1))
print(len(y2))
print(len(y3))
print(len(y4))
print(len(y5))

X_train = np.array(dataset[:80]).reshape(-1, 36, 100, 1)
print("XXXX")
print(len(X_train))
y_train = [
    np.array(y0[:80]),
    np.array(y1[:80]),
    np.array(y2[:80]),
    np.array(y3[:80]),
    np.array(y4[:80]),
    np.array(y5[:80]),
]

print(len(y_train[0]))

X_test = np.array(dataset[80:]).reshape(-1, 36, 100, 1)
y_test = [
    np.array(y0[80:]),
    np.array(y1[80:]),
    np.array(y2[80:]),
    np.array(y3[80:]),
    np.array(y4[80:]),
    np.array(y5[80:]),
]

print(X_test.shape)

from keras.layers import Activation, Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.models import Model
from keras.layers.merge import Concatenate


def build_model():
    input_ = Input(shape=(36, 100, 1))

    # conv layer 1
    model = BatchNormalization()(input_)
    model = Conv2D(64, (5, 5), activation ='relu', padding='same')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)

    # conv layer 2
    model = BatchNormalization()(model)
    model = Conv2D(128, (5, 5), activation ='relu', padding='valid')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)

    # conv layer 3
    model = BatchNormalization()(model)
    model = Conv2D(256, (5, 5), activation ='relu', padding='valid')(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(0.5)(model)

    # fully connected layer
    model = Flatten()(model)
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(512, activation='relu')(model)

    x0 = Dense(36, activation='softmax')(model)
    x1 = Dense(36, activation='softmax')(model)
    x2 = Dense(36, activation='softmax')(model)
    x3 = Dense(36, activation='softmax')(model)
    x4 = Dense(36, activation='softmax')(model)
    x5 = Dense(36, activation='softmax')(model)

    x = [x0, x1, x2, x3, x4, x5]

    model = Model(inputs=input_, outputs=x)
    return model

model = build_model()


print(model.summary())


model.compile(loss='sparse_categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=20)

res = model.evaluate(X_test, y_test)

print(res)

print(res[6:])

model.save("test.h5")

for x in range(81, 88):
    print(labels[x])
    # Image.fromarray(np.array(X_test[0].reshape(36, 100), dtype=np.uint8))
    res = model.predict(np.array([dataset[x]]).reshape(1, 36, 100, 1))
    print(index2ch[res[0].argmax(1)[0]], 
        index2ch[res[1].argmax(1)[0]], 
        index2ch[res[2].argmax(1)[0]], 
        index2ch[res[3].argmax(1)[0]], 
        index2ch[res[4].argmax(1)[0]],
        index2ch[res[5].argmax(1)[0]])
