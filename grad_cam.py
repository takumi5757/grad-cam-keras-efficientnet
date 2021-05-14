import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as preprocess_input_efficientnet,
)


def grad_cam(img, model):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model(preprocess_input_efficientnet(img[None, ...]))
        cls = np.argmax(predictions)
        loss = predictions[:, cls]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, "float32")
    gate_r = tf.cast(grads > 0, "float32")
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.zeros(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (205, 650))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    output_image = cv2.addWeighted(
        cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0
    )

    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.title("img"), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(output_image)
    plt.title("heatmap"), plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    # make a model
    # chage variables
    shape = (650, 205, 3)
    num_classes = 100

    input_tensor = Input(shape=shape)
    pretrained_model = tf.keras.applications.EfficientNetB0(
        input_tensor=input_tensor, include_top=False, weights="imagenet"
    )

    inputs = pretrained_model.input
    x = pretrained_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation="softmax")(x)

    LAYER_NAME = pretrained_model.layers[-1].name
    model = Model(
        inputs=inputs, outputs=[pretrained_model.get_layer(LAYER_NAME).output, output]
    )

    layer_names = [l.name for l in model.layers]
    # fine-tuning
    idx = layer_names.index("block7a_expand_conv")
    for layer in model.layers[:idx]:
        layer.trainable = False

    # train
    learning_rate = 0.001
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        loss_weights=[0, 1],
        metrics=["accuracy"],
    )

    img_path = None
    img = cv2.imread(img_path)
    target_size = shape[:-1][::-1]
    if img.shape[:-1] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    grad_cam(img, model)
