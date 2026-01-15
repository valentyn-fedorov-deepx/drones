import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, optimizers
from typing import Union, Callable
import keras_cv
import numpy as np

import tensorflow as tf
from data.data import PascalPartInstanceCropSemanticDataset, PascalFullDataset, PascalPartInstanceCropSemanticKerasDataset
import yaml
import cv2

class PartSegmentationModel(keras.Model):
    def __init__(self, n_classes: int, model_name: str = 'unet', backbone: str = 'resnet50_v2_imagenet',
                 encoder_weights: str = 'imagenet', encoder_freeze: bool = False,
                 optimizer: Union[dict, str] = 'Adam', loss: Union[str, Callable] = 'focal'):
        super(PartSegmentationModel, self).__init__()
        
        self.model = keras_cv.models.DeepLabV3Plus.from_preset(
            backbone,
            num_classes=9,
            input_shape=[320, 320, 3],
        )
        
        # self.model = keras_cv.models.DeepLabV3Plus(
        #     num_classes=n_classes,
        #     backbone=self.backbone,
        # )
        
        if encoder_freeze:
            for layer in self.model.backbone.layers:
                layer.trainable = False

        self.n_classes = n_classes
        self.optimizer_name = optimizer

        if isinstance(loss, str):
            if loss == 'dice':
                self.loss_fn = keras_cv.losses.DiceLoss(from_logits=True)
            elif loss == 'focal':
                self.loss_fn = keras_cv.losses.FocalLoss(from_logits=True)
            elif loss == 'cross_entropy':
                self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
            else:
                raise ValueError("Unknown loss type.")
        else:
            self.loss_fn = loss

    def call(self, inputs):
        return self.model(inputs)

    def compile_model(self):
        if isinstance(self.optimizer_name, str):
            if self.optimizer_name.lower() == 'adam':
                optimizer = optimizers.Adam(learning_rate=1e-4)
        elif isinstance(self.optimizer_name, dict):
            if self.optimizer_name['name'] == 'adam':
                optimizer = optimizers.Adam(**self.optimizer_name['params'])
        else:
            raise ValueError("Incorrect optimizer config.")

        self.compile(optimizer=optimizer, loss=self.loss_fn, metrics=['accuracy'])

    def train_step(self, batch):
        images, masks = batch
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_fn(masks, predictions)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Compute the metrics
        accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(masks, predictions))

        return {"loss": loss, "accuracy": accuracy}

    def test_step(self, batch):
        images, masks = batch
        predictions = self(images, training=False)
        loss = self.loss_fn(masks, predictions)

        accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(masks, predictions))

        return {"loss": loss, "accuracy": accuracy}

if __name__ == "__main__":
    model = PartSegmentationModel(n_classes=20, backbone='resnet50_v2_imagenet')
    print(model.summary())
    model.compile_model()

    # Example data
    X_train = np.random.rand(32, 320, 320, 3)  # Example batch of images
    y_train = np.random.randint(0, 9, (32, 320, 320, 1))  # Example batch of masks (one-hot encoded)

    #model.fit(X_train, y_train, batch_size=32, epochs=10)
    
    with open('configs/train_person_pose.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    
    train_dataset = PascalPartInstanceCropSemanticKerasDataset(**data_config["train"],
                                                            labels_to_idx=data_config["labels_to_idx"],
                                                            labels_change_class=data_config["labels_change_class"],
                                                            batch_size=16)
    
    print(len(train_dataset))
    
    input_s, mask = train_dataset[4]

    # print(item)

    print(input_s.shape, mask.shape)
    print(input_s.shape, input_s[0,:])

    mask[1,:,:,1][mask[1,:,:,1] == 1] = 255
    cv2.imwrite('mask_my.jpg', mask[1,:,:,1])

    # backbone = keras_cv.models.ResNet50V2Backbone.from_preset("resnet50_v2_imagenet")
        
    # model = keras_cv.models.DeepLabV3Plus(
    #     num_classes=9,
    #     backbone=backbone,
    # )
    
    # converter = tf.lite.TFLiteConverter.from_keras_model(model) 
    # model = converter.convert()
    
    # with open('converted_model.tflite', 'wb') as f:     
    #     f.write(model)
