import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, optimizers, callbacks
from typing import Union, Callable
import keras_cv
import numpy as np

import tensorflow as tf
from data.data import PascalPartInstanceCropSemanticKerasDataset, CocoPartInstanceCropSemanticKerasDataset
import yaml

import wandb
from wandb.integration.keras import WandbCallback

# wandb.init(
#     project="keras_body_parts_workable",  
#     name="pascal_run_fixed",          
# )

my_callbacks = [
    callbacks.ModelCheckpoint(
    'best_model_pascal.keras',               
    monitor='val_loss',           
    save_best_only=True,           
    mode='min',                   
    verbose=1),

    tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  
    patience=2,  
    verbose=1),
    
    callbacks.EarlyStopping(
    monitor='val_loss',            
    patience=5,                    
    mode='min',                   
    verbose=1),
        
    # WandbCallback(
    # save_model=(False),
    # save_graph=(False))
    
]

#model = keras.models.load_model('/nvme0n1-disk/stepan.severylov/people-track/aux_codes/part_seg/best_model_pascal.keras')

model = keras_cv.models.SegFormer.from_preset(
    # "deeplab_v3_plus_resnet50_pascalvoc",
    "segformer_b2",
    num_classes=9,
    input_shape=[224, 224, 3],
)

# model.layers.append(layers.Conv2D(1, (1, 1)))

# import ipdb; ipdb.set_trace()

def preprocess_tfds_inputs(inputs):
    def unpackage_tfds_inputs(tfds_inputs):
        return {
            "images": tfds_inputs["image"] / 255,
            "segmentation_masks": tfds_inputs["class_segmentation"],
        }

    outputs = inputs.map(unpackage_tfds_inputs)
    outputs = outputs.map(keras_cv.layers.Resizing(height=224, width=224))
    outputs = outputs.batch(4, drop_remainder=True)
    return outputs

# train_ds = keras_cv.datasets.pascal_voc.segmentation.load(split="sbd_train")
# eval_ds = keras_cv.datasets.pascal_voc.segmentation.load(split="sbd_eval")

# train_ds = preprocess_tfds_inputs(train_ds)
# eval_ds = preprocess_tfds_inputs(eval_ds)

# data = next(train_ds.as_numpy_iterator())
# import ipdb; ipdb.set_trace()


lr = 0.07 * 8 / 16
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=lr, weight_decay=0.0001,
        momentum=0.9, clipnorm=10.0
    ),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[
        keras.metrics.CategoricalAccuracy(),
        keras.metrics.MeanIoU(num_classes=9, sparse_y_true=False, sparse_y_pred=False),
    ],
)

with open("configs/train_person_pose.yaml", 'r') as f:
        config = yaml.safe_load(f)

data_config = config["data"]

train_dataset = PascalPartInstanceCropSemanticKerasDataset(**data_config["train"],
                                                           labels_to_idx=data_config["labels_to_idx"],
                                                           labels_change_class=data_config["labels_change_class"],
                                                           batch_size=8, image_size=224)

val_dataset = PascalPartInstanceCropSemanticKerasDataset(**data_config["val"],
                                                         labels_to_idx=data_config["labels_to_idx"],
                                                         labels_change_class=data_config["labels_change_class"],
                                                         batch_size=8, image_size=224)

t_images_path = "/nvme0n1-disk/stepan.severylov/datasets/coco-seg/images/train2017"
t_labels_path = "/nvme0n1-disk/stepan.severylov/datasets/coco/labels/train2017"

v_images_path = "/nvme0n1-disk/stepan.severylov/datasets/coco-seg/images/val2017"
v_labels_path = "/nvme0n1-disk/stepan.severylov/datasets/coco/labels/val2017"

# coco_train_dataset = CocoPartInstanceCropSemanticKerasDataset(images_path=t_images_path,
#                                                               labels_path=t_labels_path,
#                                                               image_size=224)

# coco_val_dataset = CocoPartInstanceCropSemanticKerasDataset(images_path=v_images_path,
#                                                             labels_path=v_labels_path,
#                                                             data_limit=1000,
#                                                             image_size=224)


# val_dataset = PascalPartInstanceCropSemanticDataset(**data_config["val"],
#                                                     labels_to_idx=data_config["labels_to_idx"],
#                                                     labels_change_class=data_config["labels_change_class"])


# X = np.ones(shape=(32, 96, 96, 3))
# y = np.zeros(shape=(32, 96, 96, 9))

# print("X shape:", X.shape)  # Expected: (32, 320, 320, 3)
# print("y shape:", y.shape)

# model.fit(coco_train_dataset, validation_data=coco_val_dataset, epochs=10, callbacks=my_callbacks)

model.fit(train_dataset, validation_data=val_dataset, epochs=500, callbacks=my_callbacks)
model.save("final_10epo.keras")

def dict_to_tuple(x):
    import tensorflow as tf

    return x["images"], tf.one_hot(
        tf.cast(tf.squeeze(x["segmentation_masks"], axis=-1), "int32"), 21
    )

# train_ds = train_ds.map(dict_to_tuple)
# eval_ds = eval_ds.map(dict_to_tuple)

# model.fit(train_ds, validation_data=eval_ds, epochs=10)