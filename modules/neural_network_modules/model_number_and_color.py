"""
Models to detect number and color.
"""
import numpy as np
import tensorflow as tf

class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, top_brightness, **kwargs):
        super().__init__(**kwargs)
        self.top_brightness = top_brightness
    def call(self, x):
        br = tf.random.uniform([1], minval=0, maxval=self.top_brightness, dtype=tf.dtypes.float32)[0]
        return tf.image.adjust_brightness(x, br)

class DataAug(tf.keras.Model):
    def __init__(self, brightness=.28, **kwargs):
        super().__init__()
        self.transformations = [tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'),
                                tf.keras.layers.RandomZoom((-0.2, 0.2), fill_mode='nearest'), RandomBrightness(brightness),
                                tf.keras.layers.RandomRotation(0.5, fill_mode='nearest')]  
    def call(self, x):#ignora my_training
        for t in self.transformations:
            x = t(x)
        return x

data_aug = DataAug()

class ModelPrediction(tf.keras.Model):
    def __init__(self, base_model):
        super(ModelPrediction, self).__init__()
        self.base_model = base_model 
        self.bright = tf.image.adjust_brightness
        self.rotate = tf.image.rot90
    
    def call(self, photo):
        angles_r = [0, 1, 2, 3]
        br = [.1,.2]
        batch = tf.stack([self.bright(self.rotate(photo[0], angle), b) for angle in angles_r for b in br])
        all_results = self.base_model(batch)
        return tf.expand_dims(tf.math.reduce_mean(all_results, axis=0), axis=0)

class MyModel_color(tf.keras.Model):
    def __init__(self):
        super(MyModel_color, self).__init__()
        self.data_aug = data_aug
        self.rescaling = tf.keras.layers.Rescaling(1./127.5, offset=-1)
        self.dense_layers = [tf.keras.layers.Flatten(),
                               tf.keras.layers.Dense(16, activation='tanh'),
                              tf.keras.layers.Dense(4, activation='softmax')]

    def call(self, x, data_aug=False):
        if data_aug:
            x = self.data_aug(x) 
        x = self.rescaling(x)
        for layer in self.dense_layers:
            x = layer(x)
        return x

class MyModel_number(tf.keras.Model):
    def __init__(self, base_model):
        super(MyModel_number, self).__init__()
        self.data_aug = data_aug
        self.rescaling = tf.keras.layers.Rescaling(1./127.5, offset=-1)
        self.base_model = base_model
        self.predition_head = [tf.keras.layers.GlobalAveragePooling2D(),
                               tf.keras.layers.Dense(32, activation='relu'),
                              tf.keras.layers.Dense(14, activation='softmax')]

    def call(self, x, data_aug=False):
        if data_aug:
            x = self.data_aug(x) 
        x = self.rescaling(x)
        x = self.base_model(x, training=False)
        for layer in self.predition_head:
            x = layer(x)
        return x

def load_model_number(loc, use_augmented_input=True):
    _ = tf.keras.models.load_model(loc)
    pred_model_number = _.layers[0]
    pred_model_number.trainable = False

    model_number = MyModel_number(pred_model_number)
    model_number(np.zeros((3,96, 96, 3)), data_aug=False)

    _ = tf.keras.models.load_model(loc)
    model_number.set_weights(_.get_weights())
    if use_augmented_input:
        model_number_aug = ModelPrediction(model_number)
        model_number_aug(np.zeros((3,96, 96, 3)))
        return model_number_aug
    return model_number

def load_model_color(loc, use_augmented_input=True):
    model_color = MyModel_color()
    model_color(np.zeros((3, 20, 20, 3)), data_aug=False)

    _ = tf.keras.models.load_model(loc)
    model_color.set_weights(_.get_weights())
    if use_augmented_input:
        model_color_aug = ModelPrediction(model_color)
        model_color_aug(np.zeros((3, 20, 20, 3)))
        return model_color_aug
    return model_color
