import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class EmbryoClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ['2-cell', '4-cell', '8-cell', 'Blastocyst', 'Morula'] 
        self.img_size = (160, 160) # Turbo resolution

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img).astype('float32')
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def predict(self, image_path):
        processed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_img)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        return self.classes[class_idx], confidence, predictions[0]

    def get_gradcam(self, image_path, layer_name='Conv_1'):
        """
        Generates Grad-CAM heatmap for MobileNetV2 with Keras 3 compatibility.
        """
        img_array = self.preprocess_image(image_path)
        
        # 1. Find the base model (MobileNetV2) inside the Sequential model
        # Our Sequential structure: [base_model, GlobalAveragePooling2D, Dropout, Dense]
        base_model = self.model.layers[0]
        
        # 2. Get the specific conv layer inside the base model
        try:
            target_layer = base_model.get_layer(layer_name)
        except:
            # Fallback: try finding the last conv layer if names differ
            conv_layers = [l for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D) or 'conv' in l.name.lower()]
            target_layer = conv_layers[-1] if conv_layers else base_model.layers[-1]

        # 3. Create a functional model that maps inputs to [conv_output, model_output]
        # We use the base_model's input and specific layer output to avoid Sequential nesting issues
        grad_model = tf.keras.models.Model(
            [base_model.input], [target_layer.output, base_model.output]
        )

        # 4. We need to handle the "head" (the pooling and dense layers) separately
        # since grad_model only goes up to base_model.output
        head_layers = self.model.layers[1:]

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_model(img_array)
            # Pass base_outputs through the head layers
            x = base_outputs
            for layer in head_layers:
                x = layer(x)
            predictions = x
            
            top_pred_index = tf.argmax(predictions[0])
            loss = predictions[:, top_pred_index]

        # 5. Calculate gradients and generate heatmap
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        # Cast to float32 to avoid "half tensor" (float16) conflicts on CPU
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        grads = tf.cast(grads, 'float32')
        output = tf.cast(output, 'float32')

        guided_grads = gate_f * gate_r * grads
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.dot(output, weights)
        cam = cv2.resize(cam, (self.img_size[0], self.img_size[1]))
        cam = np.maximum(cam, 0)
        heatmap = cam / (np.max(cam) + 1e-10)

        # Overlay
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        return superimposed_img
