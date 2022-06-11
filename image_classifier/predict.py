import argparse, os, warnings, json, logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore')

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255

    return image.numpy()

def predict(img, model, k=1):
    image = np.asarray(img)
    processed = process_image(image)
    expanded = np.expand_dims(processed, axis=0)
    pred = model.predict(expanded)
    top5_probs, top5_labels = tf.nn.top_k(pred, k=k)
    return top5_probs.numpy()[0].tolist(), top5_labels.numpy()[0].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image path")
    parser.add_argument("model", help="Classifier model")
    parser.add_argument('-t', "--top_k", help="Return the top K most likely classes", type=int, choices=range(1, 6))
    parser.add_argument('-c', "--category_name", help="Path to a JSON file mapping labels to flower names")

    args_dict = vars(parser.parse_args())
    top_k = args_dict.get("top_k")
    names_json = args_dict.get("category_name")
    image_path = args_dict.get("image")
    model = args_dict.get("model")
    
    reloaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    
    im = Image.open(image_path)
    
    if top_k:
        probs, labels = predict(im, reloaded_model, top_k)
    else:
        probs, labels = predict(im, reloaded_model)    
    
    if names_json:
        with open('./assets/flowers/flowers_classes.json') as f:
            class_names = json.load(f)
            
        for p,l in zip(probs, labels):
            print("There's {pct}% chance it's a: {label}".format(pct=round(p*100, 2), label=class_names.get('%s' % int(l+1)).upper()))
    else:
        for p,l in zip(probs, labels):
            print("There's {pct}% chance it's class: {label}".format(pct=round(p*100, 2), label=l))