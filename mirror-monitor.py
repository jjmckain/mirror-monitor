# Mirror Monitor v1
# Use picamera to detect faces, see if we recognize them, and upload a still to the cloud.  
# When we recognize the user inform MagicMirror^2, so it can customize content.
# @since 2021-03-14
# @author James McKain <mck222@gmail.com>
# @license MIT

import time
import numpy as np
import picamera
from PIL import Image
from tflite_runtime.interpreter import Intepreter

# constants - tune as needed
loop_delay_secs = 5
confidence_threshold = 0.4
path_to_labels = "./faces-labels.txt"
path_to_model = "./faces-model.tflite"
path_to_image = "./stillframe.jpg"

# init Raspberry Pi Camera
camera = picamera.PiCamera()
camera.resolution = (224, 224)  # ML model expects 224x224 image

def load_labels():
    '''Load the ML labels file'''
    with open(path_to_labels, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def mk_intepreter():
    '''Initialize & return the Intepreter'''
    interpreter = Interpreter(path_to_model)
    interpreter.allocate_tensors()
    return interpreter

def take_picture(camera):
    '''Take a picture after a quick pause for lighting to adjust, then write to disk'''
    camera.start_preview()    
    time.sleep(1)
    camera.capture(path_to_image)
    camera.stop_preview()

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """ return a sorted array of classification results """
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # if model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def tell_mirror(person_nm):
    '''Send a message over to the Node app running the mirror and let it know who's here'''
    # TODO - need to consult mirror API as to how

def main():
    '''Init the program and run a timed/infinite loop, taking & processing a picture on each iteration'''
    labels = load_labels()
    interpreter = mk_interpreter()
    #_, height, width, _ = interpreter.get_input_details()[0]['shape']

    # TODO Change this from inf loop to motion-triggered
    while (True):
        take_picture()
        image = Image.open(path_to_image)
        results = classify_image(interpreter, image)
        label_id, prob = results[0]
        person_nm = "friend"
        if prob >= confidence_threshold:
            person_nm = labels[label_id]
            tell_mirror(person_nm)

        upload_picture(person_nm)
        time.sleep(loop_delay_secs)



if __name__ == '__main__':
    main()
