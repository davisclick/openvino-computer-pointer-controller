import os
import cv2
import numpy as np
import logging
from openvino.inference_engine import IENetwork, IECore

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.60,  extensions=None):

        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.threshold = threshold
        self.extension = extensions
        self.plugin = None
        self.exec_network = None
        self.network = None

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape


    def load_model(self):

        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        
        if not self.check_model():
            exit(1)
        
        self.exec_network = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):

        input_img = self.preprocess_input(image)
        input_dict={self.input_name: input_img}  
        
        h, w = image.shape[:2]
        
        infer_request_handle = self.exec_network.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]
            coords = self.preprocess_output(outputs)
            
            if ( len(coords) == 0 ):
                return 0, 0
            
            first_coords = coords[0]
            first_coords = first_coords* np.array([w, h, w, h])
            first_coords = first_coords.astype(np.int32)
        
            cropped_face = image[first_coords[1]:first_coords[3], first_coords[0]:first_coords[2]]
            return cropped_face, first_coords

    def check_model(self):

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            logging.info("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions:
                logging.info("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    logging.info("After adding the extension still unsupported layers found")
                    return False
                logging.info("After adding the extension the issue is resolved")
            else:
                logging.info("Give the path of cpu extension")
                return False
        logging.info("All layers are supported for " + self.model_structure)

        return True 

    def preprocess_input(self, image):
        
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, self.input_shape[2], self.input_shape[3])
        return image

    def preprocess_output(self, outputs):

        coords = []
        for i in np.arange(0, outputs.shape[2]):
            confidence = outputs[0,0,i,2]
            if confidence > self.threshold:
                box = outputs[0, 0, i, 3:7]
                coords.append(box)
        return coords
