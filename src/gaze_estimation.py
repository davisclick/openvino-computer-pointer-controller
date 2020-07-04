import os
from openvino.inference_engine import IENetwork, IECore
import cv2
import math

class GazeEstimation:
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

        print('Loading Network...')
        
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        
        if( self.check_model() == 0 ):
            exit(1)
        
        self.exec_network = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)
        
        print('Network loaded.')

    def predict(self, left_eye, right_eye, head_position):

        processed_left_eye = self.preprocess_input(left_eye)
        processed_right_eye = self.preprocess_input(right_eye)
        
        input_dict= {'left_eye_image': processed_left_eye,
                                              'right_eye_image': processed_right_eye,
                                              'head_pose_angles': head_position}

        
        infer_request_handle = self.exec_network.start_async(request_id=0,inputs = input_dict)
        
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name].tolist()[0]
            
            return self.preprocess_output(outputs, head_position)

    def check_model(self):

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions == None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    return 0
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                return 0
        print("All layers are supported !!")

        return 1 

    def preprocess_input(self, image):
        
        net_input_shape = self.network.inputs['right_eye_image'].shape
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs, head_position):

        roll = head_position[2]
        gaze_vector = outputs

        cos_theta = math.cos(roll * math.pi / 180)
        sin_theta = math.sin(roll * math.pi / 180)

        x = outputs[0] * cos_theta + outputs[1] * sin_theta
        y = outputs[1] * cos_theta - outputs[0] * sin_theta

        return (x, y), gaze_vector
