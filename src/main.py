import cv2
import os
import logging
import time
import numpy as np
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
    
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--landmark", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                             "(0.6 by default)")
    parser.add_argument("-pof", "--print_output_frame", required=False, nargs='+',
                        default=[],
                        help="Example: --pof fd fl hp ge (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fl for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    
    
    return parser


def main():

    try:

        args = build_argparser().parse_args()

        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler("gaze-app.log"),
                    logging.StreamHandler()
                ])
        
        print_output_frame = args.print_output_frame
        
        logger = logging.getLogger()

        input_file_path = args.input
        feeder = None

        if input_file_path.lower()=="CAM":
                feeder = InputFeeder("cam")
        else:
            if not os.path.isfile(input_file_path):
                logger.error("Unable to find specified video file")
                exit(1)
            feeder = InputFeeder("video",input_file_path)
        

        mc = MouseController('low','fast')
        feeder.load_data()
        

        modelPathDict = {'FaceDetectionModel':args.face, 'FacialLandmarksDetectionModel':args.landmark, 
        'GazeEstimationModel':args.gazeestimation, 'HeadPoseEstimationModel':args.headpose}
        
        for fileNameKey in modelPathDict.keys():
            
            if not os.path.isfile(modelPathDict[fileNameKey]+'.xml'):
                logger.error("Unable to find specified "+fileNameKey+" xml file")
                exit(1)


        logging.info("============== Models Load time ===============") 
        face_detection = FaceDetection(args.face, args.device, args.prob_threshold, args.cpu_extension)
        start_time = time.time()
        face_detection.load_model()
        logging.info("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        landmarks_detection = FacialLandmarksDetection(args.landmark, args.device, args.cpu_extension)
        start_time = time.time()
        landmarks_detection.load_model()
        logging.info("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        gaze_estimation = GazeEstimation(args.gazeestimation, args.device, args.cpu_extension)
        start_time = time.time()
        gaze_estimation.load_model()
        logging.info("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

        headpose_estimation = HeadPoseEstimation(args.headpose, args.device, args.cpu_extension)
        start_time = time.time()
        headpose_estimation.load_model()
        logging.info("Headpose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )
        logging.info("==============  End =====================") 
        
        frame_count = 0
        fd_infertime = 0
        lm_infertime = 0
        hp_infertime = 0
        ge_infertime = 0

        for ret, frame in feeder.next_batch():
            if not ret:
                break
            frame_count += 1
            
        
            key = cv2.waitKey(60)
            start_time = time.time()
            cropped_face, face_coords = face_detection.predict(frame.copy())
            fd_infertime += time.time() - start_time

            if type(cropped_face) == int:
                continue
            
            start_time = time.time()
            headpose_out = headpose_estimation.predict(cropped_face.copy())
            hp_infertime += time.time() - start_time

            start_time = time.time()
            left_eye, right_eye, eye_coords = landmarks_detection.predict(cropped_face.copy())
            lm_infertime += time.time() - start_time
            
            start_time = time.time()
            new_mouse_coord, gaze_vector = gaze_estimation.predict(left_eye, right_eye, headpose_out)
            ge_infertime += time.time() - start_time
            
            if (not len(print_output_frame) == 0):
                preview_frame = frame.copy()
                if 'fd' in print_output_frame:
                    preview_frame = cropped_face
                    cv2.rectangle(frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3)
                    
                if 'fl' in print_output_frame:
                    cv2.rectangle(cropped_face, (eye_coords[0][0], eye_coords[0][1]), (eye_coords[0][2], eye_coords[0][3]),(0, 255, 0),2)
                    cv2.rectangle(cropped_face, (eye_coords[1][0], eye_coords[1][1]), (eye_coords[1][2], eye_coords[1][3]),(0, 255, 0),2)
                    
                if 'hp' in print_output_frame:
                    cv2.putText(cropped_face, 
                                "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(headpose_out[0],headpose_out[1],headpose_out[2]), 
                                                                                            (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
                    
                    
                    face = frame[face_coords[1]:face_coords[3],face_coords[0]:face_coords[2]]
                    xmin, ymin,_ , _ = face_coords
                    face_center = (xmin + face.shape[1] / 2, ymin + face.shape[0] / 2, 0)
                    headpose_estimation.draw_axes(frame, face_center, headpose_out[0], headpose_out[1], headpose_out[2])
                
                if 'ge' in print_output_frame:
                    
                    cropped_h, cropped_w = cropped_face.shape[:2]
                    arrow_length = 0.3 * cropped_h

                    gaze_arrow_x = gaze_vector[0] * arrow_length
                    gaze_arrow_y = -gaze_vector[1] * arrow_length

                    cv2.arrowedLine(cropped_face, (eye_coords[0][0], eye_coords[0][1]),
                                    (int(eye_coords[0][2] + gaze_arrow_x), int(eye_coords[0][3] + gaze_arrow_y)), (0, 255, 0), 2)
                    cv2.arrowedLine(cropped_face, (eye_coords[1][0], eye_coords[1][1]),
                                    (int(eye_coords[1][2] + gaze_arrow_x), int(eye_coords[1][3] + gaze_arrow_y)), (0, 255, 0), 2)

                    
                    
                    #frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = cropped_face
                                
                
                if len(preview_frame) != 0:
                    img_hor = np.hstack((cv2.resize(preview_frame, (800, 800)),cv2.resize(frame, (800, 800))))
                else:
                    img_hor = cv2.resize(frame, (800, 800))
                    
                cv2.imshow("Monitor",img_hor)
            
            if frame_count % 5 == 0:
                mc.move(new_mouse_coord[0],new_mouse_coord[1])

            

            if key==27:
                    break
            
        #logging inference times
        if(frame_count > 0):
            logging.info("============== Models Inference time ===============") 
            logging.info("Face Detection:{:.1f}ms".format(1000* fd_infertime/frame_count))
            logging.info("Facial Landmarks Detection:{:.1f}ms".format(1000* lm_infertime/frame_count))
            logging.info("Headpose Estimation:{:.1f}ms".format(1000* hp_infertime/frame_count))
            logging.info("Gaze Estimation:{:.1f}ms".format(1000* ge_infertime/frame_count))
            logging.info("============== End ===============================")
        
        logger.info("Video stream ended...")
        cv2.destroyAllWindows()
        feeder.close()

    except Exception as ex:
        logging.exception("Error in inference:" + str(ex))
     
    

if __name__ == '__main__':

    #arg = '-fd ../models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001 -fl ../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i ../media/demo.mp4 -d CPU -pof fd fl hp ge'.split(' ')
    #args = build_argparser().parse_args(arg)

    main() 
 