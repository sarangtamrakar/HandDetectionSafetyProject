import tensorflow as tf
import imutils
import numpy as np
import sys
from utils import label_map_util
from utils import alert_check
import cv2

path_to_graph = 'frozen_graphs'+'/ssd5_optimized_inference_graph.pb'
path_to_labels = 'frozen_graphs/Glove_label_map.pbtxt'



def load_inference_graph():
    print('>>>>>>>>>>>loading graph..')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
        print('>>>>>>>>> graph is loaded')
    return detection_graph,sess


def detect_objects(image_np,sess,detection_graph):
    # define input & output tensors...
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_expended = np.expand_dims(image_np,axis=0)

    (boxes,scores,classes,num)    = sess.run([detection_boxes,detection_scores,detection_classes,num_detections],feed_dict={image_tensor:image_expended})

    return np.squeeze(boxes),np.squeeze(scores),np.squeeze(classes)



def distance_to_camera(avg_width_of_object,focal_length_of_cam,pixel_length_of_box):
    distance = (avg_width_of_object * focal_length_of_cam)/pixel_length_of_box
    return int(distance)

def draw_boxes_on_image(image_np,im_height,im_width,boxes,scores,classes,num_of_hand_to_detects,score_thresh,line2_height):


    color = 0
    color1 = (255,0,0)
    color2 = (0,0,255)
    focal_length = 875
    avg_width_of_hand = 4.0
    hand_count = 0





    for i in range(num_of_hand_to_detects):

        if (scores[i] > score_thresh):



            # define classe manually without .pbtxt
            if classes[i] == 1:
                id = 'hand'
            else:
                id = 'gloved_hand'
                avg_width_of_hand = 3.0


            # define color:
            if i == 0:
                color = color1
            else:
                color = color2



            # getting cordintes of boxes , & graph by default gives normalized value that why we are multiple with actual length..
            ymin,xmin,ymax,xmax = (boxes[i][0] * im_height,boxes[i][1] * im_width,boxes[i][2] * im_height, boxes[i][3] * im_width)

            ymin, xmin, ymax, xmax = int(ymin),int(xmin),int(ymax),int(xmax)

            p1 = (xmin,ymin)
            p2 = (xmax,ymax)

            pixel_length_of_boxes = (xmax-xmin)


            distance = distance_to_camera(avg_width_of_hand,focal_length,pixel_length_of_boxes)

            if (distance):
                hand_count = hand_count + 1
            
            # draw boxes..
            cv2.rectangle(img=image_np,pt1=p1,pt2=p2,color=color,thickness=2,lineType=8,shift=0)

            # draw classs..
            cv2.putText(img=image_np,text='Hand : '+str(i) +' '+ str(id),org=(xmin,int(ymin - 5)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50,color=color,thickness=2)
            
            # draw confidence score
            cv2.putText(img=image_np,text='confidence score : {0:0.2f}:'.format(scores[i]),org=(xmin,int(ymin-30)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50,color=color,thickness=2)

            # distance to camera..
            cv2.putText(img=image_np,text='distance to camera '+str(distance)+'inches',org=(int(im_width * 0.60),int(im_height * 0.90) + int(20*i)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.40,color=color,thickness=1,lineType=8)

            # draw safe line
            a = alert_check.draw_safeline(xmin,xmax,ymin,line2_height,image_np,color)

        if hand_count == 0:
            b = 0
        else:
            b = 1

    return b


def draw_fps_on_image(fps,image_np):
    cv2.putText(img=image_np,text='FPS : {}'.format(fps),org=(10,15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50,color=(255,0,0),thickness=2,lineType=8)


















































