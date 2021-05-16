import cv2
from utils import detector_util
from utils import orient_lines
from utils import alert_check
import argparse
from imutils.video import VideoStream
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d','--display',dest='display',default=1,type=int,help='DISPLAY THE IMAGE ON SCREEN WITH THE HELP OF OPEN CV')
args = vars(parser.parse_args())



detection_graph,sess = detector_util.load_inference_graph()


if __name__ == '__main__':

    vs = VideoStream(0).start()


    frame_counts = 0
    start_time = datetime.now()
    line1_pct = float(0.15)
    line2_pct = float(0.25)

    im_height,im_width = (None,None)

    num_of_hands_detect = 4
    score_thresh = 0.80

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    fps = 0



    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)


            # convert it into rgb.
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if im_height == None:
                im_height,im_width = frame.shape[:2]


            # run the image to tensorflow graph..
            boxes,scores,classes = detector_util.detect_objects(frame,sess,detection_graph)

            # draw safe_lines on image..
            line2_height = orient_lines.draw_safe_lines(line1_pct,line2_pct,frame)

            # draw bbox & some texts
            b = detector_util.draw_boxes_on_image(frame,im_height,im_width,boxes,scores,classes,num_of_hands_detect,score_thresh,line2_height)


            # calculate FPS
            total_time = (datetime.now() - start_time).total_seconds()
            frame_count = frame_count + 1

            fps = frame_count / total_time




            if args['display']:
                # display fps on the frame..
                detector_util.draw_fps_on_image(fps,frame)

                cv2.imshow('Detection',frame)
                cv2.release()



                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow()
                    vs.stop()
                    break
            print("Average FPS: ", str("{0:.2f}".format(fps)))

    except Exception as e:
        print("Average FPS: ", str("{0:.2f}".format(fps)))










