import cv2
from playsound import playsound

def draw_safeline(xmin,xmax,ymin,line2_height,image_np,color):
    mid_point = (int((xmin + xmax)/2),ymin)

    if (mid_point):
        cv2.line(img=image_np,pt1=mid_point,pt2=(mid_point[0],line2_height),color=color,thickness=2,lineType=8)
        dist = float(ymin - line2_height)

    if (dist <= 0):
        im_height,im_width = image_np.shape[:2]

        # show alert on screen..
        cv2.putText(img=image_np,text='ALERT',org=(int(im_width * 0.45),30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=4,lineType=8)
        #cv2.rectangle(img=image_np,pt1=(int((im_width * 0.50)-20),15),pt2=(int((im_width * 0.50) + 60),30),color=(0,0,255),thickness=8)


        playsound('utils/alert.wav')

        return 1
    else:
        return 0


