import cv2


def draw_safe_lines(line1_pct,line2_pct,image_np):

    im_height,im_width = image_np.shape[:2]

    position = int(im_width * 0.65)


    # first we will put text to about the line..
    cv2.putText(img=image_np,text='RED LINE : MACHINE BORDER LINE',org=(position,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.40,color=(0,0,255),thickness=1,lineType=8)
    cv2.putText(img=image_np, text='BLUE LINE : SAFETY BORDER LINE', org=(position, 50),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.40, color=(255, 0, 0), thickness=1, lineType=8)

    # draw lines...
    cv2.line(img=image_np,pt1=(0,int(im_height * line1_pct)),pt2=(int(im_width),int(im_height * line1_pct)),color=(0,0,255),thickness=2,lineType=8,shift=0)
    cv2.line(img=image_np,pt1=(0,int(im_height * line2_pct)),pt2=(int(im_width),int(im_height * line2_pct)),color=(255,0,0),thickness=2,lineType=8,shift=0)

    line2_height = int(im_height * line2_pct)

    return line2_height






