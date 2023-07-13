import cv2
import time
import numpy as np
import mediapipe as mp
from gtts import gTTS
mp_facemesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
denormalize_coord=mp_drawing._normalized_to_pixel_coordinates
def dist(p1,p2):
    dist=sum([(i-j)**2 for i,j in zip(p1,p2)]) **0.5
    return dist
    
def get_mar(landmarks,chosen_mouth_idx,w,h):
    for face_id,face_landmarks in enumerate(landmarks):
        coords_points_mouth=[]
        for i in chosen_mouth_idx:
            lm=landmarks[i]
            coord_point=denormalize_coord(lm.x,lm.y,w,h)
            coords_points_mouth.append(coord_point)

    p62_p68=dist(coords_points_mouth[6],coords_points_mouth[7])
    p63_p67=dist(coords_points_mouth[0],coords_points_mouth[1])
    p64_p66=dist(coords_points_mouth[4],coords_points_mouth[5])
    mouth_h=dist(coords_points_mouth[2],coords_points_mouth[3])
    mouth_v=(p62_p68+p63_p67+p64_p66)/3
    mar=mouth_v/mouth_h
    return mar,coords_points_mouth  
    
def get_ear(landmarks,chosen_le_idx,chosen_re_idx,w,h):
    coords_points_left=[]
    coords_points_right=[]
    for i in chosen_le_idx:
        lm=landmarks[i]
        coord_left=denormalize_coord(lm.x,lm.y,w,h)
        coords_points_left.append(coord_left)
    for i in chosen_re_idx:
        lm=landmarks[i]
        coord_right=denormalize_coord(lm.x,lm.y,w,h)
        coords_points_right.append(coord_right)
    p2_p6_l=dist(coords_points_left[1],coords_points_left[5])
    p3_p5_l=dist(coords_points_left[2],coords_points_left[4])
    p1_p4_l=dist(coords_points_left[0],coords_points_left[3])
    p2_p6_r=dist(coords_points_right[1],coords_points_right[5])
    p3_p5_r=dist(coords_points_right[2],coords_points_right[4])
    p1_p4_r=dist(coords_points_right[0],coords_points_right[3])
    left_ear=(p2_p6_l+p3_p5_l)/(2.0 * p1_p4_l)
    right_ear=(p2_p6_r+p3_p5_r)/(2.0 * p1_p4_r)
    ear=(left_ear+right_ear)/2.0
    return ear,(coords_points_left,coords_points_right)

def plot_eye_mouth_landmarks(frame,left_lm_coord,right_lm_coord,mouth_coord,color):
    for lm_coord in [left_lm_coord,right_lm_coord,mouth_coord]:
        if lm_coord:
            for coord in lm_coord:
                cv2.circle(frame,coord,2,color,-1)
    frame=cv2.flip(frame,1)
    return frame
    
def plot_text(image,text,origin,color,font=cv2.FONT_HERSHEY_SIMPLEX,fntScale=0.8,thickness=2):
    image=cv2.putText(image,text,origin,font,fntScale,color,thickness)
    return image 

class ObjectVideoFrameHandler:
    def __init__(self):
        self.eye_idxs = {
        "left" : [362,385,387,263,374,380],
        "right" : [33,160,158,133,153,144] }
        self.mouth_idxs=[13,14,308,78,312,317,82,87]
        self.red=(0,0,255)
        self.f=open("coco.names","r")
        self.names=self.f.read().strip().split("\n")
        self.colors=np.random.randint(0,255,size=(len(self.names),3),dtype="uint8")
        self.green=(0,255,0)
        self.state_tracker={
         "start_time":time.perf_counter(),
         "drowsy_time":0.0,
         "color":self.green,
         "play_alarm":False }
        self.ear_txt_pos=(10,30)
        self.mar_txt_pos=(10,60)
        self.create_audio=True
        self.net=cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
    def process(self,frame: np.array,thresholds:dict,known_distance,known_width,proxy):
        frame.flags.writeable=False
        frame=np.ascontiguousarray(frame)
        h,w,_=frame.shape
        drowsy_time_txt_pos=(10,int(h // 2 *1.7))
        alm_txt_pos=(10,int(h // 2*1.85))       
        with mp_facemesh.FaceMesh(static_image_mode=True) as face_mesh:
            results=face_mesh.process(frame)
        if results.multi_face_landmarks:
            landmarks=results.multi_face_landmarks[0].landmark
            EAR,coordinates = get_ear(landmarks,self.eye_idxs["left"],self.eye_idxs["right"],w,h)
            MAR,mouth_coordinates=get_mar(landmarks,self.mouth_idxs,w,h)
            frame=plot_eye_mouth_landmarks(frame,coordinates[0],coordinates[1],mouth_coordinates,self.state_tracker["color"])
            if EAR<thresholds["EAR_THRESH"]:
                end_time=time.perf_counter()
                self.state_tracker["drowsy_time"]+=end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"]=end_time
                self.state_tracker["color"]=self.red
                if self.state_tracker["drowsy_time"] >= thresholds["wait_time"]:
                    self.state_tracker["play_alarm"]=True
                    plot_text(frame,"WAKE UP! WAKE UP!",alm_txt_pos,self.state_tracker["color"])
                    
            else:
                self.state_tracker["start_time"]=time.perf_counter()
                self.state_tracker["drowsy_time"]=0.0
                self.state_tracker["color"]=self.green
                self.state_tracker["play_alarm"]=False
            ear_txt= f"EAR : {round(EAR,2)}"
            mar_txt= f"MAR : {round(MAR,2)}"
            drowsy_time_txt= f"DROWSY: {round(self.state_tracker['drowsy_time'],3)} Secs"
            plot_text(frame,ear_txt,self.ear_txt_pos,self.state_tracker["color"])
            if MAR>thresholds["MAR_THRESH"]:
                mar_txt= f"MAR : {round(MAR,2)}: Yawning"
                plot_text(frame,mar_txt,self.mar_txt_pos,(0,0,255))
            else:
                plot_text(frame,mar_txt,self.mar_txt_pos,self.state_tracker["color"])
            plot_text(frame,drowsy_time_txt,drowsy_time_txt_pos,self.state_tracker["color"])
        else:
            self.state_tracker["start_time"]=time.perf_counter()
            self.state_tracker["drowsy_time"]=0.0
            self.state_tracker["color"]=self.green
            self.state_tracker["play_alarm"]=False
        play_alarm_object=False
        ln=self.net.getLayerNames()
        new_ln=list()
        for i in self.net.getUnconnectedOutLayers():
            new_ln.append(ln[i-1])
        blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
        self.net.setInput(blob)
        layersOutput=self.net.forward(new_ln)
        boxes=[]
        confidences=[]
        classIds=[]
        for output in layersOutput:
            for detection in output:
                scores=detection[5:]
                classId=np.argmax(scores)
                confidence=scores[classId]
                if confidence>0.5:
                    box=detection[0:4]*np.array([w,h,w,h])
                    (centerX,centerY,width,height)=box.astype("int")
                    x=int(centerX-(width/2))
                    y=int(centerY-(height/2))
                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIds.append(classId)
        detect=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)
        if len(detect)>0:
            for i in detect.flatten():
                (x,y)=(boxes[i][0],boxes[i][1])
                (w,h)=(boxes[i][2],boxes[i][3])
                color=[int(c) for c in self.colors[classIds[i]]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                text="{}:{:.4f}".format(self.names[classIds[i]],confidences[i])
                cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                if proxy:
                    focal_length=(w*known_distance)/known_width
                    inches=(known_width*focal_length)/w
                    cv2.putText(frame,"Dist %.2fft" % (inches/12),(w-50,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            if self.create_audio and self.state_tracker["play_alarm"]:
                object_list=list()
                for i in detect:
                    object_list.append(self.names[classIds[i]])
                val=max(object_list,key=object_list.count)
                text="Wake up Wake up " + " " + val + " is near"
                obj=gTTS(text=text,lang="en")
                obj.save("speech-1.mp3")
            
                
        return frame,self.state_tracker["play_alarm"]
            