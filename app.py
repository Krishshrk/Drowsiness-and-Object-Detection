import streamlit as st
from gtts import gTTS
from pathlib import Path
from playsound import playsound   
from Object_detection import ObjectVideoFrameHandler
from Drowsiness_detection import VideoFrameHandler
from Audio_frame import AudioFrameHandler
import threading
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
import av
import os

drowsiness_sound="Wake up Wake up"
object_sound="Wake up Wake up vehicle is near"

st.set_page_config(
page_title="Object and drowsiness detection",
page_icon="https://thumbs.dreamstime.com/z/brown-yellow-eye-texture-black-fringe-brown-yellow-eye-texture-black-fringe-white-background-118702165.jpg",
layout="centered",
initial_sidebar_state="expanded")

page_bg_img='''
    <style>
    .stApp{
    background-image: url('https://media.istockphoto.com/id/184372817/photo/abstract-speed-motion-in-highway-tunnel.jpg?b=1&s=170667a&w=0&k=20&c=q53RBVBmQz7CO-2Iu1ikEm4d-hFAgDGcMGkjNoOTfbE=');
    background-size:cover;
    }
    </style>
    '''
st.markdown(page_bg_img,unsafe_allow_html=True)

colt1,colt2=st.columns(spec=[4,1],gap="small")
with colt1:
    st.write('<h1 style="color:crimson;">Accident Prevention System</h1>', 
    unsafe_allow_html=True)

with colt2:
    st.title(":sleeping: :blue_car:")

col1,col2,col2a=st.columns(spec=[1,1,1])
with col1:
    EAR_THRESH=st.slider("Eye Aspect ratio threshold",0.0,0.4,0.25,0.01)
with col2:
    wait_time=st.slider("Seconds to wait before alarm",0.0,5.0,1.0,0.25)
with col2a:
    MAR_THRESH=st.slider("Mouth Aspect ratio threshold",0.0,0.4,0.25,0.01)

thresholds={
"EAR_THRESH":EAR_THRESH,
"MAR_THRESH":MAR_THRESH,
"wait_time":wait_time}

col3,col4=st.columns(2,gap="medium")

with col3:
    known_distance=st.slider("Distance of the approaching vehicles(inch)",1,100,33)

with col4:
    known_width=st.slider("Width of the object approaching(inch)",1,100,60)

st.info("To Include Object Detection and proximity click the button or checkbox")
st.warning("Note: Proximity can be checked only when object detection is enabled")

col5,col6=st.columns(2,gap="medium")
with col5:
    val=st.checkbox("Detect Object")
with col6:
    proxy=st.checkbox("Include Proximity")

video_handler=VideoFrameHandler()
audio_handler=AudioFrameHandler("audio//Alarm.mp3")
if val:
    audio_handler=AudioFrameHandler("speech-1.mp3")
    
video_handler_object=ObjectVideoFrameHandler()
lock=threading.Lock()
shared_state= {"play_alarm": False}

def video_frame_callback(frame: av.VideoFrame):
    frame=frame.to_ndarray(format="bgr24")
    if val:
        frame,play_alarm=video_handler_object.process(frame,thresholds,known_distance,known_width,proxy)
    else:   
        frame,play_alarm=video_handler.process(frame,thresholds)
    with lock: 
        shared_state["play_alarm"]=play_alarm
            
    return av.VideoFrame.from_ndarray(frame,format="bgr24")
    
def audio_frame_callback(frame: av.AudioFrame):
    with lock:
        play_alarm=shared_state["play_alarm"]
    new_frame: av.AudioFrame=audio_handler.process(frame,play_sound=play_alarm)
    return new_frame
     
ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        #rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this to config for cloud deployment.
        #media_stream_constraints={"video": {"height": {"ideal": 480}}, "audio": True},
        #video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
    )

    