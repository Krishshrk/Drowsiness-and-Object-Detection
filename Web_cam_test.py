from streamlit_webrtc import webrtc_streamer,VideoProcessorBase
import streamlit as st
import av
import threading
from matplotlib import pyplot as plt
import cv2

lock=threading.Lock()
img_container={"img":None}

val=st.checkbox("Flip it")
plot=st.checkbox("Plot it")

        
def video_flip(frame):
    img=frame.to_ndarray(format="bgr24")
    flip=img[::-1,:,:] if val else img
    return av.VideoFrame.from_ndarray(flip,format="bgr24")
        
    
    
def plot_graph(frame):
    img=frame.to_ndarray(format="bgr24")
    img=cv2.flip(img,1)
    with lock:
        img_container["img"]=img
    return av.VideoFrame.from_ndarray(img,format="bgr24")

ctx=webrtc_streamer(key="flip",video_frame_callback=plot_graph)
fig_place=st.empty()
fig,ax=plt.subplots(1,1)

while ctx.state.playing:
    with lock:
        img=img_container["img"]
    if img is None:
        continue
        
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ax.cla()
    ax.hist(gray.ravel(),256,[0,256])
    fig_place.pyplot(fig)
    
