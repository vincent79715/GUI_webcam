import os
import cv2
import time
import glob 
import threading
import numpy as np
import logging as log
import multiprocessing as mp
from tkinter import *
from PIL import Image,ImageTk
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IENetwork, IECore

def GUIrefresh():
    global last_time,run_p,znum,ipic,t0,bGet,bmpInference,ampInference
    while bRuning:
        if bGet:
            imgname,waittime,bGet = "None","",False
            # save image
            if ret and bSave and run_all>run_p and time.time()-last_time > interval_time:
                last_time += interval_time
                sdir = f'{zdir:03}/' if run_all>1 else ''
                if not os.path.isdir(f'{zdir:03}') and run_all>1: os.mkdir(f'{zdir:03}')
                while os.path.exists(f'{sdir}{znum:04}.jpg'): znum+=1
                imgname = f'{sdir}{znum:04}.jpg'
                mp.Process(name='mpSaveImage',target=mpSave, args=(imgname,frame)).start()
                run_p,znum = run_p+1,znum+1
            if bSave and run_p>=run_all: set_state(True) #End
            # show error or image
            if ret:
                waittime = f'{int((1-run_p/run_all)*continue_time)}' if (bSave and run_p<run_all) else ''
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)
            else:
                img = msg
            # GUI refresh 
            ipic += 1
            picturebox["image"]=img
            picturebox.image=img
            if imgname!="None": label_message["text"]=imgname
            if picturebox["text"]!=waittime: picturebox["text"]=waittime
            if time.time()-t0>2:
                iFPS,ipic = ipic/(time.time()-t0),0
                t0 = time.time()
                if ret: label_FPS["text"]=f'FPS:{iFPS:.2f}'
        else:time.sleep(0.001)
def mpSave(name,img):
    cv2.imwrite(name,img)
def GetFrame():
    global cap, ret, frame,bGet
    vsource,fps = 0,0
    lastput = time.time()
    while bRuning:
        ret, frame = cap.read()
        bGet = True
        if ret:
            if frame.shape[0]!=640: frame = cv2.resize(frame,(640,480))
            if fps>0:
                t = 0.015-(time.time()-lastput)
                if t>0 and t<0.02: time.sleep(t)
            lastput = time.time()
        else:
            cap = cv2.VideoCapture(vsource)
            time.sleep(0.01)
            fps,vsource = 0, vsource+1
            if vsource>10: vsource=0
def KeyPress(event=None):
    key = event.keysym
    if key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
    elif key=='q' or key=='Escape': Exit()
def button_save_click():
    global last_time,run_p,run_all,interval_time,znum
    if not bSave:
        interval_time,run_p,run_all,znum = 0,0,1,1
        last_time = time.time()-0.001
        set_state(False)
def button_continue_click():
    global last_time,run_p,run_all,interval_time,znum,zdir
    if not bSave:
        interval_time,run_p,run_all,znum,zdir = 1/image_sec,0,continue_time*image_sec,1,1
        last_time = time.time()-interval_time-0.001
        set_state(False)
        while os.path.isdir(f'{zdir:03}'): zdir+=1
def scale_interval_scroll(v):
    global image_sec
    image_sec = int(v)
    scale_interval.config(label=f'save {v} image/sec')
def scale_continue_scroll(v):
    global continue_time
    continue_time = int(v)
    scale_continue.config(label=f'continue save {v} sec')
def set_state(bstate):
    global bSave
    if bstate:
        bSave=False
        scale_interval['state'] = 'normal'
        scale_continue['state'] = 'normal'
        button_save['state'] = 'normal'
        button_continue['state'] = 'normal'
    else:
        bSave=True
        scale_interval['state'] = 'disable'
        scale_continue['state'] = 'disable'
        button_save['state'] = 'disable'
        button_continue['state'] = 'disable'
def menu_click():
    global bInference
    scmd = menuVar.get()
    if os.path.exists(f'{scmd}.xml') or scmd=="None":
        bInference = scmd!="None"
        qxml.put(scmd)
def dirfind(Dir,Ans,target,layer):
    if len(Ans)>0:return
    for s in os.listdir(Dir):
        if s[0] == '.': continue
        if target in s: Ans.append(os.path.join(Dir,s))
        newDir=os.path.join(Dir,s)
        if layer<6 and os.path.isdir(newDir): dirfind(newDir,Ans,target,layer+1)
    return Ans
def Exit():
    global bRuning
    bRuning = False 
    time.sleep(0.2)
    root.destroy()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def getmodel(model_xml,model_bin,device,cpu_extension,labels,log):
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    if len(net.inputs.keys()) != 1: log.warning("Sample supports only single input topologies")
    if len(net.outputs) != 1: log.warning("Sample supports only single output topologies")
    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)
    if labels and os.path.exists(labels):
        with open(labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    return ie,net,input_blob,out_blob,exec_net,labels_map
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
if __name__ == '__main__':
    # Start Tkinter
    root = Tk()
    root.title("Capture tool")
    root.bind("<Key>",KeyPress)
    root.protocol("WM_DELETE_WINDOW", Exit)
    # add 
    scale_interval = Scale(root, label='save 1 image/sec', from_=1, to=30, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=2, command=scale_interval_scroll)
    scale_interval.grid(row=0,columnspan=2)
    scale_interval.set(10)
    scale_continue = Scale(root, label='continue save 1 sec', from_=1, to=60, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=4, command=scale_continue_scroll)
    scale_continue.grid(row=1,columnspan=2)
    scale_continue.set(10)
    picturebox = Label(root,text = '', compound='center', font=("Times", 150),fg="white") 
    picturebox.grid(row=2,columnspan=2,padx=20)
    button_save = Button(root,text = 'save image',width=30, height=5,command = button_save_click)
    button_save.grid(row=3,column=0,padx=20,pady=5,sticky='w')
    button_continue = Button(root,text = 'continue save',width=30, height=5,command = button_continue_click)
    button_continue.grid(row=3,column=1,padx=20,pady=5,sticky='e')
    label_FPS = Label(root,text = '') 
    label_FPS.grid(row=4,column=0,sticky='w')
    label_message = Label(root,text = '') 
    label_message.grid(row=4,column=1,sticky='e')
    # variable
    run_p,run_all,znum,zdir,image_sec,continue_timem,ipic=1,1,1,1,1,1,0
    last_time,t0 = time.time(),time.time()
    bRuning,bSave,bGet = True,False,False
    msg = np.zeros(640*480*3).reshape(480,640,3).astype(np.uint8)
    cv2.putText(msg , "Camera error" , (80,260), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
    msg = ImageTk.PhotoImage(image=Image.fromarray(msg))
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()
    # threading
    threading.Thread(target=GetFrame,daemon=True).start()
    threading.Thread(target=GUIrefresh,daemon=True).start()
    root.mainloop()
