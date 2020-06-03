import os
import cv2
import time
import glob 
import pymongo
import threading
import numpy as np
import logging as log
import multiprocessing as mp
from tkinter import *
from PIL import Image,ImageTk
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IENetwork, IECore

def assign_task():
    global last_time,run_p,znum,Error_times
    while bRuning:
        if qFrame.qsize()>0:
            Error_times = 0
            imgname = "None"
            ret, frame = qFrame.get()
            # save image
            if ret and bGet and run_all>run_p and time.time()-last_time > interval_time:
                last_time += interval_time
                sdir = f'{zdir:03}/' if run_all>1 else ''
                if not os.path.isdir(f'{zdir:03}') and run_all>1: os.mkdir(f'{zdir:03}')
                while os.path.exists(f'{sdir}{znum:04}.jpg'): znum+=1
                imgname = f'{sdir}{znum:04}.jpg'
                if znum%2==0: qSave.put([imgname,frame*1]) 
                else: threading.Thread(target=threadSave, args = (imgname,frame*1),daemon=True).start()
                run_p,znum = run_p+1,znum+1
            if bGet and run_p>=run_all: set_state(True) #End
            # show error or image
            if ret:
                s2 = f'{int((1-run_p/run_all)*continue_time)}' if (bGet and run_p<run_all) else ''
                qInference.put([ret,frame*1,imgname,s2])
                if qInference.qsize()>1: qInference.get()
            else: qUI.put([False,[],'',''])
        elif Error_times>1000: qUI.put([False,[],'',''])
        else: Error_times +=1
        time.sleep(0.001)
def Inference(qInference,qxml,qUI):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    name = "None"
    while True:
        if qxml.qsize()>0:
            name = qxml.get()
            if name != "None":
                model_xml = name + ".xml"
                model_bin = name + ".bin"
                model_labels = name + ".txt"
                ie,net,input_blob,out_blob,exec_net,labels_map = getmodel(model_xml,model_bin,"CPU",None,model_labels,log)
                Nn, Nc, Nh, Nw = net.inputs[input_blob].shape
        if qInference.qsize()>0:
            ret,img,s1,s2 = qInference.get()
            if name != "None":
                img2 = cv2.resize(img, (Nw, Nh))
                if Nc==1: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY ).reshape(Nh, Nw, 1)
                img2 = img2.transpose((2, 0, 1)).reshape(Nn, Nc, Nh, Nw)
                res = exec_net.infer(inputs={input_blob: img2})[out_blob][0]
                if len(res.shape)==1:
                    probs = np.squeeze(res)
                    No1 = np.argsort(probs)[::-1][0]
                    label = labels_map[No1] if labels_map else '#{}'.format(No1)
                    cv2.putText(img, '{}:{:.2f}%'.format(label, probs[No1]*100), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1, cv2.LINE_AA)
                elif len(res.shape)==2:
                    print(res)
                elif len(res.shape)==3:
                    res = res[0]
                    ih,iw = img.shape[:-1]
                    for obj in res:
                        if obj[2] > 0.5:
                            xmin,ymin,xmax,ymax = np.int32(obj[3:]*[iw,ih,iw,ih])
                            index,prob = int(obj[1]-1),obj[2]*100
                            label = labels_map[index] if labels_map else '#{}'.format(index)
                            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                            cv2.putText(img, '{}:{:.2f}%'.format(label, prob), (xmin+5,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            qUI.put([ret,img,s1,s2])
        else: time.sleep(0.001)    
def UIrefresh():
    iFPS,ipic = 0,0
    t0,t1 = time.time(),time.time()
    while bRuning:
        if qUI.qsize()>0:
            while qUI.qsize()>1: qUI.get()
            ret,img,text1,text2 = qUI.get()
            # show error or image
            img = ImageTk.PhotoImage(img) if ret else msg
            ipic += 1
            # GUI refresh 
            if text1!="None":label_message.config(text=text1)
            picturebox.config(image=img,text=text2)
            picturebox.image=img
            t1 = time.time()
            if t1-t0>2:
                iFPS = ipic/(t1-t0)
                t0 = time.time()
                ipic = 0
                if ret: label_FPS.config(text=f'FPS:{iFPS:.2f}')
        else: time.sleep(0.001)

def mpSaveImage(qSave):
    while True:
        if qSave.qsize()>0:
            name,img = qSave.get()
            cv2.imwrite(name,img)
        else: time.sleep(0.001)
def threadSave(name,img):
    cv2.imwrite(name,img)
def GetFrame(qFrame,qSource):
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()
    vsource,fps = 0,0
    lastput = time.time()
    while True:
        if qSource.qsize()>0:
            cap = cv2.VideoCapture(qSource.get()) 
            fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if ret:
            if frame.shape[0]!=640: frame = cv2.resize(frame,(640,480))
            if fps>0:
                t = 0.015-(time.time()-lastput)
                if t>0 and t<0.02: time.sleep(t)
            qFrame.put([ret, frame])
            lastput = time.time()
        else:
            qFrame.put([ret, []])
            cap = cv2.VideoCapture(vsource)
            time.sleep(0.01)
            fps,vsource = 0, vsource+1
            if vsource>10: vsource=0
        if qFrame.qsize()>1: qFrame.get()
def KeyPress(event=None):
    key = event.keysym
    if key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
    elif key=='q' or key=='Escape': Exit()
def button_save_click():
    global last_time,run_p,run_all,interval_time,znum
    if not bGet:
        interval_time,run_p,run_all,znum = 0,0,1,1
        last_time = time.time()-0.001
        set_state(False)
def button_continue_click():
    global last_time,run_p,run_all,interval_time,znum,zdir
    if not bGet:
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
    global bGet
    if bstate:
        bGet=False
        scale_interval['state'] = 'normal'
        scale_continue['state'] = 'normal'
        button_save['state'] = 'normal'
        button_continue['state'] = 'normal'
    else:
        bGet=True
        scale_interval['state'] = 'disable'
        scale_continue['state'] = 'disable'
        button_save['state'] = 'disable'
        button_continue['state'] = 'disable'
def menu_click():
    scmd = menuVar.get()
    if os.path.exists(f'{scmd}.xml'):
        qxml.put(scmd)
def Refreshmenu():
    global local_xml,DLS_xml
    _xml1,_xml2 = glob.glob(r'*.xml'),[]
    db = pymongo.MongoClient('mongodb://localhost:27017/')["dls"]["Project"]
    x = db.find()
    for i in x :
        L = DLS_home+str(i['_id'])+"/model/frozen_FP32/frozen_inference_graph"
        if os.path.exists(f'{L}.xml'):
            _xml2.append([i['name'],L])
    if _xml1 != local_xml or _xml2!=DLS_xml:
        local_xml,DLS_xml = _xml1,_xml2
        last = Tmenu.index("end")
        if last>1:Tmenu.delete(2,last)
        for n in local_xml:
            n=n[:-4]
            Tmenu.add_radiobutton(label=n,command=menu_click,variable=menuVar,value=n)
        Tmenu.add_separator()
        for n1,n2 in DLS_xml:
            Tmenu.add_radiobutton(label=n1,command=menu_click,variable=menuVar,value=n2)
    root.after(500,Refreshmenu) 
def dirfind(Dir,Ans,target,layer):
    if len(Ans)>0:return
    for s in os.listdir(Dir):
        if s[0] == '.': continue
        if target in s: Ans.append(os.path.join(Dir,s))
        newDir=os.path.join(Dir,s)
        if layer<6 and os.path.isdir(newDir): dirfind(newDir,Ans,target,layer+1)
    return Ans
def Exit():
    global bRuning,qUI,qSave,qFrame,qSource,qInference,qxml
    bRuning = False 
    mpSave.terminate()
    mpFrame.terminate()
    mpInference.terminate()
    mpSave.join()
    mpFrame.join()
    mpInference.join()
    del qUI,qSave,qFrame,qSource,qInference,qxml
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
    if len(net.inputs.keys()) == 1: log.warning("Sample supports only single input topologies")
    if len(net.outputs) == 1: log.warning("Sample supports only single output topologies")

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
    root = Tk()
    root.title("Capture tool")
    root.bind("<Key>",KeyPress)
    root.protocol("WM_DELETE_WINDOW", Exit)
    
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

    menu0 = Menu(root)
    root.config(menu=menu0)
    menuVar = StringVar()
    menuVar.set(1)
    Tmenu = Menu(menu0,tearoff=0)
    menu0.add_cascade(label="Read xml",menu=Tmenu)
    Tmenu.add_radiobutton(label="None",command=menu_click,variable=menuVar,value="None")
    Tmenu.add_separator()
    local_xml,DLS_xml="",""
    DLS_home = dirfind("/home/",[],"NNFramework",0)[0]+"/tf/datasets/"
    Refreshmenu()

    run_p,run_all,znum,zdir,image_sec,continue_timem,Error_times=1,1,1,1,1,1,1
    last_time = time.time()
    bRuning,bGet = True,False
    msg = np.zeros(640*480*3).reshape(480,640,3).astype(np.uint8)
    cv2.putText(msg , "Camera error" , (80,260), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
    msg = ImageTk.PhotoImage(image=Image.fromarray(msg))

    qUI,qSave,qFrame,qSource,qInference,qxml = mp.Queue(),mp.Queue(),mp.Queue(),mp.Queue(),mp.Queue(),mp.Queue()
    threading.Thread(target=assign_task,daemon=True).start()
    threading.Thread(target=UIrefresh,daemon=True).start()
    mpSave = mp.Process(name='mpSaveImage',target=mpSaveImage, args=(qSave,));
    mpSave.start();
    mpFrame = mp.Process(name='GetFrame',target=GetFrame, args=(qFrame,qSource));
    mpFrame.start()
    mpInference = mp.Process(name='Inference',target=Inference, args=(qInference,qxml,qUI));
    mpInference.start()
    
    root.mainloop()
