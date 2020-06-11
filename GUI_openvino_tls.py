 
import os
import cv2
import time
import glob 
import pymongo
import threading
import numpy as np
import logging as log
import multiprocessing as mp
from math import exp as exp
import tkinter as tk
from PIL import Image,ImageTk
from openvino.inference_engine import IENetwork, IECore
from bson.objectid import ObjectId

def GUIrefresh():
    global last_time,run_p,znum,ipic,t0,bGet
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
                mp.Process(name='mpSaveImage',target=mpSave, args=(imgname,frame),daemon=True).start()
                run_p,znum = run_p+1,znum+1
            if bSave and run_p>=run_all: set_state(True) #End
            # show error or image
            if ret:
                waittime = f'{int((1-run_p/run_all)*continue_time)}' if (bSave and run_p<run_all) else ''
                if bInference:
                    np.copyto(npImage, frame.reshape(640*480*3)) 
                    mpStep.value = 1
                    while mpStep.value > 0: time.sleep(0.001)
                    img = npImage.reshape(480,640,3)
                else:
                    img = frame
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            else:
                img = msg
            # GUI refresh 
            ipic += 1
            picturebox["image"]=img
            picturebox.image=img
            label_Time["text"] = f'     {mpTime.value:.2f}' if bInference else ""
            if imgname!="None": label_message["text"]=imgname
            if picturebox["text"]!=waittime: picturebox["text"]=waittime
            if time.time()-t0>2:
                iFPS,ipic,t0 = ipic/(time.time()-t0),0,time.time()
                if ret: label_FPS["text"]=f'     FPS:{iFPS:.2f}'
        else:time.sleep(0.001)
def Refreshmenu():
    global local_xml,DLS_xml
    _xml1,_xml2,_xml3 = sorted(glob.glob(r'*.xml')),sorted(glob.glob(r'.tls/*.xml')),[]
    for i in dls_db["Project"].find() :
        L = DLS_home+str(i['_id'])+DLS_end
        if os.path.exists(f'{L}.xml'):
            _xml3.append([i['name'],L])
    if _xml1+_xml2 != local_xml or _xml3!=DLS_xml:
        local_xml,DLS_xml = _xml1+_xml2,_xml3
        last = Tmenu.index("end")
        if last>0:Tmenu.delete(1,last)
        if len(local_xml)>0:Tmenu.add_separator()
        for n in _xml1:
            Tmenu.add_radiobutton(label=n[:-4],command=menu_click,variable=menuVar,value=n[:-4])
        for n in _xml2:
            Tmenu.add_radiobutton(label=n[5:-4],command=menu_click,variable=menuVar,value=n[:-4])
        if len(DLS_xml)>0:Tmenu.add_separator()
        for n1,n2 in DLS_xml:
            Tmenu.add_radiobutton(label=n1,command=menu_click,variable=menuVar,value=n2)
        if len(local_xml)==0 and len(DLS_xml)==0:
            empty = tk.Menu(root)
            root.config(menu=empty)
        elif root["menu"]!=menu0:
            root.config(menu=menu0)
    root.after(500,Refreshmenu) 
def Inference(mpIndex,mpStep,mpImage,mpTime,qxml):
    img = np.frombuffer(mpImage,dtype=np.uint8).reshape(480,640,3)
    ie,name = None,""
    while mpStep.value != -1:
        if mpIndex.value==1: 
            time.sleep(0.001)
            continue
        if qxml.qsize()>0:
            lastname,name = name,qxml.get()
            if name != "None" and lastname != name:
                if ie is not None: del ie,net,input_blob,out_blob,exec_net,labels_map,Nn,Nc,Nh,Nw
                ie,net,input_blob,out_blob,exec_net,labels_map = getmodel(f'{name}',"CPU",None)
                Nn, Nc, Nh, Nw = net.inputs[input_blob].shape
                if labels_map is None and DLS_end in name:
                    pbtxt = name.replace(DLS_end, DLS_label)
                    labels_map = readpbtxt(pbtxt) if os.path.exists(pbtxt) else readDLS(name)
        if mpStep.value == 1:
            tt0 = time.time()
            img2 = cv2.resize(img, (Nw, Nh))
            if Nc==1: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).reshape(Nh, Nw, 1)
            img2 = img2.transpose((2, 0, 1)).reshape(Nn, Nc, Nh, Nw)
            res = exec_net.infer(inputs={input_blob: img2})
            if len(res[out_blob][0].shape)==1:
                probs = np.squeeze(res[out_blob][0])
                index = np.argsort(probs)[::-1][0]
                label = labels_map[index] if (labels_map is not None and len(labels_map)>index) else f'#{index}'
                cv2.putText(img, f'{label}:{probs[index]:.2%}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1, cv2.LINE_AA)
            elif len(res[out_blob][0].shape)==2:
                print(res)
            elif len(res[out_blob][0].shape)==3:
                ih,iw = img.shape[:-1]
                objects = GetyoloAns(net,res,0.5, Nh, Nw) if "yolo" in name else res[out_blob][0][0]
                for obj in objects:
                    if obj[2] > 0.5:
                        xmin,ymin,xmax,ymax = np.int32(obj[3:]*[iw,ih,iw,ih])
                        index,prob = int(obj[1]),obj[2]
                        label = labels_map[index] if (labels_map is not None and len(labels_map)>index) else f'#{index}'
                        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                        cv2.putText(img,f'{label}:{prob:.2%}', (xmin+5,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                del objects
            del res,img2
            mpStep.value = 0
            mpTime.value = (time.time()-tt0)*1000
        else: time.sleep(0.001)
def yolov3(mpIndex,mpStep,mpImage,mpTime,qxml):
    img = np.frombuffer(mpImage,dtype=np.uint8).reshape(480,640,3)
    name = ""
    while mpStep.value != -1:
        if mpIndex.value==0: 
            time.sleep(0.001)
            continue
        if qxml.qsize()>0:
            lastname,name = name,qxml.get()
            if name != "None" and lastname != name:
                ie,net,input_blob,out_blob,exec_net,labels_map = getmodel(f'{name}',"CPU",None)
                Nn, Nc, Nh, Nw = net.inputs[input_blob].shape
        if mpStep.value == 1:
            tt0 = time.time()
            img2 = cv2.resize(img, (Nw, Nh)).transpose((2, 0, 1)).reshape(Nn, Nc, Nh, Nw)
            res = exec_net.infer(inputs={input_blob: img2})
            if "yolo" in name:
                ih,iw = img.shape[:-1]
                objects = GetyoloAns(net,res,0.5, Nh, Nw)
                for obj in objects:
                        xmin,ymin,xmax,ymax = np.int32(obj[3:]*[iw,ih,iw,ih])
                        index,prob = int(obj[1]),obj[2]
                        label = labels_map[index] if (labels_map is not None and len(labels_map)>index) else f'#{index}'
                        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                        cv2.putText(img,f'{label}:{prob:.2%}', (xmin+5,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                del objects
            del res,img2
            mpStep.value = 0
            mpTime.value = (time.time()-tt0)*1000
        else: time.sleep(0.001)
def mpSave(name,img):
    cv2.imwrite(name,img)
def GetFrame():
    global cap, ret, frame,bGet
    vsource = 0
    while bRuning:
        if qSource.qsize()>0:
            cap = cv2.VideoCapture(qSource.get()) 
        ret, frame = cap.read()
        bGet = True
        if ret:
            if frame.shape[0]!=640: 
                frame = cv2.resize(frame,(640,480))
        else:
            cap,vsource = cv2.VideoCapture(vsource),vsource+1 if vsource<10 else 0
            time.sleep(0.005)
def KeyPress(event=None):
    key = event.keysym
    if key=='s' or key=='space': save_click()
    elif key=='c' or key=='Return': continue_click()
    elif key=='q' or key=='Escape': Exit()
def save_click():
    global last_time,run_p,run_all,interval_time,znum
    if not bSave:
        interval_time,run_p,run_all,znum = 0,0,1,1
        last_time = time.time()-0.001
        set_state(False)
def continue_click():
    global last_time,run_p,run_all,interval_time,znum,zdir
    if not bSave:
        interval_time,run_p,run_all,znum,zdir = 1/image_sec,0,continue_time*image_sec,1,1
        last_time = time.time()-interval_time-0.001
        set_state(False)
        while os.path.isdir(f'{zdir:03}'): zdir+=1
def interval_scroll(v):
    global image_sec
    image_sec = int(v)
    scale_interval.config(label=f'save {v} image/sec')
def continue_scroll(v):
    global continue_time
    continue_time = int(v)
    scale_continue.config(label=f'continue save {v} sec')
def set_state(bstate):
    global bSave
    bSave = not bstate
    scale_interval['state'] = scale_continue['state'] = 'normal' if bstate else 'disable'
    button_save['state'] = button_continue['state'] = 'normal' if bstate else 'disable'
def menu_click():
    global bInference
    scmd = menuVar.get()
    if os.path.exists(f'{scmd}.xml') or scmd=="None":
        mpIndex.value = 1 if scmd.endswith("yolov3") else 0
        bInference = scmd!="None"
        qxml.put(scmd)
def dirfind(ndir,dlist,target,layer):
    if len(dlist)>0: return
    for s in os.listdir(ndir):
        if s[0] == '.': continue
        if target in s: dlist.append(os.path.join(ndir,s))
        newDir=os.path.join(ndir,s)
        if layer<6 and os.path.isdir(newDir): dirfind(newDir,dlist,target,layer+1)
    return dlist
def Exit():
    global bRuning
    bRuning,bInference = False,False
    mpStep.value = -1
    root.destroy()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def getmodel(name,device,cpu_extension):
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")
    # Read IR
    log.info(f'Loading network files:\n\t{name}.xml\n\t{name}.bin')
    net = IENetwork(model=f'{name}.xml', weights=f'{name}.bin')
    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)
    if os.path.exists(f'{name}.txt'):
        with open(f'{name}.txt', 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = readpbtxt(f'{name}.pbtxt')
    return ie,net,input_blob,out_blob,exec_net,labels_map
def readpbtxt(name):
    if not os.path.exists(name): return None
    log.info("Read pbtxt")
    with open(name, 'r') as f: txt = f.read()[1:].split("item")
    search = "display_name:" if "display_name:" in txt else "name:"
    pbid = [int(i.split('id:')[1].split()[0]) for i in txt]
    pbname = [i.split(search)[1].split('\"')[1] for i in txt]
    labels_map = [pbname[pbid.index(i)] if i in pbid else i for i in range(max(pbid)+1)]
    return labels_map
def readDLS(name):
    try:
        sid,Out = name.split("/")[-4],""
        x=dls_db["Project"].find_one({'_id':ObjectId(sid)})['config']['selectedLabels']
        labels_map = [dls_db["DatasetLabel"].find_one({'_id':ObjectId(sid)})['name'] for sid in x]
        del sid,Out,x
        return labels_map
    except:return None
class YoloParams:
    def __init__(self, param):
        self.num = int(param['num']) if 'num' in param else 3
        self.coords = int(param['coords']) if 'coords' in param else 4
        self.classes = int(param['classes']) if 'classes' in param else 80
        self.anchors = [float(a) for a in param['anchors'].split(',')] if 'anchors' in param else [10.0, 13.0, 16.0,
                       30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.
        if self.isYoloV3:
            mask = [int(idx)*2 for idx in param['mask'].split(',')]
            self.num = len(mask)
            self.anchors = np.array([[self.anchors[idx], self.anchors[idx+1]] for idx in mask]).flatten()
def parse_yolo_region(blob, params, threshold, net_h, net_w):
    grid_h, grid_w = blob.shape[1:]
    objects = list()
    blob = blob.reshape((params.num,-1,grid_h, grid_w))
    side_square = grid_h * grid_w
    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row,col = i // grid_w, i % grid_w
        for n in range(params.num):
            scale = blob[n,4,row,col]
            if scale < threshold: continue
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x,y,w,h = blob[n,:4,row,col]
            x,y = (col + x) / grid_w, (row + y) / grid_h
            # Value for exp is very big number in some cases so following construction is using here
            try:w_exp,h_exp = exp(w),exp(h)
            except OverflowError: continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (net_w if params.isYoloV3 else grid_w)
            h = h_exp * params.anchors[2 * n + 1] / (net_h if params.isYoloV3 else grid_h)
            blob[n,5:,row,col] = blob[n,5:,row,col]*scale
            prob = blob[n,5:,row,col]
            for j in range(params.classes):
                if prob[j] > threshold:
                    objects.append([0, j, prob[j], x-w/2, y-h/2, x+w/2, y+h/2])
    return objects
def intersection_over_union(box_1, box_2):
    xmin,ymin,xmax,ymax = box_1[3:]
    xmin2,ymin2,xmax2,ymax2 = box_2[3:]
    w_overlap = min(xmax, xmax2) - max(xmin, xmin2)
    h_overlap = min(ymax, ymax2) - max(ymin, ymin2)
    if (w_overlap < 0 or h_overlap < 0): return 0
    area_overlap = w_overlap * h_overlap
    box_1_area = (ymax - ymin) * (xmax - xmin)
    box_2_area = (ymax2 - ymin2) * (xmax2 - xmin2)
    area_union = box_1_area + box_2_area - area_overlap
    return 0 if area_union <= 0 else area_overlap / area_union
def GetyoloAns(net,output,threshold, net_h, net_w):
    objects = list()
    for layer_name, out_blob in output.items():
        layer_params = YoloParams(net.layers[layer_name].params)
        objects += parse_yolo_region(out_blob[0], layer_params, threshold, net_h, net_w)
    # Filtering overlapping boxes with respect to the --threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj[2], reverse=True)
    for i in range(len(objects)):
        if objects[i][2] == 0: continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > threshold:
                objects[j][2] = 0
    # Drawing objects with respect to the --threshold CLI parameter
    objects = [obj for obj in objects if obj[2] >= threshold]
    return np.array(objects)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO)
    DLS_home = dirfind("/home/",[],"NNFramework",0)[0]+"/tf/datasets/"
    DLS_end,DLS_label = "/model/frozen_FP32/frozen_inference_graph","/label/label.pbtxt"
    dls_db = pymongo.MongoClient('mongodb://localhost:27017/',connect=False)["dls"]
    # Start Process
    qSource,qxml = mp.Queue(10),mp.Queue(10)
    mpImage = mp.Array('b', 640*480*3, lock=False) # 'b' 1 byte
    mpStep,mpIndex,mpTime = mp.Value('i', 0),mp.Value('i', 0),mp.Value('f', 0.0)
    npImage = np.frombuffer(mpImage,dtype=np.uint8)
    mp.Process(name='Inference',target=Inference, args=(mpIndex,mpStep,mpImage,mpTime,qxml),daemon=True).start()
    mp.Process(name='yolov3',target=yolov3, args=(mpIndex,mpStep,mpImage,mpTime,qxml),daemon=True).start()
    # Start Tkinter
    root = tk.Tk()
    root.title("Capture tool")
    root.bind("<Key>",KeyPress)
    root.protocol("WM_DELETE_WINDOW", Exit)
    # add 
    scale_interval = tk.Scale(root, label='save 1 image/sec', from_=1, to=30, orient=tk.HORIZONTAL,length=480, showvalue=0, tickinterval=2, command=interval_scroll)
    scale_interval.grid(row=0,columnspan=2)
    scale_interval.set(10)
    scale_continue = tk.Scale(root, label='continue save 1 sec', from_=1, to=60, orient=tk.HORIZONTAL,length=480, showvalue=0, tickinterval=4, command=continue_scroll)
    scale_continue.grid(row=1,columnspan=2)
    scale_continue.set(10)
    picturebox = tk.Label(root,text = '', compound='center', font=("Times", 150),fg="white") 
    picturebox.grid(row=2,columnspan=2,padx=20)
    button_save = tk.Button(root,text = 'save image',width=30, height=5,command = save_click)
    button_save.grid(row=3,column=0,padx=20,pady=5,sticky='w')
    button_continue = tk.Button(root,text = 'continue save',width=30, height=5,command = continue_click)
    button_continue.grid(row=3,column=1,padx=20,pady=5,sticky='e')
    label_FPS = tk.Label(root,text = '') 
    label_FPS.grid(row=4,column=0,sticky='w')
    label_Time = tk.Label(root,text = '') 
    label_Time.grid(row=1,columnspan=2,sticky=tk.W+tk.S)
    label_message = tk.Label(root,text = '') 
    label_message.grid(row=4,column=1,sticky='e')
    # add Menu
    menu0 = tk.Menu(root)
    root.config(menu=menu0)
    menuVar = tk.StringVar()
    Tmenu = tk.Menu(menu0,tearoff=0)
    menu0.add_cascade(label="Read xml",menu=Tmenu)
    Tmenu.add_radiobutton(label="None",command=menu_click,variable=menuVar,value="None")
    local_xml,DLS_xml="",""
    Refreshmenu()
    # variable
    run_p,run_all,znum,zdir,image_sec,continue_timem,ipic=1,1,1,1,1,1,0
    last_time,t0 = time.time(),time.time()
    bRuning,bSave,bGet,bInference = True,False,False,False
    msg = np.zeros(640*480*3).reshape(480,640,3).astype(np.uint8)
    cv2.putText(msg , "Camera error" , (80,260), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
    msg = ImageTk.PhotoImage(image=Image.fromarray(msg))
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()
    # threading
    threading.Thread(target=GetFrame,daemon=True).start()
    threading.Thread(target=GUIrefresh,daemon=True).start()
    root.mainloop()
