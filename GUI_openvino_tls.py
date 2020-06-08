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
from tkinter import *
from PIL import Image,ImageTk
from argparse import ArgumentParser, SUPPRESS
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
                mp.Process(name='mpSaveImage',target=mpSave, args=(imgname,frame)).start()
                run_p,znum = run_p+1,znum+1
            if bSave and run_p>=run_all: set_state(True) #End
            # show error or image
            if ret:
                waittime = f'{int((1-run_p/run_all)*continue_time)}' if (bSave and run_p<run_all) else ''
                if bInference:
                    np.copyto(npInference, frame.reshape(640*480*3)) 
                    bmpInference.value = 1
                    while bmpInference.value == 1: time.sleep(0.001)
                    img = npInference.reshape(480,640,3)
                else:
                    img = frame
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
def Refreshmenu():
    global local_xml,DLS_xml
    _xml1,_xml2,_xml3 = glob.glob(r'*.xml'),glob.glob(r'.tls/*.xml'),[]
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
            n=n[:-4]
            Tmenu.add_radiobutton(label=n,command=menu_click,variable=menuVar,value=n)
        for n in _xml2:
            n=n[:-4]
            Tmenu.add_radiobutton(label=n[5:],command=menu_click,variable=menuVar,value=n)
        if len(DLS_xml)>0:Tmenu.add_separator()
        for n1,n2 in DLS_xml:
            Tmenu.add_radiobutton(label=n1,command=menu_click,variable=menuVar,value=n2)
        if len(local_xml)==0 and len(DLS_xml)==0:
            empty = Menu(root)
            root.config(menu=empty)
        else:
            root.config(menu=menu0)
    root.after(500,Refreshmenu) 
def Inference(cindex,b,ampInference,qxml):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    img = np.frombuffer(ampInference,dtype=np.uint8).reshape(480,640,3)
    dls_db = pymongo.MongoClient('mongodb://localhost:27017/')["dls"]
    ie = None
    name = ""
    while b.value != -1:
        if cindex.value==1: 
            time.sleep(0.001)
            continue
        if qxml.qsize()>0:
            lastname = name
            name = qxml.get()
            if name != "None" and lastname != name:
                if ie is not None: del ie,net,input_blob,out_blob,exec_net,labels_map,Nn,Nc,Nh,Nw
                ie,net,input_blob,out_blob,exec_net,labels_map = getmodel(f'{name}.xml',f'{name}.bin',"CPU",None,f'{name}.txt',log)
                Nn, Nc, Nh, Nw = net.inputs[input_blob].shape
                if labels_map is None and DLS_end in name:
                    sid,Out = name.split("/")[-4],""
                    x=dls_db["Project"].find_one({'_id':ObjectId(sid)})['config']['selectedLabels']
                    labels_map = [dls_db["DatasetLabel"].find_one({'_id':ObjectId(sid)})['name'] for sid in x]
                    del sid,Out,x
        if b.value == 1:
            tt0 = time.time()
            img2 = cv2.resize(img, (Nw, Nh))
            if Nc==1: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).reshape(Nh, Nw, 1)
            img2 = img2.transpose((2, 0, 1)).reshape(Nn, Nc, Nh, Nw)
            res = exec_net.infer(inputs={input_blob: img2})
            if len(res[out_blob][0].shape)==1:
                probs = np.squeeze(res[out_blob][0])
                No1 = np.argsort(probs)[::-1][0]
                label = labels_map[No1] if labels_map else '#{}'.format(No1)
                cv2.putText(img, '{}:{:.2f}%'.format(label, probs[No1]*100), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1, cv2.LINE_AA)
            elif "yolo" in name:
                objects = GetyoloAns(net,res,0.5,img.shape,img2.shape)
                for obj in objects:
                        xmin,ymin,xmax,ymax = obj['xmin'],obj['ymin'],obj['xmax'], obj['ymax']
                        index,prob = obj['class_id'],obj['confidence']*100
                        label = labels_map[index] if labels_map else '#{}'.format(index)
                        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                        cv2.putText(img, '{}:{:.2f}%'.format(label, prob), (xmin+5,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                del objects
            elif len(res[out_blob][0].shape)==2:
                print(res)
            elif len(res[out_blob][0].shape)==3:
                ih,iw = img.shape[:-1]
                for obj in res[out_blob][0][0]:
                    if obj[2] > 0.5:
                        xmin,ymin,xmax,ymax = np.int32(obj[3:]*[iw,ih,iw,ih])
                        index,prob = int(obj[1]-1),obj[2]*100
                        label = labels_map[index] if labels_map else '#{}'.format(index)
                        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                        cv2.putText(img, '{}:{:.2f}%'.format(label, prob), (xmin+5,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
            del res,img2
            b.value = 0
            print("Inference",(time.time()-tt0)*1000)
        else: time.sleep(0.001)
def yolov3(cindex,b,ampInference,qxml):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    img = np.frombuffer(ampInference,dtype=np.uint8).reshape(480,640,3)
    dls_db = pymongo.MongoClient('mongodb://localhost:27017/')["dls"]
    name = ""
    while b.value != -1:
        if cindex.value==0: 
            time.sleep(0.001)
            continue
        if qxml.qsize()>0:
            lastname = name
            name = qxml.get()
            if name != "None" and lastname != name:
                ie,net,input_blob,out_blob,exec_net,labels_map = getmodel(f'{name}.xml',f'{name}.bin',"CPU",None,f'{name}.txt',log)
                Nn, Nc, Nh, Nw = net.inputs[input_blob].shape
        if b.value == 1:
            tt0 = time.time()
            img2 = cv2.resize(img, (Nw, Nh))
            if Nc==1: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).reshape(Nh, Nw, 1)
            img2 = img2.transpose((2, 0, 1)).reshape(Nn, Nc, Nh, Nw)
            res = exec_net.infer(inputs={input_blob: img2})
            if "yolo" in name:
                objects = GetyoloAns(net,res,0.5,img.shape,img2.shape)
                for obj in objects:
                        xmin,ymin,xmax,ymax = obj['xmin'],obj['ymin'],obj['xmax'], obj['ymax']
                        index,prob = obj['class_id'],obj['confidence']*100
                        label = labels_map[index] if labels_map else '#{}'.format(index)
                        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                        cv2.putText(img, '{}:{:.2f}%'.format(label, prob), (xmin+5,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                del objects
            del res,img2
            b.value = 0
            print("Inference",(time.time()-tt0)*1000)
        else: time.sleep(0.001)
def mpSave(name,img):
    cv2.imwrite(name,img)
def GetFrame():
    global cap, ret, frame,bGet
    vsource,fps = 0,0
    lastput = time.time()
    while bRuning:
        if qSource.qsize()>0:
            cap = cv2.VideoCapture(qSource.get()) 
            fps = cap.get(cv2.CAP_PROP_FPS)
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
        cmpindex.value = 1 if scmd.endswith("yolov3") else 0
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
    global bRuning,bInference,qSource,qxml
    bRuning,bInference = False,False
    time.sleep(0.4)
    bmpInference.value = -1
    mpInference.join()
    del qSource,qxml
    time.sleep(0.1)
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
class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)
            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors
        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.
    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]
def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)
def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)
def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)
    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side
    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects
def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union
def GetyoloAns(net,output,threshold,frameshape,in_frameshape):
    objects = list()
    for layer_name, out_blob in output.items():
        layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
        objects += parse_yolo_region(out_blob, in_frameshape[2:], frameshape[:-1], layer_params, threshold)
    # Filtering overlapping boxes with respect to the --threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > threshold:
                objects[j]['confidence'] = 0

    # Drawing objects with respect to the --threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= threshold]
    return objects
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
if __name__ == '__main__':
    DLS_home = dirfind("/home/",[],"NNFramework",0)[0]+"/tf/datasets/"
    DLS_end = "/model/frozen_FP32/frozen_inference_graph"
    dls_db = pymongo.MongoClient('mongodb://localhost:27017/')["dls"]
    # Start Process
    qSource,qxml = mp.Queue(10),mp.Queue(10)
    ampInference = mp.Array('b', 640*480*3, lock=False) # 'b' 1 byte
    bmpInference = mp.Value('i', 0)
    cmpindex = mp.Value('i', 0)
    npInference = np.frombuffer(ampInference,dtype=np.uint8)
    mpInference = mp.Process(name='Inference',target=Inference, args=(cmpindex,bmpInference,ampInference,qxml))
    mpInference.start()
    mpyolo = mp.Process(name='yolov3',target=yolov3, args=(cmpindex,bmpInference,ampInference,qxml))
    mpyolo.start()
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
    # add Menu
    menu0 = Menu(root)
    root.config(menu=menu0)
    menuVar = StringVar()
    menuVar.set(1)
    Tmenu = Menu(menu0,tearoff=0)
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
