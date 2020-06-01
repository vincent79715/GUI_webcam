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

def read_camera():
    global run_p,last_time,bGet,znum,zdir
    showtext1,showtext2='',''
    vsource = 0
    Error_times = 0
    while bRuning:
        if qFrame.qsize()>0:
            Error_times = 0
            t0 = time.time()
            ret, frame = qFrame.get()
            # save image
            t1 = time.time()
            if ret and bGet and run_all>run_p and time.time()-last_time > interval_time:
                last_time += interval_time
                sdir = f'{zdir:03}/' if run_all>1 else ''
                if not os.path.isdir(f'{zdir:03}') and run_all>1: os.mkdir(f'{zdir:03}')
                while os.path.exists(f'{sdir}{znum:04}.jpg'): znum+=1
                showtext1 = f'{sdir}{znum:04}.jpg'
                qSave.put([showtext1,frame*1])      
                run_p,znum = run_p+1,znum+1
            if bGet and run_p>=run_all: set_state(True) #End
            # show error or image
            t2 = time.time()
            if ret:
                showtext2 = f'{int((1-run_p/run_all)*continue_time)}' if (bGet and run_p<run_all) else ''
                qInference.put([ret,frame*1,showtext1,showtext2])
            else:
                qGUI.put([ret,[],'',''])
            if qGUI.qsize()>1: qGUI.get()
            t3 = time.time()
            if t3-t0>0.01:
                print(f'{qFrame.qsize()},{qGUI.qsize()},{qSave.qsize()},{threading.activeCount()} : {(t3-t0)*1000:6.2f} , {(t1-t0)*1000:6.2f} , {(t2-t1)*1000:6.2f} , {(t3-t2)*1000:6.2f}')
        else: 
            time.sleep(0.001)
            Error_times +=1
            if Error_times>1000:
                qGUI.put([False,[],'',''])
                Error_times=1000
def GUIrefresh():
    while bRuning:
        if qGUI.qsize()>0:
            ret,img,text1,text2 = qGUI.get()
            # show error or image
            if ret : 
                img = ImageTk.PhotoImage(img)
            else: 
                img = msg
            # GUI refresh 
            label_message.config(text=text1)
            picturebox.config(image=img,text=text2)
            picturebox.image=img
        else: time.sleep(0.001)
def SaveImage(qSave):
    while True:
        if qSave.qsize()>0:
            name,img = qSave.get()
            cv2.imwrite(name,img)
        else: time.sleep(0.001)
def GetFrame(qFrame):
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()
    vsource = 0
    while True:
        ret, frame = cap.read()
        if ret:
            qFrame.put([ret, frame])
        else:
            qFrame.put([ret, []])
            cap = cv2.VideoCapture(vsource)
            time.sleep(0.01)
            vsource += 1
            if vsource>10:vsource=0
        if qFrame.qsize()>2: qFrame.get()
def Inference(qInference,qxml,qGUI):
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
                print("Read xml")
            else:
                print("continue")
        if qInference.qsize()>0:
            ret,img,s1,s2 = qInference.get()
            if name != "None":
                img2 = cv2.resize(img, (Nw, Nh)
                if Nc==1: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY ).reshape(Nh, Nw, 1)
                img2 = img.transpose((2, 0, 1)).reshape(Nn, Nc, Nh, Nw)
                res = exec_net.infer(inputs={input_blob: img2})[out_blob][0]
                probs = np.squeeze(res)
                No1 = np.argsort(probs)[::-1][0]
                label = labels_map[No1] if labels_map else '#{}'.format(No1)
                cv2.putText(img, '{}:{:.2f}%'.format(label, probs[No1]*100), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 1, cv2.LINE_AA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            qGUI.put([ret,img,s1,s2])
        else: time.sleep(0.001)        
def KeyPress(event=None):
    key = event.keysym
    if key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
    elif key=='q' or key=='Escape': Exit()
def button_save_click():
    global last_time,run_p,run_all,interval_time,bGet,znum
    if not bGet:
        interval_time,run_p,run_all,znum = 0,0,1,1
        last_time = time.time()-0.001
        set_state(False)
def button_continue_click():
    global last_time,run_p,run_all,interval_time,zdir,znum
    if not bGet:
        interval_time,run_p,run_all,znum,zdir = 1/image_sec,0,continue_time*image_sec,1,zdir+1
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
    qxml.put(menuVar.get())
    print(menuVar.get())
def Exit():
    global bRuning
    bRuning = False 
    mpSave.terminate()
    mpFrame.terminate()
    mpInference.terminate()
    mpSave.join()
    mpFrame.join()
    mpInference.join()
    print("Exit mpSave、mpFrame、mpInference")
    while qGUI.qsize()>0: qGUI.get()
    print("Exit qGUI")
    while qSave.qsize()>0: qSave.get()
    print("Exit qSave")
    while qFrame.qsize()>0: qFrame.get()
    print("Exit qFrame")
    while qxml.qsize()>0: qxml.get()
    print("Exit qxml")
    while qInference.qsize()>0: qInference.get()
    print("Exit qInference")
    thread1.join()
    print("Exit thread1")
    time.sleep(0.2)
    print("Exit root")
    root.destroy()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",type=str)
    args.add_argument("-i", "--input", help="Path to an image/video file.Default value is cam", default="cam", type=str)
    args.add_argument("-l", "--cpu_extension",help="Optional. Required for CPU custom layers. ", type=str, default=None)
    args.add_argument("-d", "--device",help="CPU, GPU, FPGA, HDDL, MYRIAD or HETERO .Default value is CPU",default="CPU", type=str)
    args.add_argument("-lb", "--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    return parser
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

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)
    if labels:
        with open(labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    return ie,net,input_blob,out_blob,exec_net,labels_map
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    
    root = Tk()
    root.title("Capture tool")
    root.bind("<Key>",KeyPress)
    root.protocol("WM_DELETE_WINDOW", Exit)
    
    scale_interval = Scale(root, label='save 1 image/sec', from_=1, to=20, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=2, command=scale_interval_scroll)
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
    label_message = Label(root,text = '') 
    label_message.grid(row=4,columnspan=2,sticky='e')
    menu0 = Menu(root)
    root.config(menu=menu0)

    menuVar = StringVar()
    menuVar.set(1)
    editmenu = Menu(menu0,tearoff=0)
    editmenu.add_radiobutton(label="None",command=menu_click,variable=menuVar,value="None")
    _xml = glob.glob(r'*.xml')
    for n in _xml:
        n=n[:-4]
        editmenu.add_radiobutton(label=n,command=menu_click,variable=menuVar,value=n)
    menu0.add_cascade(label="編輯",menu=editmenu)
    
    run_p,run_all=1,1
    last_time = time.time()
    bRuning,bGet = True,False
    msg = np.zeros(640*480*3).reshape(480,640,3).astype(np.uint8)
    cv2.putText(msg , "Camera error" , (80,260), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
    msg = ImageTk.PhotoImage(image=Image.fromarray(msg))

    znum,zdir,image_sec,continue_time=1,0,1,1

    qGUI = mp.Queue()
    thGUI = threading.Thread(target=GUIrefresh,daemon=True)
    thGUI.start()
    
    qSave = mp.Queue()
    mpSave = mp.Process(name='SaveImage',target=SaveImage, args=(qSave,))
    mpSave.start()
    
    qFrame = mp.Queue()
    mpFrame = mp.Process(name='GetFrame',target=GetFrame, args=(qFrame,))
    mpFrame.start()
    
    qInference = mp.Queue()
    qxml = mp.Queue()
    mpInference = mp.Process(name='Inference',target=Inference, args=(qInference,qxml,qGUI))
    mpInference.start()
    
    thread1 = threading.Thread(target=read_camera,daemon=True)
    thread1.start()
 
    root.mainloop()
