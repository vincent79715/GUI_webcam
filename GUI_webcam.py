import os
import cv2
import time
import threading
import numpy as np
import multiprocessing as mp
from tkinter import *
from PIL import Image,ImageTk

def read_camera():
    global cap,run_p,last_time,bGet,inum,zdir,q
    showtext1,showtext2='',''
    vsource = 0
    while bRuning:
        t0 = time.time()
        ret, frame = cap.read()
        # save image
        t1 = time.time()
        if ret and bGet and run_all>run_p and time.time()-last_time > interval_time:
            last_time += interval_time
            sdir = f'{zdir:03}/' if run_all>1 else ''
            if not os.path.isdir(f'{zdir:03}') and run_all>1: os.mkdir(f'{zdir:03}')
            while os.path.exists(f'{sdir}{inum:04}.jpg'): inum+=1
            showtext1 = f'{sdir}{inum:04}.jpg'
            qsave.put([showtext1,frame*1])      
            run_p,inum = run_p+1,inum+1
        if bGet and run_p>=run_all: set_state(True) #End
        # show error or image
        t2 = time.time()
        if not bRuning: return
        if ret:
            showtext2 = f'{int((1-run_p/run_all)*continue_time)}' if (bGet and run_p<run_all) else ''
            qGUI.put([ret,cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),showtext1,showtext2])
        else:
            qGUI.put([ret,[],' ',' '])
            cap = cv2.VideoCapture(vsource)
            cv2.waitKey(50)
            vsource += 1
            if vsource>10:vsource=0
        if qGUI.qsize()>1: qGUI.get()
        t3 = time.time()
        print(f'{qGUI.qsize()},{qsave.qsize()},{threading.activeCount()} : {(t3-t0)*1000:6.2f} , {(t1-t0)*1000:6.2f} , {(t2-t1)*1000:6.2f} , {(t3-t2)*1000:6.2f}')
def GUIrefresh():
    global qGUI
    while bRuning:
        if qGUI.qsize()>0:
            ret,img,text1,text2 = qGUI.get()
            # show error or image
            if ret : 
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)
            else: 
                img = msg
            # GUI refresh 
            label_message.config(text=text1)
            picturebox.config(image=img,text=text2)
            picturebox.image=img
        else: cv2.waitKey(5)
def Save_image():
    global qsave
    while bRuning:
        if qsave.qsize()>0:
            name,img = qsave.get()
            cv2.imwrite(name,img)
        else: cv2.waitKey(5)
def KeyPress(event=None):
    key = event.keysym
    if key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
    elif key=='q' or key=='Escape': Exit()
def button_save_click():
    global last_time,run_p,run_all,interval_time,bGet,inum
    if not bGet:
        interval_time,run_p,run_all,inum = 0,0,1,1
        last_time = time.time()-0.001
        set_state(False)
def button_continue_click():
    global last_time,run_p,run_all,interval_time,zdir,inum
    if not bGet:
        interval_time,run_p,run_all,inum,zdir = 1/image_sec,0,continue_time*image_sec,1,zdir+1
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
def Exit():
    global bRuning
    bRuning = False 
    time.sleep(0.1)
    while qGUI.qsize()>0: qGUI.get()
    while qsave.qsize()>0: qsave.get()
    time.sleep(0.1)
    cap.release()
    root.destroy()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
if __name__ == '__main__':
    root = Tk()
    root.title("Capture tool")
    root.bind("<Key>",KeyPress)
    root.protocol("WM_DELETE_WINDOW", Exit)
    
    scale_interval = Scale(root, label='save 1 image/sec', from_=1, to=10, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=1, command=scale_interval_scroll)
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

    run_p,run_all=1,1
    last_time = time.time()
    bRuning,bGet = True,False
    msg = np.zeros(640*480*3).reshape(480,640,3).astype(np.uint8)
    cv2.putText(msg , "Camera error" , (80,260), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
    msg = ImageTk.PhotoImage(image=Image.fromarray(msg))

    cap = cv2.VideoCapture(0) 
    inum,zdir,image_sec,continue_time=1,0,1,1

    qGUI = mp.Queue()
    qsave = mp.Queue()
    thread1 = threading.Thread(target=read_camera,daemon=True)
    thread2 = threading.Thread(target=GUIrefresh,daemon=True)
    thread3 = threading.Thread(target=Save_image,daemon=True)
    thread1.start()
    thread2.start()
    thread3.start()

    root.mainloop()
