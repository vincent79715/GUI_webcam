import os
import cv2
import time
import threading
import numpy as np
import multiprocessing as mp
from tkinter import *
from PIL import Image,ImageTk

def read_camera():
    global cap,run_p,last_time,bGet,z
    showtext1,showtext2='',''
    while bRuning:
        
        t0 = time.time()
        ret, frame = cap.read()
        # save image
        t1 = time.time()
        if not bRuning: break
        if ret and bGet and run_all>run_p and time.time()-last_time > interval_time:
            last_time += interval_time
            showtext1 = f'{z:04}.jpg'
            threading.Thread(target=Save_image, args=(showtext1,frame*1)).start()          
            run_p,z = run_p+1,z+1
        if bGet and run_p>=run_all:
            bGet = False
            set_state(True)
            showtext2 = ''          
        # show error or image
        t2 = time.time()
        if not bRuning: break
        if ret:
            if bGet and run_all>1 and run_p>1:
                T = int((1-run_p/run_all)*100)
                showtext2 = f'{T}'
            threading.Thread(target=GUIrefresh, args=(ret,cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),showtext1,showtext2),daemon=True).start() 
        else:
            threading.Thread(target=GUIrefresh, args=(ret,msg,showtext1,showtext2),daemon=True).start() 
            cap = cv2.VideoCapture(0)
        t3 = time.time()
        print(f'{threading.activeCount()} : {(t3-t0)*1000:5.1f} , {(t1-t0)*1000:5.1f} , {(t2-t1)*1000:5.1f} , {(t3-t2)*1000:5.1f}')
def GUIrefresh(bSucess,img,text1,text2):
    # show error or image
    if not bRuning: return
    if bSucess :
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
    # GUI refresh 
    if not bRuning: return
    label_message.config(text=text1)
    picturebox.config(image=img,text=text2)
    picturebox.image=img

def Save_image(name,img):
    cv2.imwrite(name,img)
def KeyPress(event=None):
    key = event.keysym
    if key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
    elif key=='q' or key=='Escape': Exit()
def button_save_click():
    global start_time,last_time,run_p,run_all,interval_time,bGet
    if not bGet:
        start_time,last_time = time.time(),time.time()-0.001
        interval_time,run_p,run_all = 0,0,1
        bGet = True
def button_continue_click():
    global start_time,last_time,run_p,run_all,interval_time,bGet
    if not bGet:
        set_state(False)
        start_time,last_time = time.time(),time.time()-0.001
        interval_time,run_p,run_all = 1/image_sec,0,continue_time*image_sec
        bGet = True
def scale_interval_scroll(v):
    global image_sec
    image_sec = int(v)
    scale_interval.config(label=f'save {v} image/sec')
def scale_continue_scroll(v):
    global continue_time
    continue_time = int(v)
    scale_continue.config(label=f'continue save {v} sec')
def set_state(bstate):
    if bstate:
        scale_interval['state'] = 'normal'
        scale_continue['state'] = 'normal'
        button_save['state'] = 'normal'
        button_continue['state'] = 'normal'
    else:
        scale_interval['state'] = 'disable'
        scale_continue['state'] = 'disable'
        button_save['state'] = 'disable'
        button_continue['state'] = 'disable'
def Exit():
    print("Exit")
    bRuning = False  
    cap.release()
    cv2.waitKey(200)
    root.destroy()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
if __name__ == '__main__':
    root = Tk()
    root.title("Capture tool")
    root.bind("<Key>",KeyPress)
    root.protocol("WM_DELETE_WINDOW", Exit)
    
    scale_interval = Scale(root, label='save 1 image/sec', from_=1, to=30, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=2, command=scale_interval_scroll)
    scale_interval.grid(row=0,columnspan=2)
    scale_interval.set(3)
    scale_continue = Scale(root, label='continue save 1 sec', from_=1, to=60, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=4, command=scale_continue_scroll)
    scale_continue.grid(row=1,columnspan=2)
    scale_continue.set(3)
    picturebox = Label(root,text = '', compound='center', font=("Times", 150),fg="white") 
    picturebox.grid(row=2,columnspan=2,padx=20)
    button_save = Button(root,text = 'save image',width=30, height=5,command = button_save_click)
    button_save.grid(row=3,column=0,padx=20,pady=5,sticky='w')
    button_continue = Button(root,text = 'continue save',width=30, height=5,command = button_continue_click)
    button_continue.grid(row=3,column=1,padx=20,pady=5,sticky='e')
    label_message = Label(root,text = '') 
    label_message.grid(row=4,columnspan=2,sticky='e')



    cap = cv2.VideoCapture(0) 
    z,image_sec,continue_time=1,1,1
    while os.path.exists(f'{z:04}.jpg'): z+=1
    run_p,run_all=1,1
    start_time,last_time = time.time(),time.time()
    bRuning,bGet = True,False
    msg = np.zeros(640*480*3).reshape(480,640,3).astype(np.uint8)
    cv2.putText(msg , "Camera error" , (80,260), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
    msg = ImageTk.PhotoImage(image=Image.fromarray(msg))

    threading.Thread(target=read_camera,daemon=True).start()
    root.mainloop()
