import os
import cv2
import time
import _thread
import numpy as np
import multiprocessing as mp
from tkinter import *
from PIL import Image,ImageTk

def read_camera():
    global cap,run_p,last_time,bGet,z
    showtext=''
    while bRuning:
        t0 = time.time()
        ret, frame = cap.read()
        t1 = time.time()
        # show error or image
        if ret :
            if bGet and run_all>1 and run_p>1:
                T = int((1-run_p/run_all)*100)
                showtext = f'{T}'
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            img = msg
            cap = cv2.VideoCapture(0)
        t2 = time.time()
        # save image
        if ret and bGet and run_all>run_p and time.time()-last_time > interval_time:
            last_time += interval_time     
            _thread.start_new_thread(Save_image,(f'{z:04}.jpg',frame.copy()))			
            label_message.config(text=f'{z:04}.jpg')
            run_p,z = run_p+1,z+1
        if bGet and run_p>=run_all:
            bGet = False
            set_state(True)
            showtext = ''
        t3 = time.time()
        picturebox.config(image=img,text=showtext)
        picturebox.image=img
        t4 = time.time()
        print(f'{(t4-t0)*1000:5.1f} , {(t1-t0)*1000:5.1f} , {(t2-t1)*1000:5.1f} , {(t3-t2)*1000:5.1f} , {(t4-t2)*1000:5.1f}')
def Save_image(name,img):
    cv2.imwrite(name,img)
def KeyPress(event=None):
    key = event.keysym
    if key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
    elif key=='q' or key=='Escape': quit()
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
def quit():
    bRuning = False
    cv2.waitKey(100)
    cap.release()
    cv2.waitKey(100)
    root.destroy()
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
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
root = Tk()
root.title("Capture tool")
root.bind("<Key>",KeyPress)

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

_thread.start_new_thread(read_camera,())

root.mainloop()
