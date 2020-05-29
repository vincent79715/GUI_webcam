import os
import cv2
import time
import _thread
import numpy as np
from tkinter import *
from PIL import Image,ImageTk

def read_camera():
    global cap, ret, frame
    while True:
        ret, frame = cap.read()
        if ret :
            img = ImageTk.PhotoImage(image=Image.fromarray(frame[:,:,::-1]))
        else:
            img = msg
            cap = cv2.VideoCapture(0)
        picturebox.config(image=img)
        picturebox.image=img
def continue_save(t1=-1,t2=-1):
    global last_time,z
    set_state(False)
    if t1==-1 : t1=interval_time
    if t2==-1 : t2=continue_time
    while time.time()-start_time < 60:
        cv2.waitKey(1)
        if time.time()-last_time > t1:
            last_time=time.time()
            while os.path.exists(f'{z:04}.jpg'): z+=1
            cv2.imwrite(f'{z:04}.jpg',frame)
            label_message.config(text=f'{z:04}.jpg')
        if time.time()-start_time > t2:
            break
    set_state(True)
def KeyPress(event=None):
    key = event.keysym
    if key=='q' or key=='Escape': quit()
    elif key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
def button_save_click():
    global start_time
    start_time = time.time()
    if ret: _thread.start_new_thread(continue_save,(0,0))
def button_continue_click():
    global start_time
    start_time = time.time()
    if ret:  _thread.start_new_thread(continue_save,())
def scale_interval_scroll(v):
    global interval_time
    interval_time = 1/int(v)
    scale_interval.config(label=f'save {v} image/sec')
def scale_continue_scroll(v):
    global continue_time
    continue_time = int(v)
    scale_continue.config(label=f'continue save {v} sec')
def quit():
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
picturebox = Label(root) 
picturebox.grid(row=2,columnspan=2,padx=20)
button_save = Button(root,text = 'save image',width=30, height=5,command = button_save_click)
button_save.grid(row=3,column=0,padx=20,pady=5,sticky='w')
button_continue = Button(root,text = 'continue save',width=30, height=5,command = button_continue_click)
button_continue.grid(row=3,column=1,padx=20,pady=5,sticky='e')
label_message = Label(root,text = '') 
label_message.grid(row=4,columnspan=2,sticky='e')

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
z,continue_time,interval_time=1,1,1
start_time,last_time = time.time(),time.time()

msg = np.zeros(640*480*3).reshape(480,640,3).astype(np.uint8)
cv2.putText(msg , "Camera error" , (80,260), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)
msg = ImageTk.PhotoImage(image=Image.fromarray(msg))

_thread.start_new_thread(read_camera,())

root.mainloop()
