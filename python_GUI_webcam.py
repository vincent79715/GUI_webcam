import os
import cv2
import time
from tkinter import *
from PIL import Image,ImageTk

def read_camera():
    global frame,last_time
    ret, frame = cap.read()
    img = ImageTk.PhotoImage(image=Image.fromarray(frame[:,:,::-1]))
    picturebox.config(image=img)
    picturebox.image=img
    root.after(1, read_camera)
def continue_save(t1=-1,t2=-1):
    global last_time,z
    if t1==-1 : t1=interval_time
    if t2==-1 : t2=continue_time
    if time.time()-last_time > t1:
        last_time=time.time()
        while os.path.exists(str(z).zfill(4)+".jpg"): z+=1
        cv2.imwrite(str(z).zfill(4)+".jpg",frame)
        label_message.config(text='save '+str(z).zfill(4)+".jpg")
    if time.time()-start_time < t2:
        root.after(1, continue_save)
def KeyPress(event=None):
    key = event.keysym
    if key=='q' or key=='Escape': quit()
    elif key=='s' or key=='space': button_save_click()
    elif key=='c': button_continue_click()
def button_save_click():
    continue_save(0,0)
def button_continue_click():
    global start_time
    start_time = time.time()
    continue_save()
def scale_interval_scroll(v):
    global interval_time
    interval_time = 1/int(v)
    scale_interval.config(label='save '+v+' image/sec')
def scale_continue_scroll(v):
    global continue_time
    continue_time = int(v)
    scale_continue.config(label='continue save '+v+' sec')
def set_center(w, h):
    w0,h0 = root.winfo_screenwidth(),root.winfo_screenheight()
    x,y = int((w0-w)/2),int((h0-h)/2)
    root.geometry(f'{w}x{h}+{x}+{y}')
def quit():
    cap.release()
    root.destroy()
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
z,continue_time,interval_time=1,1,1
start_time,last_time = time.time(),time.time()

root = Tk()
root.title("Capture tool")
set_center(685,725)
root.bind("<Key>",KeyPress)


scale_interval = Scale(root, label='save 1 image/sec', from_=1, to=30, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=2, borderwidth=1, command=scale_interval_scroll)
scale_interval.grid(row=0,columnspan=2)
scale_interval.set(3)
scale_continue = Scale(root, label='continue save 1 sec', from_=1, to=60, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=4, borderwidth=2, command=scale_continue_scroll)
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

read_camera()
root.resizable(width=False,height=False)
root.mainloop()
