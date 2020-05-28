import os
import cv2
import time
from tkinter import *
from PIL import Image,ImageTk

def read_camera():
    global frame
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
root.geometry("750x730+300+100")
root.bind("<Key>",KeyPress)

scale_interval = Scale(root, label='save 1 image/sec', from_=1, to=30, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=2, resolution=1, command=scale_interval_scroll)
scale_interval.pack()
scale_interval.set(3)
scale_continue = Scale(root, label='continue save 1 sec', from_=1, to=60, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=4, resolution=1, command=scale_continue_scroll)
scale_continue.pack()
scale_continue.set(3)
picturebox = Label(root) 
picturebox.pack()
Frame1 = Frame(root)
Frame1.pack()
button_save = Button(Frame1,text = 'save image',width=30, height=5,command = button_save_click)
button_save.pack(side='left', padx=50)
button_continue = Button(Frame1,text = 'continue save',width=30, height=5,command = button_continue_click)
button_continue.pack(side='right', padx=50)
label_message = Label(root,text = '') 
label_message.pack(side='right')

read_camera()
root.mainloop()
