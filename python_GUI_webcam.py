import os
import cv2
import time
from tkinter import *
from PIL import Image,ImageTk

z=0
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
start_time,last_time = time.time(),time.time()
continue_time,interval_time=1,1

def read_camera():
    ret, frame = cap.read()
    img = ImageTk.PhotoImage(image=Image.fromarray(frame[:,:,::-1]))
    label1.config(image=img)
    label1.image=img
    root.after(1, read_camera)

def continue_save(t1=-1,t2=-1):
    global last_time,z
    if t1==-1 : t1=interval_time
    if t2==-1 : t2=continue_time
    if time.time()-last_time > t1:
        last_time=time.time()
        while os.path.exists(str(z).zfill(4)+".jpg"): z+=1
        cv2.imwrite(str(z).zfill(4)+".jpg",frame)
        label2.config(text='save '+str(z).zfill(4)+".jpg")
    if time.time()-start_time < t2:
        root.after(1, continue_save)

def keypress(event=None):
    key = event.keysym
    print(f"{event.keysym}:{event.char}:{event.keycode}")
    if key=='Escape': quit()
    elif key=='s' or key=='space': button1_fn()
    elif key=='c': button2_fn()
    
def button1_fn():
    continue_save(0,0)
def button2_fn():
    global start_time
    start_time = time.time()
    continue_save()

def Scale1_fn(v):
    global interval_time
    interval_time = 1/int(v)
    Scale1.config(label='save '+v+' image/sec')
def Scale2_fn(v):
    global continue_time
    continue_time = int(v)
    Scale2.config(label='continue save '+v+' sec')

def quit():
    cap.release()
    root.destroy()


root = Tk()
root.title("Capture tool")
root.geometry("750x730+300+100")
root.bind("<Key>",keypress)

Scale1 = Scale(root, label='save 1 image/sec', from_=1, to=30, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=2, resolution=1, command=Scale1_fn)
Scale1.pack()

Scale2 = Scale(root, label='continue save 1 sec', from_=1, to=60, orient=HORIZONTAL,length=480, showvalue=0, tickinterval=4, resolution=1, command=Scale2_fn)
Scale2.pack()

Scale1.set(3)
Scale2.set(3)

label1 = Label(root) 
label1.pack()

Frame1 = Frame(root)
Frame1.pack()

button1 = Button(Frame1,text = 'save image',width=30, height=5,command = button1_fn)
button1.pack(side='left', padx=50)
button2 = Button(Frame1,text = 'continue save',width=30, height=5,command = button2_fn)
button2.pack(side='right', padx=50)


label2 = Label(root,text = '') 
label2.pack(side='right')

read_camera()
root.mainloop()
