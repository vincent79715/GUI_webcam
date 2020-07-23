import cv2
import time
import numpy as np
from os import getcwd
from os.path import isfile,split
from glob import glob
from PIL import Image,ImageTk
from tkinter import Tk,Label,Button
from tkinter.ttk import Combobox
from threading import Thread
from functools import partial
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class mybox:
  def __init__(self,bw=608,bh=608):
    self.n = 0
    self.c = (0,0,255)
    self.bw,self.bh = bw,bh
    self.xmin,self.ymin,self.xmax,self.ymax = 0,0,0,0
    self.x,self.y,self.w,self.h = 0,0,0,0
  def set_file(self,array):
    n,x,y,w,h = array
    self.set_xywh(n,x,y,w,h)
  def set_xywh(self,n,x,y,w,h):
    self.set_n(n)
    self.x,self.y,self.w,self.h = x,y,w,h
    x,y,w,h = x,y,w,h = x*self.bw,y*self.bh,w*self.bw,h*self.bh
    self.xmin,self.ymin,self.xmax,self.ymax = int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)
  def set_minmax(self,xmim,ymin,xmax,ymax):
    self.xmin,self.ymin,self.xmax,self.ymax = xmim,ymin,xmax,ymax
    if self.xmin>self.xmax: self.xmin,self.xmax = self.xmax,self.xmin
    if self.ymin>self.ymax: self.ymin,self.ymax = self.ymax,self.ymin
    self.x,self.y,self.w,self.h = (xmax+xmim)/2/self.bw,(ymax+ymin)/2/self.bh,(xmax-xmim)/self.bw,(ymax-ymin)/self.bh
  def set_min(self,xmim,ymin):
    self.set_minmax(xmim,ymin,self.xmax,self.ymax)
  def set_max(self,xmax,ymax):
    self.set_minmax(self.xmin,self.ymin,xmax,ymax)
    self.xmax,self.ymax = xmax,ymax
  def remove(self):
    self.set_minmax(0,0,0,0)
    self.set_n(0)
  def set_n(self,n):
    self.n = int(n)
    if n==0:self.c=(0,0,255)
    elif n==1:self.c=(0,255,255)
    elif n==2:self.c=(255,255,255)
    elif n==3:self.c=(0,255,0)
    elif n==4:self.c=(255,0,0)
    elif n==5:self.c=(255,255,0)
    elif n==6:self.c=(255,0,255)
    else: self.c=(0,0,0)
  def stryolo(self,num=0):
    if num==1:return f'{self.n} {1-self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}' #cv2.flip(img2, 1)
    elif num==2:return f'{self.n} {1-self.y:.6f} {self.x:.6f} {self.h:.6f} {self.w:.6f}'#cv2.ROTATE_90_CLOCKWISE
    elif num==3:return f'{self.n} {self.y:.6f} {1-self.x:.6f} {self.h:.6f} {self.w:.6f}'#cv2.ROTATE_90_COUNTERCLOCKWISE
    elif num==4:return f'{self.n} {1-self.x:.6f} {1-self.y:.6f} {self.w:.6f} {self.h:.6f}'#cv2.ROTATE_180
    else: return f'{self.n} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}'
  def box(self):
    return self.c,self.xmin,self.ymin,self.xmax,self.ymax
  def bshow(self):
    return self.xmin!=self.xmax and self.ymin!=self.ymax
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bMotion,gx1,gx2,gy1,gy2 = False,0,0,0,0
Nnum,Lnum,selectNum = 0, -1, 0
bRuning = True
ndir = getcwd()

tt=0
if tt==0: path = ndir+"/o/"
elif tt==1: path = "/home/tls/darknet/AOI/yolo/"

allFileList = sorted(glob(path + "*.jpg"))
box = [mybox() for i in range(10)]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def GUIrefresh():
  global Lnum,bMotion
  img0 = None
  while bRuning:
    if Nnum != Lnum:
      Lnum = Nnum
      Name = allFileList[Nnum]
      nnn = split(Name)[1].split(".")[0]
      img0 = cv2.imread(Name)
      for i in range(10):
        box[i].remove()
        label[i]["fg"]="gray"
        boxtype[i]["state"]="disabled"
      label[0]["fg"]="black"
      boxtype[0]["state"]="readonly"
      bMotion = False
      if isfile(path+nnn+".txt"):
        f = open(path+nnn+".txt","r") 
        ss = f.read().split("\n")
        for i in range(len(ss)):
          if len(ss[i])>0:
            A = np.float64(ss[i].split(" "))
            box[i].set_file(A)
            label[i]["fg"]="black"
            boxtype[i].current(int(A[0]))
            boxtype[i]["state"]="readonly"
            selectbox(0)
    elif img0 is None: img0 = msg
    
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    if bMotion:
      box[selectNum].set_minmax(gx1,gy1,gx2,gy2)

    for i in range(10):
      if box[i].bshow():
        c,x1,y1,x2,y2 = box[i].box()
        cv2.rectangle(img1, (x1,y1), (x2,y2), c, 1)
    image = ImageTk.PhotoImage(Image.fromarray(img1))
    picturebox["image"]=image
    picturebox.image=image
    time.sleep(0.03)

def Motion(event):
  global bMotion,gx1,gx2,gy1,gy2
  x0, y0 = event.x-2, event.y-2
  x0 = x0 if x0>0 else 0
  y0 = y0 if y0>0 else 0
  x0 = x0 if x0<608 else 607
  y0 = y0 if y0<608 else 607
  if not bMotion and event.state==272:
    gx1,gy1 = x0,y0
    gx2,gy2 = x0,y0
    bMotion = True
  elif bMotion and event.state==272:
    gx2,gy2 = x0,y0
  elif bMotion and event.state==16:
    bMotion = False
def label_click(i):
  if i==0:
    selectbox(i)
    return

  if label[i]["fg"]=="gray":
    if label[i-1]["fg"]=="black":
      label[i]["fg"]="black"
      boxtype[i]["state"]="readonly"
  else:
    if label[i]["bg"]=="green" and (i==9 or label[i+1]["fg"]=="gray"):
      label[i]["fg"]="gray"
      boxtype[i]["state"]="disabled"
      boxtype[i].current(0)
      box[i].remove()
      selectbox(i-1)

  if label[i]["fg"]=="black":
    selectbox(i)
def boxtype_Select(i,v):
  box[i].set_n(boxtype[i].current())
def selectbox(z):
  global selectNum
  for i in range(10):
    label[i]["bg"]="gray"
  label[z]["bg"]="green"
  selectNum = z
def change_click(v):
  global Nnum
  Nnum = menu.current()
def next_click():
  global Nnum
  Nnum = Nnum+1 if Nnum<len(allFileList)-1 else 0
  menu.current(Nnum)
def pre_click():
  global Nnum
  Nnum = Nnum-1 if Nnum>0 else len(allFileList)-1
  menu.current(Nnum)
def save_click():
  Name = allFileList[Nnum]
  nnn = split(Name)[1].split(".")[0]
  Sout = box[0].stryolo()
  for i in range(1,10):
    if box[i].bshow():
      Sout +="\n"+box[i].stryolo()
  f = open(path+nnn+".txt","w")
  f.write(Sout)
  f.close()
def KeyPress(event=None):
  key = event.keysym
  #print(key)
  if key=='q' or key=='Escape': Exit()
  elif key=='a' or key=='Left': pre_click()
  elif key=='s': save_click()
  elif key=='d' or key=='Right': next_click()
def Exit():
  global bRuning
  bRuning=False
  root.destroy()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == '__main__':
  #Tkinter
  root = Tk()
  root.title("Labelling tool")
  root.bind("<Key>",KeyPress)
  root.protocol("WM_DELETE_WINDOW", Exit)
  picturebox = Label(root,text = '', compound='center', font=("Times", 150),fg="white") 
  picturebox.grid(rowspan=20,column=0)
  picturebox.bind('<Motion>',Motion)


  v = [split(i)[1] for i in allFileList]
  menu = Combobox(root,values=v,width=18, state="readonly",justify="center")
  menu.grid(row=0,column=1,columnspan=2)
  menu.current(0)
  menu.bind("<<ComboboxSelected>>", change_click)


  label = [Button(root,text = f'{i+1}',fg="gray",bg="gray",width=3, command = partial(label_click,i)) for i in range(10)]
  for i in range(10): label[i].grid(row=i+2,column=1)

  boxtype = [Combobox(root,values=["短路","空焊"], state="disabled", width=10) for i in range(10)]
  for i in range(10): 
    boxtype[i].grid(row=i+2,column=2)
    boxtype[i].current(0)
    boxtype[i].bind("<<ComboboxSelected>>", partial(boxtype_Select,i))

  button_next = Button(root,text = '>>',width=20,command = next_click)
  button_next.grid(row=20,column=0,sticky='e')
  button_pre = Button(root,text = '<<',width=20,command = pre_click)
  button_pre.grid(row=20,column=0,sticky='w')
  button_save = Button(root,text = 'save',width=20,command = save_click)
  button_save.grid(row=20,column=0,sticky='n')

  msg = np.zeros(608*608*3).reshape(608,608,3).astype(np.uint8)
  selectbox(0)

  Thread(target=GUIrefresh,daemon=True).start()
  root.mainloop()
