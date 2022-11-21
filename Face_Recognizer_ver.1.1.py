import cv2 as cv
from functools import partial
import tkinter as tk
import numpy as np
import os
import shutil
from tkinter import messagebox
from tkinter import *

# Constants
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
X_TEXT_PADDING = 5
Y_TEXT_PADDING = 5
DATASET_PATH = "C:\\Dataset1"

poiskovik300 = tk.Tk()
poiskovik300.title("Face Recognizer Menu")
poiskovik300.geometry("300x250")
poiskovik300.title("Face Recognizer ver.1.0")
poiskovik300.resizable(False, False)
name = "None"

def buttonCallback():
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('face_all.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv.CascadeClassifier(cascadePath)
    font = cv.FONT_HERSHEY_SIMPLEX
    cam = cv.VideoCapture(0, cv.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    WIN_NAME = "Camera"
    is_closed_flag = False
    is_running = True
    while is_running:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(10, 10))

        for (х, у, w, h) in faces:
            cv.rectangle(img, (х, у), (х + w, у + h), (0, 255, 0), 2)
            Id, confidence = recognizer.predict(gray[у:у + h, х:х + w])
            if Id >= 1 and Id < 31:
                Id = "Maxim"
            elif Id >= 31 and Id < 61:
                Id = "Vadim"
            elif Id >= 61 and Id < 91:
                Id = "Sergey"
            elif Id >= 91 and Id < 121:
                Id = "Matvey"
            elif Id >= 121 and Id < 151:
                Id = "Nikita"
            if (100 - confidence < 30):
                Id = "None"

            cv.putText(img, str(Id), (х + X_TEXT_PADDING, у - Y_TEXT_PADDING),
                       font, 1, WHITE, 2)

        if cv.getWindowProperty(WIN_NAME, cv.WND_PROP_VISIBLE) == 0:
            if is_closed_flag:
                is_running = False
                break
            is_closed_flag = True
        cv.imshow(WIN_NAME, img)
        if (cv.waitKey(100) & 0xff) == 27:
            is_running = False
            break

    cam.release()
    cv.destroyAllWindows()

def NewUser(poiskovik300):
    def getname(poiskovik300):
        global name
        name = edit.get()
        win.destroy()
        poiskovik300.destroy()

    def CamReading():
        WIN_NAME = "Collecting data..."
        cam = cv.VideoCapture(0, cv.CAP_DSHOW)
        face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_id = 'Newuser'
        count = 0
        while True:
            ret, img = cam.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
                count += 1
                cv.imwrite(DATASET_PATH + '\\user.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])
            
            cv.imshow(WIN_NAME, img)
            if (cv.waitKey(100) & 0xff) == 0:
                pass
            if count >= 60:
                break

        cam.release()
        cv.destroyAllWindows()
        return True

    def FileCreator():
        path = DATASET_PATH
        recognizer = cv.face.LBPHFaceRecognizer_create()
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        ids = []
        for imagePath in imagePaths:
            img = cv.imread(imagePath)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces.append(img)
            id = int(os.path.split(imagePath)[-1].split('.')[2])
            ids.append(id)
        try:
            recognizer.train(faces, np.array(ids))
        except cv.error as e:
            return False
        recognizer.write('NewUserData.yml')
        return True
        

    def buttonCallback():
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.read('NewUserData.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv.CascadeClassifier(cascadePath)

        font = cv.FONT_HERSHEY_SIMPLEX

        cam = cv.VideoCapture(0, cv.CAP_DSHOW)
        cam.set(3, 640)
        cam.set(4, 480)
        WIN_NAME = "Camera"
        is_closed_flag = False
        is_running = True
        while is_running:
            ret, img = cam.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))

            for (х, у, w, h) in faces:
                global name
                try:
                    Id, confidence = recognizer.predict(gray[у:у + h, х:х + w])
                except cv.error as e:
                    is_running = False
                    break
                if Id < 61:
                    Id = name
                    rect_color = GREEN
                if (100 - confidence < 30):
                    Id = "None"
                    rect_color = RED
                cv.putText(img, str(Id), (х + X_TEXT_PADDING, у - Y_TEXT_PADDING), font, 1, WHITE, 2)
                cv.rectangle(img, (х, у), (х + w, у + h), rect_color, 2)

            if cv.getWindowProperty(WIN_NAME, cv.WND_PROP_VISIBLE) == 0:
                if is_closed_flag:
                    is_running = False
                    break
                is_closed_flag = True
            cv.imshow(WIN_NAME, img)
            if (cv.waitKey(100) & 0xff) == 27:
                is_running = False
                break

        cam.release()
        cv.destroyAllWindows()



    os.mkdir(DATASET_PATH)

    win = tk.Tk()
    win.geometry('400x200')
    win.configure(background='#4B0082')
    t1 = tk.Label(win, text='Введите ваше имя')
    t1.config(font=('Comic Sans', 10, 'bold'))
    t1.pack()
    
    button = tk.Button(win, text='Подтвердить', command=partial(getname, poiskovik300))
    
    t1.configure(bg='#FFFF00',bd=3)
    button.configure(bg='#FFD700',bd=3)
    edit = tk.Entry(win, width=40)
    edit.place(x=83)
    edit.place(rely=0.4)
    
    t1.place(x=140)
    t1.place(rely=0.2)
    
    button.place(x=160)
    button.place(rely=0.6)
    
    
    win.rowconfigure(4, weight=1)
    win.columnconfigure(0, weight=1)
    win.columnconfigure(1, weight=0)
    win.mainloop()
    CamReading()
    is_valid_train_session = FileCreator()

    if os.path.isfile("NewUserData.yml"):
        shutil.rmtree(DATASET_PATH)
    if is_valid_train_session:
        buttonCallback()


main_menu = tk.Menu()
file_menu = tk.Menu()

main_menu.add_cascade(label="Info", menu=file_menu, background='#4B0082')
poiskovik300.config(menu=main_menu, background='#4B0082')


def check():
    tk.messagebox.showinfo('About', 'Эта программа выполняет функцию распознавания лица на основе нейронной сети, сделанной с помощью библиотеки OpenCV. В ней две функции: распознавание лица разработчиков и распознавание лица людей, которые могут временно внести себя в базу данных')

file_menu.add_command(label='About', command=check, foreground='#FFFFFF')

def clicked():
    tk.messagebox.showinfo('Developers', ' Альшов В.Р (aLvttt)\n Пугачёв Н.Я (Keus)\n Шишков М.А (shisu!)\n Пашкевич М.Э (peppik)\n Моняхин С.Д (Luca Changretta)')

file_menu.add_command(label='Developers', command=clicked, foreground='#FFFFFF')

button1 = tk.Button(poiskovik300, text="Our team recognizer", command=buttonCallback)
label_1 = tk.Label(poiskovik300, text="Face Recognizer")
button2 = tk.Button(poiskovik300, text="Recognizer of you", command=partial(NewUser, poiskovik300))
label_1.grid(row=0, column=0, columnspan=3)
button1.grid(row=1, column=0)
button2.grid(row=1, column=0)
label_1.place(x=30)
label_1.place(rely=0.2)
button2.place(x=100)
button1.place(x=95)
button1.place(rely=0.7)
button2.place(rely=0.5)
poiskovik300.rowconfigure(4, weight=1)
poiskovik300.columnconfigure(0, weight=1)
poiskovik300.columnconfigure(1, weight=0)
main_menu.configure(bg='#4B0082')
file_menu.configure(bg='#4B0082')
poiskovik300.configure(bg='#4B0082')
button1.configure(bg='#FFD700', bd=3)
button2.configure(bg='#FFD700', bd=3)
label_1.configure(bg='#FFFF00', width=30, height=2, font=('Comic Sans', 10, 'bold'), bd=4)
poiskovik300.mainloop()
