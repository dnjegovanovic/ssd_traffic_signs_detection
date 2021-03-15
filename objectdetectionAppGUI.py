import os
import tensorflow as tf
import numpy as np
import yaml
import cv2
import PIL.Image, PIL.ImageTk
import tkinter
import tkinter as tk

from tkinter import messagebox
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from utils.defaultboxes import generate_default_boxes
from utils.boxutils import decode, computeNms
from network.ssdnet import createSSD

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CLASSES = 247
BATCH_SIZE = 1


class AppSSDObjectDetection:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.rowconfigure([0, 1, 2, 3, 4, 5, 6, 7], minsize=10, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=10, weight=1)
        self.window.title(window_title)
        self.video_source = video_source  # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        self.scores = 0.0
        self.classes = ''
        self.detectVideo = 0

        self.useImg = IntVar()
        self.useVideoCap = IntVar()

        self.useImgVideoCapture = IntVar()
        self.useImgVideoCapture.set(1)

        self.useImgRbtn = Radiobutton(window, text="Use image", variable=self.useImgVideoCapture, value=2,
                                      command=lambda: self.refresh())
        self.useImgRbtn.grid(row=0, column=0, sticky=W)
        self.useVCbtn = Radiobutton(window, text="Use video capture", variable=self.useImgVideoCapture, value=1,
                                    command=lambda: self.refresh())
        self.useVCbtn.grid(row=0, column=4, sticky=W)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=1, column=1, columnspan=3)

        label = tk.Label(text="Model path")
        label.grid(row=2, column=2)
        self.modelPathEntry = tk.Entry(width=100)
        self.modelPathEntry.grid(row=3, column=1, columnspan=3)

        btnInitModle = tk.Button(master=window, text="Load Model", width=10,
                                 command=lambda: self.initNetwork(self.modelPathEntry.get()))
        btnInitModle.grid(row=4, column=2, sticky="nsew")

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Image detect", width=10, command=self.detectOnImage)
        self.btn_snapshot.grid(row=5, column=0, sticky="nsew")

        self.loadNewImage = tkinter.Button(window, text="Load new image", width=15, command=self.reloadImage)
        self.loadNewImage.grid(row=6, column=0, sticky="nsew")

        self.btnDetectV = tkinter.Button(window, text="Video Detect", width=10, command = self.actDetectVideo)
        self.btnDetectV.grid(row=5, column=4, sticky="nsew")

        self.listBox = tkinter.Listbox(window, height=2,
                                       width=50,
                                       bg="grey",
                                       activestyle='dotbox',
                                       font="Helvetica",
                                       fg="yellow")
        self.listBox.grid(row=6, column=1, columnspan=3)
        self.listBox.insert(0, "Score:")
        self.listBox.insert(1, "Class:")



        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        print("Image state={}".format(self.useImg.get()))
        print("Video state={}".format(self.useVideoCap.get()))

        if self.useImgVideoCapture.get() == 1:
            self.update()
        self.window.mainloop()

    def actDetectVideo(self):
        if self.useImgVideoCapture.get() == 1:
            if self.detectVideo == 0:
                self.detectVideo = 1

    def upDataeBox(self):
        s = 'Score:'
        sArr = s.split()
        for v in self.scores:
            sArr.append(str(v))
            sArr.append(',')
        ss = ''.join(sArr)

        c = 'Class:'
        cArr = c.split()
        for v in self.classes:
            cArr.append(str(v))
            cArr.append(',')
        cc = ''.join(cArr)

        self.listBox.delete(0)
        self.listBox.delete(1)

        self.listBox.insert(0, ss)
        self.listBox.insert(1, cc)

    def refresh(self):

        if self.useImgVideoCapture.get() == 1:
            print("Start Video ...")
            self.canvas.delete("all")
            self.update()
        elif self.useImgVideoCapture.get() == 2:
            print("Load image ...")
            self.vid.stopStream()
            self.canvas.delete("all")
            self.open_img()

    def reloadImage(self):
        if self.useImgVideoCapture.get() == 2:
            print("Load image ...")
            self.vid.stopStream()
            self.canvas.delete("all")
            self.open_img()
            self.scores = [0.0]
            self.classes = ['']
            self.upDataeBox()
        else:
            messagebox.showwarning("Warning", "Chose image mode?")


    def update(self):
        # Get a frame from the video source
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.vid.get_frame() == None and self.useImgVideoCapture.get() == 1:
            self.vid = MyVideoCapture(self.video_source)

            ret, frame = self.vid.get_frame()

            if ret:
                if self.detectVideo == 1 and hasattr(self, 'ssd'):
                    img_resOrg = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
                    img_res = np.array(img_resOrg, dtype=np.float32)
                    img_res = img_res.reshape(-1, 300, 300, 3)
                    img_res = (img_res / 127.0) - 1
                    boxes, self.classes, self.scores = self.predict(img_res)
                    boxes *= (frame.shape[1], frame.shape[0]) * 2

                    for i, box in enumerate(boxes):
                        top_left = (box[0], box[1])
                        bot_right = (box[2], box[3])
                        cv2.rectangle(frame, top_left, bot_right, (0, 0, 255), 2)
                        sc = round(float(self.scores[i]), 2)
                        cv2.putText(frame, "sc:{}-cls:{}".format(sc, self.classes[i]),
                                    (int(top_left[0]), int(top_left[1] - 10)), font,
                                    0.5, (255, 0, 0), 2, cv2.LINE_AA)

                    # b, g, r = cv2.split(frame)
                    # img = cv2.merge((r, g, b))
                    frame = Image.fromarray(frame)
                    self.photo = ImageTk.PhotoImage(frame)

                    #self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                    self.upDataeBox()
                else:
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            self.window.after(self.delay, self.update)
        elif self.useImgVideoCapture.get() == 1:

            ret, frame = self.vid.get_frame()

            if ret:
                if self.detectVideo == 1 and hasattr(self, 'ssd'):
                    img_resOrg = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
                    img_res = np.array(img_resOrg, dtype=np.float32)
                    img_res = img_res.reshape(-1, 300, 300, 3)
                    img_res = (img_res / 127.0) - 1
                    boxes, self.classes, self.scores = self.predict(img_res)
                    boxes *= (frame.shape[1], frame.shape[0]) * 2

                    for i, box in enumerate(boxes):
                        top_left = (box[0], box[1])
                        bot_right = (box[2], box[3])
                        cv2.rectangle(frame, top_left, bot_right, (0, 0, 255), 2)
                        sc = round(float(self.scores[i]), 2)
                        cv2.putText(frame, "sc:{}-cls:{}".format(sc, self.classes[i]),
                                    (int(top_left[0]), int(top_left[1] - 10)), font,
                                    0.5, (255, 0, 0), 2, cv2.LINE_AA)

                    # b, g, r = cv2.split(frame)
                    # img = cv2.merge((r, g, b))
                    frame = Image.fromarray(frame)
                    self.photo = ImageTk.PhotoImage(frame)

                    # self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                    self.upDataeBox()
                else:
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            self.window.after(self.delay, self.update)

    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename

    def detectOnImage(self):
        print(hasattr(self, 'ssd'))
        if not hasattr(self, 'ssd') or self.useImgVideoCapture.get() != 2:
            messagebox.showwarning("Warning", "SSD Model is not loaded or Image detection not in use?")
        else:
            if self.detectVideo == 1:
                self.detectVideo = 0
            font = cv2.FONT_HERSHEY_SIMPLEX

            img_resOrg = cv2.resize(self.ssdImage, (300, 300), interpolation=cv2.INTER_AREA)
            img_res = np.array(img_resOrg, dtype=np.float32)
            img_res = img_res.reshape(-1, 300, 300, 3)
            img_res = (img_res / 127.0) - 1
            boxes, self.classes, self.scores = self.predict(img_res)
            boxes *= (self.ssdImage.shape[1], self.ssdImage.shape[0]) * 2

            for i, box in enumerate(boxes):
                top_left = (box[0], box[1])
                bot_right = (box[2], box[3])
                cv2.rectangle(self.ssdImage, top_left, bot_right, (0, 0, 255), 2)
                sc = round(float(self.scores[i]), 2)
                cv2.putText(self.ssdImage, "sc:{}-cls:{}".format(sc, self.classes[i]), (int(top_left[0]), int(top_left[1] - 10)), font,
                            0.5, (255, 0, 0), 2, cv2.LINE_AA)

            b, g, r = cv2.split(self.ssdImage)
            img = cv2.merge((r, g, b))
            self.image = Image.fromarray(img)
            self.image = ImageTk.PhotoImage(self.image)

            self.canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)
            self.upDataeBox()

    def open_img(self):

        x = self.openfn()
        img = cv2.imread(x)
        img = cv2.resize(img, (int(self.vid.width), int(self.vid.height)), interpolation=cv2.INTER_AREA)
        self.ssdImage = img.copy()

        b, g, r = cv2.split(img)
        img = cv2.merge((r, g, b))
        self.image = Image.fromarray(img)
        self.image = ImageTk.PhotoImage(self.image)

        self.canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)

    def initNetwork(self, checkpointPath):
        if not os.path.exists(checkpointPath):
            messagebox.showwarning("Path warning","Model file do not exist on this path?")
        else:
            print(checkpointPath)
            print("Config path: {}".format(r'/config.yml'))
            with open(r'./config.yml') as f:

                cfg = yaml.load(f)

            try:
                config = cfg['SSD300']
            except AttributeError:
                raise ValueError('Unknown architecture: {}'.format('ssd300'))

            self.default_boxes = generate_default_boxes(config)
            print("Default boxes:{}".format(len(self.default_boxes)))
            try:
                self.ssd, latestEpoch = createSSD(NUM_CLASSES, 'ssd300',
                                                  'specified',
                                                  '',
                                                  checkpointPath)

                print('Latest epoch: {}'.format(latestEpoch))
            except Exception as e:
                print(e)
                print('The program is exiting...')
                sys.exit()

    def predict(self, imgs):
        confs, locs = self.ssd(imgs)

        confs = tf.squeeze(confs, 0)
        locs = tf.squeeze(locs, 0)

        confs = tf.math.softmax(confs, axis=-1)
        classes = tf.math.argmax(confs, axis=-1)
        scores = tf.math.reduce_max(confs, axis=-1)

        boxes = decode(self.default_boxes, locs)

        out_boxes = []
        out_labels = []
        out_scores = []

        for c in range(1, NUM_CLASSES):
            cls_scores = confs[:, c]

            score_idx = cls_scores > 0.6
            # cls_boxes = tf.boolean_mask(boxes, score_idx)
            # cls_scores = tf.boolean_mask(cls_scores, score_idx)
            cls_boxes = boxes[score_idx]
            cls_scores = cls_scores[score_idx]

            nms_idx = computeNms(cls_boxes, cls_scores, 0.45, 200)
            cls_boxes = tf.gather(cls_boxes, nms_idx)
            cls_scores = tf.gather(cls_scores, nms_idx)
            cls_labels = [c] * cls_boxes.shape[0]

            out_boxes.append(cls_boxes)
            out_labels.extend(cls_labels)
            out_scores.append(cls_scores)

        out_boxes = tf.concat(out_boxes, axis=0)
        out_scores = tf.concat(out_scores, axis=0)

        boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
        classes = np.array(out_labels)
        scores = out_scores.numpy()

        return boxes, classes, scores


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None)

    def stopStream(self):
        if self.vid.isOpened():
            self.vid.release()

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == '__main__':
    # Create a window and pass it to the Application object
    AppSSDObjectDetection(tkinter.Tk(), "Object Detection SSD")
