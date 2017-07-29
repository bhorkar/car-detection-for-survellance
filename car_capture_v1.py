
# Using Android IP Webcam video .jpg stream (tested) in Python2 OpenCV3

import urllib
import cv2
print(cv2.__version__)
import numpy as np
import time
import datetime
import os
import errno    

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
#!/usr/bin/env python
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)





# Replace the URL with your own IPwebcam shot.jpg IP:port
url_v  = "http://192.168.1.85:8080/video?.mjpeg"

cascade_src = 'cars.xml'
cv2.ocl.setUseOpenCL(False)
car_cascade = cv2.CascadeClassifier(cascade_src)
fgbg = cv2.createBackgroundSubtractorMOG2()




CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')




NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}




def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
  
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= 0.5)[0]

        add = False; 
        for i in inds:
            add = True
            bbox = dets[i, :4]
            score = dets[i, -1]
            print score, cls
            if cls == "car" or cls == "person":
                pt1 =  (bbox[0], bbox[1])
                pt2 = (bbox[2], bbox[3])
            
                cv2.rectangle(im,pt1, pt2, (0,0,255), 2)
                cv2.putText(im,'{:s} {:.3f}'.format(cls, score),(bbox[0], (int)((bbox[1]- 2))), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0,255,0), 1)
    return im, add;

        




def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

     




#def gen():
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    vid = cv2.VideoCapture()
    vid.open(url_v) 
    cpu_mode = 0;
    prototxt = os.path.join(cfg.MODELS_DIR, NETS['vgg16'][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS['vgg16'][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    index = 0
    while True:
        if 1 :
            success, img = vid.read()
            fgmask = fgbg.apply(img)
           
            cars = car_cascade.detectMultiScale(fgmask, 1.1, 1)
            h,w,d = img.shape;
            minY = int(h*0.2)
            maxY = int(w*0.8)
            check_car = False;
            for (x,y,w,h) in cars:
                if y > minY and y < maxY and w > 20:
                    print x,y,w,h
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    check_car = True;
            if  check_car:
              img1, save = demo(net, img)
              if save:
                 img = img1
                 dictname = datetime.datetime.today().strftime('%Y-%m-%d');
                 mkdir_p(dictname);
                 filename = dictname +  '/count_' + str(index) + '.jpg';
                 print "saving" + filename;
                 cv2.imwrite(filename, img);
                 time.sleep(1) 
                 index = index + 1;
            #cv2.imshow('IPWebcam',img)
	    cv2.imwrite('t.jpg', img)
        #yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


"""Video streaming generator function."""


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


#if __name__ == '__main__':
#    app.run(host='0.0.0.0', debug=True, threaded=True)

