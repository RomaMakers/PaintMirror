#A Simple mjpg stream http server for the Raspberry Pi Camera
#  inspired by https://gist.github.com/n3wtron/4624820
#edited by Norbert (mjpeg part) from a file from Copyright Jon Berg , turtlemeat.com,
#MJPEG Server for the webcam
#
# Face Detection and display on http://localhost:8080/image.mjpg
#
# by Dariomas @ FabLab Roma Makers - 27/09/2015
#

import io,string,cgi,time
from os import curdir, sep
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2 as cv
import numpy as np
import re

camera=None
cameraQuality=75
cascade=None
    
def scale_rect(x, y, w, h, scal):
        tw = w * scal
        th = h * scal
        nw = int(round(tw))
        nh = int(round(th))
        fx = x -( (nw - w) / 2)
        fy = y - ((nh - h) / 2)
        nx = int(round(fx))
        ny = int(round(fy))
        if nx < 0 :
            nx = 0
        if ny < 0 :
            ny = 0
        if (nx + nw) > 639 :
            nw = 639 - nx
        if (ny + nh) > 479 :
            nw = 479 - nx
        return nx, ny ,nw, nh
    
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global cameraQuality
        try:
            self.path=re.sub('[^.a-zA-Z0-9]', "",str(self.path))
            if self.path=="" or self.path==None or self.path[:1]==".":
                return
            if self.path.endswith(".html"):
                f = open(curdir + sep + self.path)
                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return
            if self.path.endswith(".mjpeg"):                
                
                self.send_response(200)
                self.wfile.write("Content-Type: multipart/x-mixed-replace; boundary=--aaboundary")
                self.wfile.write("\r\n\r\n")
                self.end_headers()
                #stream=io.BytesIO()
                global camera
                rawCapture = PiRGBArray(camera, size=(640, 480))
                canvas = np.zeros((608, 1600, 3), dtype = "uint8")
                # capture frames from the camera
                try:
                    global cascade
                    nface = 0
                    for frame in camera.capture_continuous(rawCapture, format="bgr"): #, use_video_port=True):
                        # Reset the stream for the next capture
                        rawCapture.seek(0)
                        ##start=time.time()
                        # grab the raw NumPy array representing the image, then initialize the timestamp
                        # and occupied/unoccupied text
                        image = frame.array
                        ##print("Time to capture bgr+numpy = %.4f" % (time.time()-start))
                        #Convert to grayscale
                        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
                        gray = cv.equalizeHist(gray)
                        #Look for faces in the image using the loaded cascade file
                        rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(20, 20), flags = cv.CASCADE_SCALE_IMAGE)
                        if len(rects) == 0:
                            # clear the stream in preparation for the next frame
                            #rawCapture.truncate()
                            continue
                            rects = []
                        #else:
                        #      rects[:,2:] += rects[:,:2]
                        ##print("Time to Cascade = %.4f" % (time.time()-start))
                        print "Found "+str(len(rects))+" face(s)"
                        #Draw a rectangle around every found face
                        for (x,y,w,h) in rects:
                            #print "Found rect "+str((x,y,w,h))
                            (nx,ny,nw,nh) = scale_rect(x, y, w, h, 1.5)
                            #cv.rectangle(canvas,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
                            roi  = np.copy(image[ny:ny+nh,nx:nx+nw])
                            print "ROI size "+str(roi.shape)
                            # we need to keep in mind aspect ratio so the image does
                            # not look skewed or distorted -- therefore, we calculate
                            # the ratio of the new image to the old image
                            larg = 320
                            r = round (larg / roi.shape[1])
                            nr = roi.shape[0] * int(r)
                            dim = (larg, nr)
                            # perform the actual resizing of the image and show it
                            resized = cv.resize(roi, dim, interpolation = cv.INTER_AREA)
                            print "Resize "+str(resized.shape)
                            facepos = (nface * larg)
                            canvas[0:resized.shape[0], facepos:facepos + resized.shape[1]] = resized
                            nface += 1
                            if nface == 5:
                                nface = 0
                             
                        ret, cv2mat=cv.imencode(".jpeg",canvas) #,(cv.IMWRITE_JPEG_QUALITY,cameraQuality))
                        JpegData=bytearray(cv2mat)
                        ##print("Time to encode JPG = %.4f" % (time.time()-start))
                        self.wfile.write("--aaboundary\r\n")
                        self.send_header('Content-type','image/jpeg')
                        self.send_header('Content-length',len(JpegData))
                        self.end_headers()
                        self.wfile.write(JpegData)
                        self.wfile.write("\r\n\r\n\r\n")
                        # Reset the stream for the next capture
                        rawCapture.seek(0)
                        # clear the stream in preparation for the next frame
                        rawCapture.truncate()
                        #time.sleep(.5)
                except KeyboardInterrupt:
                    pass 
                return
            if self.path.endswith(".jpeg"):
                f = open(curdir + sep + self.path)
                self.send_response(200)
                self.send_header('Content-type','image/jpeg')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return
            return
        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)
    def do_POST(self):
        global rootnode, cameraQuality
        try:
            ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
            if ctype == 'multipart/form-data':
                query=cgi.parse_multipart(self.rfile, pdict)
            self.send_response(301)

            self.end_headers()
            upfilecontent = query.get('upfile')
            print "filecontent", upfilecontent[0]
            value=int(upfilecontent[0])
            cameraQuality=max(2, min(99, value))
            self.wfile.write("<HTML>POST OK. Camera Set to<BR><BR>");
            self.wfile.write(str(cameraQuality));

        except :
            pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
#class ThreadedHTTPServer(HTTPServer):
    """Handle requests in a separate thread."""

def main():
    global camera
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 10
    global cascade
    cascade_fn = "/opt/HaarCascades/face.xml"
    nested_fn  = "/opt/HaarCascades/eye.xml"
    cascade = cv.CascadeClassifier(cascade_fn)
    #nested = cv.CascadeClassifier(nested_fn)

#camera.start_preview()

    try:
        #server = ThreadedHTTPServer(('', 8080), MyHandler)
        server = HTTPServer(('', 8080), MyHandler)
        print 'started httpserver...'
        server.serve_forever()
    except KeyboardInterrupt:
        print '^C received, shutting down server'
        camera.close()
        server.socket.close()

if __name__ == '__main__':
    main()