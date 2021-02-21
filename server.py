from flask import Flask, render_template, Response
import cv2
import pickle
from os.path import expanduser, sep
import sys
import threading
import argparse
import imutils
import os
import imagezmq
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
import docopt 
from sklearn import svm
import dlib
import multiprocessing

outputFrame = None
app = Flask(__name__)

def find_camera(id):
	cameras = readConf()
	if len(cameras) == 0:
		writeConf()
		cameras = readConf()
	return cameras[int(id)]

def get_camera_count():
	cameras = readConf()
	return (len(cameras))

def updateConf():
	cameras = readConfToShell()
	writeConf(cameras)
	print ("Camera conf file updated!")

def testCam(src):
	try:
		tempcap = cv2.VideoCapture(src)
		ret, img = tempcap.read()
		state = ret
		tempcap.release()
	except:
		state = False
	return state

def writeConfFromShell(conftxt, conf='None'):
	if conf == 'None':
		conf = getConfPath()
	cameras = {}
	with open(conftxt) as f:
		for line in f:
			(key, val) = line.split()
			cameras[int(key)] = val
	f.close()
	writeConf(cameras, conf)
	out=("configuration file updated successfully!")
	print (out)

def url_addAuth(user,pw,url):
	chunks = url.split("//")
	chunks[1] = (user + ":" + pw + "@" + chunks[1])
	delimeter = "//"
	string = delimeter.join(chunks)
	print (string)
	return string
	
def record(src="rtsp://192.168.2.10/mpeg4cif", outfile="nv.capture.avi"):
	cap = cv2.VideoCapture(src)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	while True:
		ret, img = cap.read()
		if ret:
			out.write(img)
			cv2.imshow('Live View', img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	out.release()
	cap.release()
	exit()

def readConfToShell(conf='None'):
	if conf == 'None':
		conf = getConfPath()
	try:
		cameras = readConf(conf)
		for cam in cameras:
			print (cameras[cam])
	except:
			print ("")

def writeConf(cameras, conf='None'):
	if conf == 'None':
		conf = getConfPath()
	with open(conf, 'wb') as f:
		pickle.dump(cameras, f)

def getPyVersion():#determines whether use is using python2 or 3. for elimination of relative pathing issues.
	home = expanduser("~")
	version = ((str(sys.version)).split(".")[0])
	subversion = ((str(sys.version)).split(".")[1])
	py = ("python" + version + "." + subversion)
	return (home, py)

def getConfPath():
	#Get python version and build literal string for installed site-packages package location
	home, py = getPyVersion()
	conf = (str(home) + sep + ".local" + sep + "lib" + sep + str(py) + sep + "site-packages/NicVision/cams.conf")
	return conf
	

def readConf(conf='None'):
	if conf == 'None':
		conf = getConfPath()
	try:
		with open(conf, 'rb') as f:
			cameras = pickle.load(f)
	except:
		cameras = {}
		with open(conf, 'wb') as f:
			pickle.dump(cameras, f)
	return cameras
def resize(img):
	width = int(img.shape[1])
	height = int(img.shape[0])
	if width > 640 and height > 480:
		dim = (640, 480)
		img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return img

def readDbFile(datfile="/home/monkey/.local/lib/python3.6/site-packages/NicVision/nv_known_faces.dat"):
	with open(datfile, 'rb') as f:
		all_face_encodings = pickle.load(f)
	return (all_face_encodings)


def recognize(camera_id, img, datfile='/home/monkey/.local/lib/python3.6/site-packages/NicVision/nv_known_faces.dat'):
	if img is not None:
		global known_encodings, known_names
		matches = []
		clf = svm.SVC(gamma ='scale')
		clf.fit(known_encodings, known_names)
		#found_faces = face_recognition.face_locations(img)
		#found_encodings = face_recognition.face_encodings(img)
		#no = len(found_encodings)
		#print ("Camera " + str(camera_id) + " found " + str(no) + " faces!")
		found_faces = face_recognition.face_locations(img)
		#encodings = face_recognition.face_encodings(img, found_faces)
		face_count = -1
		for face in found_faces:
			print(face)
			encoding = face_recognition.face_encodings(img, [face])
			face_count = face_count + 1
			t, r, b, l = face
			face = l, t, r, b # convert to dlib rectangle format (l,t,r,b)
			name = clf.predict(encoding)
			name = str(name[0])
			splitter = '_'
			name = name.split(splitter)[0]
			print ("I see " + name + "...")
			ret = name, face, encoding
			matches.append(ret)
		#print ("Found " + str(len(matches)) + " matches!")
		return (matches, img)
	else:
		return (None, None)

def drawBox(img, boxes, label="Unknown"):
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	pil_image = Image.fromarray(rgb_img)
	# Create a Pillow ImageDraw Draw instance to draw with
	draw = ImageDraw.Draw(pil_image)
	if len(boxes) == 4:
		left, top, right, bottom = boxes
		#print (top, right, bottom, left)
		draw.rectangle(boxes, outline=(0, 0, 255))#draw box on image
	else:
		for box in boxes:#iterate through boxes if more than one provided
			left, top, right, bottom = box
			draw.rectangle(box, outline=(0, 0, 255))#draw box on image#draw box on image
	img = np.array(pil_image)
	del draw
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.putText(img, label, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	return img

def mkIoFiles():
	cameras = readConf()
	for camera_id in cameras.keys():
		name = (str(camera_id) + ".io")
		is_tracking = False
		boxes = []
		out = (is_tracking, boxes)
		with open(name, 'wb') as f:
			pickle.dump(out, f)
		f.close()


def start_tracker(box, label, rgb, inputQueue, outputQueue):
	# construct a dlib rectangle object (left, top, right, bottom)
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)
	while True:
		img = inputQueue.get()
		if img is not None:
			t.update(img)
			pos = t.get_position()
			left = int(pos.left())
			top = int(pos.top())
			right = int(pos.right())
			bottom = int(pos.bottom())
			outputQueue.put((label, (left, top, right, bottom)))



def analyze_stream(camera_id, min_confidence=0.2, datfile='/home/monkey/.local/lib/python3.6/site-packages/NicVision/nv_known_faces.dat'):
	all_face_encodings = readDbFile(datfile)
	global known_names, known_encodings
	known_names = list(all_face_encodings.keys())
	known_encodings = np.array(list(all_face_encodings.values()))
	src = ('http://127.0.0.1:5000/video_feed/' + str(camera_id) + '/')
	wd = os.getcwd()#initialize the working directories
	user = wd.split('/')[2]
	nicdir = ("/home/" + user + "/Nicole")
	nvdir = (nicdir + "/NicVision")
	prototxt = (nvdir + '/MobileNetSSD_deploy.prototxt')
	model = (nvdir + '/MobileNetSSD_deploy.caffemodel')
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	TARGETS = ["bicycle", "bus", "car", "cat", "dog", "horse", "motorbike", "person"]
	net = cv2.dnn.readNetFromCaffe(prototxt, model)
	while True:
		cap = cv2.VideoCapture(src)
		(ret, frame) = cap.read()
		if ret == False:
			is_tracking = False
			break
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		detector = dlib.get_frontal_face_detector()
		faces = detector(frame, 1)
		if len(faces) > 0:
			label = "Unidentified Face"
			is_tracking = True
			matches, detected = recognize(camera_id, rgb)
			boxes = []
			if (len(matches) > 0):
				for match in matches:
					name, box, recognized_img = match
					boxes.append(box)
				out = (is_tracking, name, boxes)
			else:
				out = (is_tracking, label, faces)
			pass
		else:
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(rgb, 0.007843, (w, h), 127.5)
			net.setInput(blob)
			detections = net.forward()
			out = (False, None, [])
			if (detections[0, 0, 1, 2]) == 0:
				is_tracking = False
			for i in np.arange(0, detections.shape[2]):
				idx = int(detections[0, 0, i, 1])
				if idx > 0:
					confidence = float(detections[0, 0, i, 2])
					object_name = CLASSES[idx]
					label = (str(object_name) + " Confidence: " + str(confidence))
					if object_name in TARGETS:
						if confidence >= min_confidence:
							if object_name == "person":
								box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])# compute the (x, y)-coordinates of the bounding box for the object
								(left, top, right, bottom) = box.astype("int")
								box = (left, top, right, bottom)
								out = (True, "Unidentified Person", box)
								print ("Person detected, " + str(out))
							else:
								box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])# compute the (x, y)-coordinates of the bounding box for the object
								(left, top, right, bottom) = box.astype("int")
								box = (left, top, right, bottom)
								out = (True, object_name, box)
								print ("non-person object", str(out))
						else:
							out = (False, "None", [])
							print ("Low confidence filter" + str(out))
		writeIoFile(camera_id, out)

		cap.release()
	cv2.destroyAllWindows()
	cap.release()


def track(camera_id, min_confidence=0.34):
	src = ('http://127.0.0.1:5000/video_feed/' + str(camera_id) + '/')
	wd = os.getcwd()#initialize the working directories
	user = wd.split('/')[2]
	nicdir = ("/home/" + user + "/Nicole")
	nvdir = (nicdir + "/NicVision")
	prototxt = (nvdir + '/MobileNetSSD_deploy.prototxt')
	model = (nvdir + '/MobileNetSSD_deploy.caffemodel')
	inputQueues = []
	outputQueues = []
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	TARGETS = ["bicycle", "bus", "car", "cat", "dog", "horse", "motorbike", "person"]
	net = cv2.dnn.readNetFromCaffe(prototxt, model)
	while True:
		cap = cv2.VideoCapture(src)
		(ret, frame) = cap.read()
		# check to see if we have reached the end of the video file
		if ret == False:
			break
		#frame = imutils.resize(frame, width=600)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		if len(inputQueues) == 0:#if input queues is empty...
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(rgb, 0.007843, (w, h), 127.5)
			net.setInput(blob)
			detections = net.forward()
			for i in np.arange(0, detections.shape[2]):
				confidence = float(detections[0, 0, i, 2])
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]
				label = (str(label) + " (Confidence: " + str(confidence))
				if label != "background" and confidence >= min_confidence:
					is_tracking = True
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])# compute the (x, y)-coordinates of the bounding box for the object
					(left, top, right, bottom) = box.astype("int")
					bb = (left, top, right, bottom)
					out = (is_tracking, label, bb)
					writeIoFile(camera_id, out)
					#create new input and output queue and append to our active tracker list.
					iq = multiprocessing.Queue()
					oq = multiprocessing.Queue()
					inputQueues.append(iq)
					outputQueues.append(oq)
					# spawn a daemon process for a new object tracker
					p = multiprocessing.Process(target=start_tracker, args=(bb, label, frame, iq, oq))
					p.daemon = True
					p.start()
				else:
					pass
		else:
			for iq in inputQueues:
				iq.put(frame)
			for oq in outputQueues:
				(label, box) = oq.get()
				out = (is_tracking, label, box)
				writeIoFile(camera_id, out)
		cap.release()
	# do a bit of cleanup
	cv2.destroyAllWindows()
	cap.release()

def writeIoFile(camera_id, outData):
	name = (str(camera_id) + '.io')
	with open(name, 'wb') as f:
		pickle.dump(outData, f)
		f.close()

def readIoFile(camera_id):
	is_tracking = False
	boxes = []
	name = (str(camera_id) + '.io')
	with open(name, 'rb') as f:
		is_tracking, label, boxes = pickle.load(f)
		f.close()
	return (is_tracking, label, boxes)

def gen_frames(camera_id):
	#get camera source from conf file, and initialize capture object and states
	cam = find_camera(camera_id)
	#TODO: replace this with nv.capture class that uses cv2 and zmq.
	cap =  cv2.VideoCapture(cam)
	iofile = (str(camera_id) + ".io")
	boxes = []
	label = "Unknown"
	camera_state = False
	is_tracking = False
	while True:
		#check io file for tracking data. Put in try/except block to circumvent 'EOF' file error due to incomplete writes during read.
		with open(iofile, 'rb') as f:
			try:
				#try reading tracking data
				is_tracking, label, boxes = pickle.load(f)# tracked_w and tracked_h is the resolution of the image that had detection done on it, used to determine where the box should go on the original feed.
			except:
				#if EOF error, set defaults
				is_tracking = False
				boxes = []
		# # Capture frame-by-frame
		camera_state, frame = cap.read()  # read the camera frame
		#TODO: if not using nv.capture class, set camera state variables like below. if using class, get states from it.
		if camera_state == False:# if cv2.VideoCapture object returns no video stream...
			connected = False#init state variables to False.
			streaming = False
		elif camera_state != True and camera_state != False:#if using the nv.capture class, then camera_state will be a tuple and meet this conditional
			connected, streaming = camera_state #unpack tuple into state variables.
		else:#Catches any other camera situation (assuming connection), and sets the streaming state variable to the capture objects success boolean
			streaming = camera_state
			connected = True
		if not streaming:#if bad stream detected from either capture method:
			break#break from loop
			#TODO: throw exception with handler that removes camera from active list
		frametype = str(type(frame))#get frame type (should be either numpy.ndarray or None)
		if frametype == "<class 'NoneType'>":#if img is empty
			connected = True#set connected var to True as it connected ok
			streaming = False# set streaming var to false as it's not yet streaming video
		if connected == True and streaming == True:# if camera is connected and streaming:
			if is_tracking == True:#if tracking boolean is True (assumes boxes provided with true value from io file):
				#draw boxes on image. Should be a list object of tuples (box.top, box.left, box.bottom, box.right)
				frame = drawBox(frame, boxes, label)#update image with boxes read from io file.
			ret, buffer = cv2.imencode('.jpg', frame)#create memory buffer for web streaming.
			frame = buffer.tobytes()#encode frame
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
				

#flask path to individual camera feeds
@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
   
	"""Video streaming route. Put this in the src attribute of an img tag."""
	return Response(gen_frames(id),
					mimetype='multipart/x-mixed-replace; boundary=frame')# sets mimetype required for streaming

#flask app landing page (index.html)
@app.route('/', methods=["GET"])
def index():
	return render_template('index.html')

#flask app fullscreen viewer html paths for camera_id
@app.route('/cam/<string:id>/')
def cam(id):
	camfile = (str(id) + '.html')
	return render_template(camfile)

def init():
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--tracker",  dest="tracker", type=str, default="mosse", help="OpenCV object tracker type")
	ap.add_argument("-w", "--webport", dest="web_port", type=int, default="9876", help="HTTP port")
	ap.add_argument("-r", "--recvport", dest="imgsrv_port", type=int, default="5555", help="ImageZMQ receiver port. Starts at 5555, increases incrementally by one for each feed.")
	ap.add_argument("-a", "--listen-address", dest="localip", type=str, default="127.0.0.1", help="Network ip address to run server at.")
	args = vars(ap.parse_args())
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}
	tracker_type = args["tracker"]
	tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
	tracked_object = None
	prototxt = 'MobileNetSSD_deploy.prototxt'
	model = 'MobileNetSSD_deploy.caffemodel'
	TARGETS = set(["dog", "person", "car"])
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
	objCount = {obj: 0 for obj in TARGETS}
	net = cv2.dnn.readNetFromCaffe(prototxt, model)
	localip = args["localip"]
	web_port = args["web_port"]
	imgsrv_port = args["imgsrv_port"]
	CONFIDENCE = 0.45
	userdir = os.path.expanduser('~')
	NVDIR = (userdir + os.path.sep + ".local/lib/python3.6/site-packages/NicVision")
	global datfile
	datfile = (NVDIR + os.path.sep + "nv_known_faces.dat")
	print (datfile)
	imageHub = imagezmq.ImageHub()
	trainpath = (NVDIR + "/training_data")
	all_face_encodings = readDbFile(datfile)
	known_names = list(all_face_encodings.keys())
	known_encodings = np.array(list(all_face_encodings.values()))
	mkIoFiles()
	pos = 0

def start_thread(camera_id):
	t = threading.Thread(target=gen_frames, args=(camera_id,))
	t.daemon = True
	t.start()
	app.run()

if __name__ == '__main__':
	init()
	#t = threading.Thread(target=gen_frames, args=(1,))
#	t.daemon = True
#	t.start()
#	app.run()
	t = start_thread(1)
