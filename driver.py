import argparse
import logging as lg
import configuration_file as config
import imutils
from scipy.spatial import distance as dist
from identify_people import *

try:
    file_format1 = "%(asctime)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s"
    lg.basicConfig(filename='social_distance.log', filemode='w', level=lg.DEBUG, format=file_format1)
    #console_log = lg.StreamHandler()
    #console_log.setLevel(lg.DEBUG)
    #console_format = lg.Formatter('%(asctime)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s')
    #console_log.setFormatter(console_format)
    #lg.getLogger('').addHandler(console_log)
    lg.info('!!!Logging Started!!!')
except Exception as E:
    lg.error("Something went wrong while creating logg file")
    lg.exception(E)

lg.info("Inside Driver code")

def argument_parsing():
    '''construct the argument parse and parse the arguments'''
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
        ap.add_argument("-o", "--output", type=str, default="",	help="path to (optional) output video file")
        ap.add_argument("-d", "--display", type=int, default=1,	help="whether or not output frame should be displayed")

        args = vars(ap.parse_args(["--input","input_files/walk2.mp4","--output","my_output.avi","--display","1"]))
        #args = vars(ap.parse_args())
        lg.info("Arguments are: {}".format(str(args)))
        return args
    except Exception as E:
        lg.error('something went wrog')
        lg.exception(str(E))
args = argument_parsing()

def model():
    try:
        labelsPath = "yolo-coco/coco.names"
        lg.info(str("LabelsPath = {}".format( labelsPath)))
        LABELS = open(labelsPath).read().strip().split("\n")
        lg.info(str("LABELS = {}".format(LABELS)))
        lg.info("Loaded the COCO class labels on which YOLO model was trained on")
        # "derive the paths to the YOLO weights and model configuration"
        weightsPath = "yolo-coco/yolov3.weights"
        configPath = "yolo-coco/yolov3.cfg"
        lg.info(str("weightsPath==>{}".format(weightsPath)))
        lg.info(str("configPath==>{}".format( configPath)))
        # "load our YOLO object detector trained on COCO dataset (80 classes)"
        lg.info("Loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # "determine only the *output* layer names that we need from YOLO"
        layer_n = net.getLayerNames()
        layer_n = [layer_n[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, layer_n,LABELS
    except Exception as E:
        lg.error('something went wrog')
        lg.exception(str(E))
net,ln,LABELS = model()

def access_frame(args):
    '''Accessing video stream and counting frames'''
    try:
        cap = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None
        lg.info("initialize the video stream and pointer to output video file")
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        lg.info("Total frames in video==>{}".format(total))
        return cap, writer, total
    except Exception as E:
        lg.error('something went wrog')
        lg.exception(str(E))

lg.info('calling function :  access_frame(args)')
vs, writer, total = access_frame(args)

def violate_f(results):
    '''after computing euclidean distance find violation set based on D < MIN_DISTANCE'''
    try:
        # "initialize the set of indexes that violate the minimum social distance"
        violate_set = set()
        # "ensure there are *at least* 2 people detections required in order to compute 2 pairwise distance maps"
        if len(results) >= 2:
            # "extract all centroids from results, compute Euclidean distances between all pairs of the centroids"
            centroids = np.array([r[2] for r in results])
            D_mat = dist.cdist(centroids, centroids, metric="euclidean")
            # "loop over the upper triangular of the distance matrix"
            for i in range(0, D_mat.shape[0]):
                for j in range(i + 1, D_mat.shape[1]):
                    # "check to see if  distance between any 2 centroid pairs is < the configured number of pixels"
                    if D_mat[i, j] < config.MIN_DISTANCE:
                        # "update our violation set with the indexes of the centroid pairs"
                        violate_set.add(i)
                        violate_set.add(j)
        return violate_set
    except Exception as E:
        lg.error("Went Wrong")
        lg.exception(E)

def colour(results, violate_set, frame):
    '''colour bounding box and coordinates of person'''
    try:
        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # "extract the bounding box and centroid coordinates, then initialize the color of the annotation"
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # "if the index pair exists within the violation set, then update the color"
            if i in violate_set:
                color = (0, 0, 255)

            # "draw (1) a bounding box around the person and (2) the  centroid coordinates of the person"
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
    except Exception as E:
        lg.error('Went Wrong')
        lg.exception(E)

def frame_display(frame, args):
    '''check to see if the output frame should be displayed to screen'''
    try:
        if args["display"] > 0:
            # "show the output frame"
            cv2.imshow("Frame", frame)
            interrupt_key = cv2.waitKey(1) & 0xFF
            return interrupt_key
            # "if the `q` key was pressed, break from the loop"
    except Exception as E:
        lg.error('Went Wrong')
        lg.exception(E)


def write_to_file(args, writer_obj, frame):
    ''' It writes output to the specified file in the argument'''
    try:
        # "if an output video file path has been supplied and the video writer has not been initialized, do so now"
        if args["output"] != "" and writer_obj is None:
            # "initialize our video writer"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer_obj = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

        # "if the video writer is not None, write the frame to the output video file"
        if writer_obj is not None:
            writer_obj.write(frame)
            writer_obj.release()
    except Exception as E:
        lg.error('Went wrong')
        lg.exception(E)

def distance_checker(vs,net,ln,LABELS,args,total,writer):
    '''check healthy distance among pair of people'''
    try:
        lg.info("Displaying social distance detection result..!!!")
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:  #"reached the end of the stream"
                break
            frame = imutils.resize(frame, width = 700)
            results = identify_person(frame, net, ln, lg, personIdx=LABELS.index("person"))
            violate = violate_f(results)
            colour(results, violate, frame)

            # "draw the total number of social distancing violations on the output frame"
            text = "# of Violations:{}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

            key = frame_display(frame, args)
            if key == ord("q"):
                break
            write_to_file(args, writer, frame)
    except Exception as E:
        lg.error('something went wrog')
        lg.exception(str(E))
    finally:
        lg.shutdown()

lg.info('calling function : distance_checker()')
distance_checker(vs,net, ln, LABELS,args, total, writer)
