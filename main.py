# To test SORT
from lib.sort.sort import Sort
# To test DEEP_SORT
from lib.deep_sort import nn_matching
from lib.deep_sort.tracker import Tracker
from lib.deep_sort.detection import Detection
from lib.deep_sort import preprocessing

from lib import config
import cv2
from darkflow.net.build import TFNet
import time
import numpy as np
from scipy.spatial import distance

from skimage.feature import hog

log = config.LOGGER

# Cam rstp access
url_rstp = config.URL_RSTP

# Capture object
cap = cv2.VideoCapture(config.URL_RSTP)

net = TFNet(config.YOLO_PEDESTRIAN_OPTIONS)

# # To test SORT
# tracker = Sort()

# To test DEEP_SORT
max_cosine_distance = 0.2
nms_max_overlap = 1.0
nn_budget = 100
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
hog = cv2.HOGDescriptor()

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


if __name__ == '__main__':
    memory = {}
    # Check if capture object is open
    while(cap.isOpened()):
        time_init = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(
            frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
        if (not ret) and (frame is None):
            # logging.warning('Processing: Frame is empty.')
            cap = cv2.VideoCapture(config.URL_RSTP)
            continue

        labels = []

        dets = []

        # Get predict from network
        results = net.return_predict(frame)
        if len(results) > 0:

            centroid_dets = []
            dets_dict = {}
            # print('Len -> results: {}'.format(len(results)))
            for i in range(len(results)):
                lb = results[i]['label']
                if lb == 'person':
                    tlx = results[i]['topleft']['x']
                    tly = results[i]['topleft']['y']
                    brx = results[i]['bottomright']['x']
                    bry = results[i]['bottomright']['y']
                    conf = results[i]['confidence']
                    x = int((tlx + brx) / 2)
                    y = int((tly + bry) / 2)
                    centroid_dets.append((x, y))
                    dets_dict[(x, y)] = lb
                    labels.append(lb)
                    # Features
                    time_1 = time.time()
                    # features = hog.compute(cv2.cvtColor(cv2.resize(frame[tly:bry, tlx:brx,:],(64,128)), cv2.COLOR_BGR2GRAY))
                    features = np.squeeze(hog.compute(cv2.resize(frame[tly:bry, tlx:brx,:],(64,128))))
                    # features, hog_image = hog(cv2.resize(frame[tly:bry, tlx:brx,:],(64,128)), orientations=9, pixels_per_cell=(8, 8),
                    #                           cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                    #                           visualize=True)
                    print('----------{}------------'.format(time.time()-time_1))
                    # dets.append([tlx, tly, brx, bry, conf])
                    dets.append(Detection([tlx, tly, brx, bry], conf, features))

            boxes = np.array([d.tlwh for d in dets])
            scores = np.array([d.confidence for d in dets])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            dets = [dets[i] for i in indices]
            
            '''
            # To test SORT
            dets = np.asarray(dets)
            tracks = tracker.update(dets)
            boxes = []
            indexIDs = []
            previous = memory.copy()
            memory = {}

            for j, track in enumerate(tracker.tracks):
                # AM: Print ID - Class - Confidence
                (x1, y1) = (int(track[0]), int(track[1]))
                (x2, y2) = (int(track[2]), int(track[3]))
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)

                # track[0] a [3] -> bounding box
                # track[4] -> ID
                boxes.append([track[0], track[1], track[2], track[3], int(
                    track[4]), dets_dict[closest_node((x, y), centroid_dets)], dets[j, -1]])
                indexIDs.append(int(track[4]))

                memory[indexIDs[-1]] = boxes[-1]
            '''

            # To test DEEP_SORT
            tracker.predict()
            tracker.update(dets)
            
            boxes = []
            indexIDs = []
            previous = memory.copy()
            memory = {}

            for j, track in enumerate(tracker.tracks):
                # AM: Print ID - Class - Confidence
                bbox = track.to_tlwh()
                (x1, y1) = (int(bbox[0]), int(bbox[1]))
                (x2, y2) = (int(bbox[2]), int(bbox[3]))
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)

                # track[0] a [3] -> bounding box
                # track[4] -> ID
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], 
                track.track_id, dets_dict[closest_node((x, y), centroid_dets)]])
                indexIDs.append(track.track_id)

                memory[indexIDs[-1]] = boxes[-1]


            for box in boxes:
                text = "{} - {}".format(box[-2], box[-1])
                cv2.putText(frame, text, (int(box[0]), int(
                    box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(
                    box[2]), int(box[3])), (0, 255, 0), 2)

        cv2.imshow('tracking', frame)
        cv2_key = cv2.waitKey(1)
        if cv2_key & 0xFF == ord('q'):
            break

        # log.info('Processing time: {} ms'.format((time.time() - time_init)*1000))
        log.info('Processing FPS: {:.1f}'.format(1/(time.time() - time_init)))
