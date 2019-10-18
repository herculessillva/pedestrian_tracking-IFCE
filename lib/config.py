import logging

YOLO_PEDESTRIAN_OPTIONS = {"model": ".conf/pedestrian/yolo-voc.cfg",
                            "load": ".conf/pedestrian/yolo-voc.weights",
                            "labels": ".conf/pedestrian/voc.names",
                            "threshold": 0.5, 
                            "gpu": 0.7}

URL_RSTP = 'rtsp://admin:l@pisco2019@10.102.1.151:554/cam/realmonitor?channel=1&subtype=0'

LOGGER = logging.getLogger()
handler = logging.StreamHandler()
# formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s","%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)