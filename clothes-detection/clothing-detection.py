# Imports retained as per your setup
import sys
import subprocess
import time
from collections import OrderedDict
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
from PIL import Image
import ailia
import threading

# Original utility imports
sys.path.append('./util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image, write_predictions  # noqa: E402
from webcamera_utils import get_capture, get_writer, calc_adjust_fsize  # noqa: E402
from imutils.video import FPS, WebcamVideoStream, VideoStream # noqa: E402
import imutils

# Logger setup
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clothing-detection/'

DATASETS_MODEL_PATH = OrderedDict([
    (
        'modanet',
        ['yolov3-modanet.opt.onnx', 'yolov3-modanet.opt.onnx.prototxt']
    ),
    ('df2', ['yolov3-df2.opt.onnx', 'yolov3-df2.opt.onnx.prototxt'])
])

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

DATASETS_CATEGORY = {
    'modanet': [
        "bag", "belt", "boots", "footwear", "outer", "dress", "sunglasses",
        "pants", "top", "shorts", "skirt", "headwear", "scarf/tie"
    ],
    'df2': [
        "short sleeve top", "long sleeve top", "short sleeve outwear",
        "long sleeve outwear", "vest", "sling", "shorts", "trousers", "skirt",
        "short sleeve dress", "long sleeve dress", "vest dress", "sling dress"
    ]
}
THRESHOLD = 0.39
IOU = 0.4
DETECTION_WIDTH = 416

# Argument parser
parser = get_base_parser(
    'Clothing detection model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-d', '--dataset', metavar='TYPE', choices=DATASETS_MODEL_PATH,
    default=list(DATASETS_MODEL_PATH.keys())[0],
    help=('Type of dataset to train the model. '
          'Allowed values are {}.'.format(', '.join(DATASETS_MODEL_PATH)))
)
parser.add_argument(
    '-dw', '--detection_width',
    default=DETECTION_WIDTH,
    help='The detection width and height for yolo.'
)
parser.add_argument(
    '-th', '--threshold', type=float,
    default=THRESHOLD,
    help='The threshold for detection.'
)
parser.add_argument(
    '-w', '--write_json',
    action='store_true',
    help='Flag to output results to json file.'
)
args = update_parser(parser)

weight_path, model_path = DATASETS_MODEL_PATH[args.dataset]
category = DATASETS_CATEGORY[args.dataset]

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess(img, resize):
    image = Image.fromarray(img)
    boxed_image = letterbox_image(image, (resize, resize))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data = np.transpose(image_data, [0, 3, 1, 2])
    return image_data


def post_processing(img_shape, all_boxes, all_scores, indices):
    indices = indices.astype(int)
    bboxes = []
    for idx_ in indices[0]:
        cls_ind = idx_[1]
        score = all_scores[tuple(idx_)]
        idx_1 = (idx_[0], idx_[2])
        box = all_boxes[idx_1]
        y, x, y2, x2 = box
        w = (x2 - x) / img_shape[1]
        h = (y2 - y) / img_shape[0]
        x /= img_shape[1]
        y /= img_shape[0]

        r = ailia.DetectorObject(
            category=cls_ind, prob=score,
            x=x, y=y, w=w, h=h,
        )
        bboxes.append(r)

    return bboxes


def detect_objects(img, detector):
    img_shape = img.shape[:2]
    img = preprocess(img, resize=args.detection_width)
    all_boxes, all_scores, indices = detector.predict({
        'input_1': img,
        'image_shape': np.array([img_shape], np.float32),
        'layer.score_threshold': np.array([args.threshold if args.threshold else THRESHOLD], np.float32),
        'iou_threshold': np.array([IOU], np.float32),
    })
    detect_object = post_processing(img_shape, all_boxes, all_scores, indices)
    return detect_object

# ======================
# Flask, Websockets, and Detection Setup
# ======================

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable websockets
vs = None
detector = None

def gen_frames():
    global vs, detector
    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=640)
        # result_frame = plot_results(detected_objects, frame, category)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpeg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

        # Send detection results via websockets
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run object detection
        # detected_objects = detect_objects(frame_rgb, detector)
        # results = [
        #     {
        #         "category": category[obj.category],
        #         "probability": float(obj.prob),
        #         "bounding_box": [obj.x, obj.y, obj.w, obj.h]
        #     }
        #     for obj in detected_objects
        # ]
        # socketio.emit('detection_results', {'objects': results})

@app.route('/')
def index():
    return render_template('index.html')  # Render the main page (index.html)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def on_connect():
    logger.info('Client connected via websocket.')

@socketio.on('disconnect')
def on_disconnect():
    logger.info('Client disconnected.')

def main():
    global detector, vs
    # Check and download model files
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # Initialize the detector
    detector = ailia.Net(model_path, weight_path, env_id=args.env_id)
    detector.set_input_shape((1, 3, args.detection_width, args.detection_width))

    # Start the video stream
    vs = VideoStream(src=0).start()

    # Start Flask and SocketIO server in a separate thread
    threading.Thread(target=lambda: socketio.run(app, host="localhost", port=5000), daemon=True).start()
    logger.info('Flask and WebSocket server started, waiting for connections...')

    # Main loop for FPS tracking
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()