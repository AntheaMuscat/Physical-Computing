import sys
import subprocess
import time
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image
import ailia

# Import original modules
sys.path.append('./util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import plot_results, load_image, write_predictions
from webcamera_utils import get_capture, get_writer, calc_adjust_fsize  # noqa: E402
from imutils.video import FPS
import imutils

# Logger
from logging import getLogger
logger = getLogger(__name__)

# Parameters
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clothing-detection/'

DATASETS_MODEL_PATH = OrderedDict([
    ('modanet', ['yolov3-modanet.opt.onnx', 'yolov3-modanet.opt.onnx.prototxt']),
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

# Argument Parser
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


# Secondary Functions

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
    image_data = np.expand_dims(image_data, 0)
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


# Main functions

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


def recognize_from_video_stream(video_url, detector):
    # Open the webcam stream using OpenCV (motion streaming URL)
    vs = cv2.VideoCapture(video_url)
    fps = FPS().start()

    frame_shown = False
    while True:
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame = imutils.resize(frame, width=640)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_object = detect_objects(x, detector)
        res_img = plot_results(detect_object, frame, category)
        cv2.imshow('frame', res_img)
        frame_shown = True

        fps.update()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.release()

def start_motion_service():
    # Start the motion service in the background
    subprocess.Popen(['sudo', 'motion'])

def main():
    start_motion_service()
    # Model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # Initialize the detector
    detector = ailia.Net(model_path, weight_path, env_id=args.env_id)
    id_image_shape = detector.find_blob_index_by_name("image_shape")
    detector.set_input_shape((1, 3, args.detection_width, args.detection_width))
    detector.set_input_blob_shape((1, 2), id_image_shape)

    # Set video stream URL (motion service)
    video_url = f"{args.video}"  # Replace with your Raspberry Pi IP

    if args.video is not None:
        # Recognize from video stream
        recognize_from_video_stream(video_url, detector)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
