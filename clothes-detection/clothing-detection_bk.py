import sys
import time
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('./util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image, write_predictions  # noqa: E402
from webcamera_utils import get_capture, get_writer, calc_adjust_fsize  # noqa: E402
from imutils.video import FPS, WebcamVideoStream
import imutils

# logger
from logging import getLogger   # noqa: E402
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


# ======================
# Arguemnt Parser Config
# ======================
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


# ======================
# Secondary Functions
# ======================

def letterbox_image(image, size):
    """
    Resize an image while maintaining its aspect ratio using padding.

    Parameters:
    image (PIL.Image.Image): The input image to be resized.
    size (tuple): The desired output size as a tuple (width, height).

    Returns:
    PIL.Image.Image: The resized image with padding to fit the desired size.
    """
    '''resize image with unchanged aspect ratio using padding'''
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
    """
    Preprocess the input image for model prediction.

    Args:
        img (numpy.ndarray): The input image array.
        resize (int): The size to which the image should be resized.

    Returns:
        numpy.ndarray: The preprocessed image data ready for model input.
    """
    image = Image.fromarray(img)
    boxed_image = letterbox_image(image, (resize, resize))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data = np.transpose(image_data, [0, 3, 1, 2])
    return image_data


def post_processing(img_shape, all_boxes, all_scores, indices):
    """
    Post-processes the output of a detection model to generate bounding boxes.

    Args:
        img_shape (tuple): The shape of the input image as (height, width).
        all_boxes (numpy.ndarray): Array containing the coordinates of all detected boxes.
        all_scores (numpy.ndarray): Array containing the scores of all detected boxes.
        indices (numpy.ndarray): Array containing the indices of the selected boxes.

    Returns:
        list: A list of ailia.DetectorObject instances representing the detected bounding boxes.
    """
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



# ======================
# Main functions
# ======================

def detect_objects(img, detector):
    """
    Detect objects in an image using a specified detector.

    Args:
        img (numpy.ndarray): The input image in which to detect objects.
        detector (object): The object detection model used to predict objects in the image.

    Returns:
        list: A list of detected objects after post-processing. Each detected object typically includes
              information such as bounding box coordinates, confidence scores, and class labels.

    """
    img_shape = img.shape[:2]

    # initial preprocesses
    img = preprocess(img, resize=args.detection_width)

    # feedforward
    all_boxes, all_scores, indices = detector.predict({
        'input_1': img,
        'image_shape': np.array([img_shape], np.float32),
        'layer.score_threshold': np.array([args.threshold if args.threshold else THRESHOLD], np.float32),
        'iou_threshold': np.array([IOU], np.float32),
    })

    # post processes
    detect_object = post_processing(img_shape, all_boxes, all_scores, indices)

    return detect_object


def recognize_from_video(video, detector):
    """
    Recognize objects from a video using a specified detector.

    Args:
        video (str): Path to the video file.
        detector (object): Object detection model to use for recognizing objects.

    Returns:
        None

    This function captures frames from the specified video, uses the provided
    detector to recognize objects in each frame, and displays the results in a
    window. If a save path is specified, it also saves the processed video.

    The function will terminate when the 'q' key is pressed or when the video
    ends. It releases the video capture and writer resources and closes all
    OpenCV windows upon completion.
    """
    vs = get_capture(video)
    fps = FPS().start()

    frame_shown = False
    while True:
        frame = vs.read()
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
    vs.stop()


def main():
    """
    Main function to perform clothes detection using a pre-trained model.

    This function checks and downloads the necessary model files, initializes
    the detector, and processes either video or image inputs for clothes detection.

    The function performs the following steps:
    1. Checks and downloads the model files if not already present.
    2. Initializes the detector with the specified model and weight paths.
    3. Sets the input shape for the detector.
    4. Processes the input based on the mode (video or image):
       - If a video file is provided, it processes the video for clothes detection.
       - If image files are provided, it processes each image for clothes detection.

    Args:
        None

    Returns:
        None
    """
    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    detector = ailia.Net(model_path, weight_path, env_id=args.env_id)
    id_image_shape = detector.find_blob_index_by_name("image_shape")
    detector.set_input_shape(
        (1, 3, args.detection_width, args.detection_width)
    )
    detector.set_input_blob_shape((1, 2), id_image_shape)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, detector)

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
