import sys
import time
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import pairwise
import ailia
import os

from config import insert_into_database

# import original modules
sys.path.append('./util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image, write_predictions  # noqa: E402
from webcamera_utils import get_capture, get_writer,\
    calc_adjust_fsize  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clothing-detection/'

DATASETS_MODEL_PATH = OrderedDict([
    ('modanet', ['yolov3-modanet.opt.onnx', 'yolov3-modanet.opt.onnx.prototxt']),
    ('df2', ['yolov3-df2.opt.onnx', 'yolov3-df2.opt.onnx.prototxt'])
])

IMAGE_PATH = "captured_image.jpg"
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

# Define predefined colors for matching
COLOR_RANGES = {
    "Red": [(0, 100, 100), (10, 255, 255)],
    "Green": [(35, 100, 100), (85, 255, 255)],
    "Blue": [(100, 150, 0), (140, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Magenta": [(145, 100, 100), (175, 255, 255)],
    "Cyan": [(85, 100, 100), (95, 255, 255)],
    "Black": [(0, 0, 0), (180, 255, 30)],
    "White": [(0, 0, 200), (180, 30, 255)],
    "Gray": [(0, 0, 100), (180, 50, 200)],
    "Orange": [(10, 100, 100), (20, 255, 255)],
    "Pink": [(145, 75, 75), (175, 255, 255)],
    "Purple": [(125, 100, 100), (145, 255, 255)],
    "Brown": [(10, 100, 20), (20, 255, 80)]
}

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
    help='The detection width and height for yolo. (default: 416)'
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
    '''
    Resize image with unchanged aspect ratio using padding and apply normalization.

    Parameters:
    image (PIL.Image.Image): The input image to be resized.
    size (tuple): The desired output size as a tuple (width, height).

    Returns:
    PIL.Image.Image: The resized image with padding and normalized color.
    '''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    # Normalize the color channels
    image_np = np.array(new_image, dtype='float32') / 255.0
    image_np = np.clip(image_np, 0, 1)  # Clamp the values between 0 and 1
    new_image = Image.fromarray((image_np * 255).astype(np.uint8))

    return new_image

def preprocess(img, resize):
    image = Image.fromarray(img)
    print(f"Original Image Size: {image.size}")
    
    boxed_image = letterbox_image(image, (resize, resize))
    print(f"Processed Image Size: {boxed_image.size}")
    
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.0
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data = np.transpose(image_data, [0, 3, 1, 2])
    return image_data

def extract_colors(bbox_img):
    try:
        hsv_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2HSV)
        avg_color = cv2.mean(hsv_img)[:3]  # Exclude alpha channel
        color_name = get_closest_color_name(avg_color)
        return color_name
    except cv2.error as e:
        print(f"Error in color extraction: {e}")
        return None


def get_closest_color_name(hsv_color):
    '''Find the closest predefined color name to the given HSV color'''
    min_distance = float('inf')
    closest_color = None
    for color_name, color_hsv_range in COLOR_RANGES.items():
        dist = min(
            np.linalg.norm(np.array(hsv_color) - np.array(color_hsv_range[0])),
            np.linalg.norm(np.array(hsv_color) - np.array(color_hsv_range[1]))
        )
        if dist < min_distance:
            min_distance = dist
            closest_color = color_name
    return closest_color

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

# ======================
# Main functions
# ======================

def detect_objects(img, detector):
    img_shape = img.shape[:2]

    # initial preprocesses
    img = preprocess(img, resize=args.detection_width)

    # feedforward
    all_boxes, all_scores, indices = detector.predict({
        'input_1': img,
        'image_shape': np.array([img_shape], np.float32),
        'layer.score_threshold': np.array([THRESHOLD], np.float32),
        'iou_threshold': np.array([IOU], np.float32),
    })

    # post processes
    detect_object = post_processing(img_shape, all_boxes, all_scores, indices)

    return detect_object

def recognize_from_image(filename, detector):
    # Prepare input data
    img = load_image(filename)
    logger.debug(f'Input image shape: {img.shape}')

    x = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Inference
    logger.info('Start inference...')
    detect_object = detect_objects(x, detector)

    # Plot result
    res_img = plot_results(detect_object, img, category)
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'Saved at: {savepath}')
    cv2.imwrite(savepath, res_img)

    # Extract colors using HSV and add to database
    for detected_object in detect_object:
        if detected_object.category < len(category):
            x, y, w, h = detected_object.x, detected_object.y, detected_object.w, detected_object.h
            bbox_img = img[int(y*img.shape[0]):int((y+h)*img.shape[0]), int(x*img.shape[1]):int((x+w)*img.shape[1])]
            color_name = extract_colors(bbox_img)
            item_name = category[detected_object.category]  # Get item name from category
            print(f"Detected: {item_name}, Color: {color_name}")

            # Save cropped image
            cropped_image_path = os.path.join("Images", f"cropped_{item_name}_{color_name}.png")
            cv2.imwrite(cropped_image_path, bbox_img)

            # Insert into the database, including the image
            insert_into_database(item_name, color_name, cropped_image_path)

def main():
    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    detector = ailia.Net(model_path, weight_path, env_id=args.env_id)
    id_image_shape = detector.find_blob_index_by_name("image_shape")
    detector.set_input_shape(
        (1, 3, args.detection_width, args.detection_width)
    )
    detector.set_input_blob_shape((1, 2), id_image_shape)

    # image mode
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        recognize_from_image(image_path, detector)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
