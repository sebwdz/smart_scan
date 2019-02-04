
import cv2

import tensorflow as tf
import numpy as np

import time

import PIL.Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_FROZEN_GRAPH = 'frozen1' + '/frozen_inference_graph.pb'

smil = PIL.Image.open('data/smil.png').convert('RGB')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_session(graph):
    with graph.as_default():
        sess = tf.Session()
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        print(tensor_dict)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    return sess, tensor_dict, image_tensor


def run_inference_for_single_image(image, sess, tensor_dict, image_tensor):

    print(image.shape)

    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


category_index = label_map_util.create_category_index_from_labelmap('data/label_map.pbtxt', use_display_name=True)


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False


sess, tensor_dict, image_tensor = load_session(detection_graph)

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    t = time.time()

    output_dict = run_inference_for_single_image(frame, sess, tensor_dict, image_tensor)

    t2 = time.time()

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    key = cv2.waitKey(2)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")