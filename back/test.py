
import images_generator
import tensorflow as tf
import numpy as np

import time

import PIL.Image

from object_detection.utils import ops as utils_ops

import matplotlib
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

PIL.Image.open('data/fixed.png').convert('RGB').resize((400, 300)).save('fixed_resized.png')

matplotlib.use('TkAgg')

PATH_TO_FROZEN_GRAPH = 'frozen' + '/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
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

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
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
batch_reader = images_generator.get_next_image('./data/INRIAPerson/Test/i_*/*', shape=(300, 300), batch_size=10)

plt.ion()

for batch in batch_reader:
    for image_np in batch:
        image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)

      # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
        #plt.figure(figsize=(300, 300))
        plt.imshow(image_np)
        plt.draw()
        plt.pause(1)
