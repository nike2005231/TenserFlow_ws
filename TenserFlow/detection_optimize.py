import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util, config_util, visualization_utils as viz_utils
from object_detection.builders import model_builder

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
PATH_TO_LABELS = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\annotations\label_map.pbtxt"
PATH_TO_CFG = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\exported-models\my_model\pipeline.config"
PATH_TO_CKPT = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\exported-models\my_model\checkpoint"

# Load model
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, image_np = cap.read()
    if not ret:
        break

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Visualize detected bounding boxes and labels
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,  # Reduce number of boxes to draw
        min_score_thresh=.80,  # Increase minimum score threshold
        agnostic_mode=False
    )

    # Display output
    cv2.imshow('Object Detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()