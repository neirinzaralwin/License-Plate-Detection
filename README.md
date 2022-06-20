# Webcam License-Plate-Detection
<p>Implementing License Plate Detection with <b>google colab</b> and some <b>javascript</b> codes</p>

![gif](https://user-images.githubusercontent.com/56222774/172134219-45e684c4-8b4b-4cab-954b-d0f71e86a405.gif)
![output_img](https://user-images.githubusercontent.com/56222774/172138612-5ef17347-cc7f-44c4-b7d7-56add4df7f65.png)

<h3>This Repository include:</h3>

- Recive Data From [Roboflow API](https://app.roboflow.com/)
- Custom Configuration
- Training The Detector
- Evaluation with Tensorboad and Save Model
- **Upload Model and Checkpoint From Your PC
- **Realtime Detection with Webcam on Google Colab

## Data Resources
I manually downloaded car license plate from Yangon Region and Mandalay Region including taxi plate. Approximately **150 photos**.

## Pretrained Model
I used <code>Faster R-CNN ResNet50 V1 640x640</code> from [Tensorflow Object Detection API Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

> I found alot of comparison what faster rcnn is better than ssd mobile net. [Youtube Video](https://www.youtube.com/watch?v=nfE0rsnJUwI&ab_channel=Chih-LyangHwang)

## Model Architecture
<img src='https://camo.githubusercontent.com/0eed6f561e63f5a146c100a4ff96fe958827388b4015681e94f1dc9e0e28e95e/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4e79616e5377616e41756e672f5468652d537061726b732d466f756e646174696f6e2d496e746572736869702f6d61696e2f6173736574732f6661737465722d72636e6e2e706e67'></img>

## Quick Setup
**Install Some APIs and required tool**

```python
import os
!pip install tensorflow==2.8
!pip install opencv-python-headless==4.1.2.30
```

```python
!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```
**Download Tensorflow Model from github**
```python
if not os.path.exists("models"):
  !git clone https://github.com/tensorflow/models
```
```python
# protobuf is like json. It is made from google. install some protobuf file.
os.chdir('./models/research/')
!protoc object_detection/protos/*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
```
```python
# test your tensorflow model
!python ./object_detection/builders/model_builder_tf2_test.py
```
**Add files**
1. **Add model_config.config, labelmap.pbtxt to /content/models/research/ file path.**
2. **Create training folder at /content/models/research/**
3. **Add [ckpt-8.data-00000-of-00001](https://drive.google.com/file/d/18oh-2p5XNy7zt9LIQ8-M6tR2wWRJvIPg/view?usp=sharing)(file size is big) and ckpt-8.index to training folder.**
4. **Create Saved Model folder in inside training folder.**
5. **Add saved_model.pb in Saved Model folder.**

**Install packages**
```python
import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import html
import time
import pathlib
import cv2


from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

%matplotlib inline
```

**Load Model**
```python
labelmap_path = './labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
```
```python
print("Loading Model...", end='')
start_time = time.time()

configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-8')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
```
**Test Image (you can add image named test.jpg to content/models/research/ to test)***
```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


IMAGE_PATHS = ['/content/models/research/test.jpg']

for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    vis_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
plt.show()
```
**Detection from Webcam on google colab**
```python
import PIL
import tensorflow_hub as hub

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
```
```python
# JavaScript to properly create our live video stream using our webcam as input
def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;
    
    var pendingResolve = null;
    var shutdown = false;
    
    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }
    
    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 640);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }
    
    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);
      
      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);
           
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);
      
      const instruction = document.createElement('div');
      instruction.innerHTML = 
          '<span style="color: red; font-weight: bold;">' +
          'When finished, click here or on the video to stop this demo</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };
      
      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 640; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);
      
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();
      
      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }
            
      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }
      
      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;
      
      return {'create': preShow - preCreate, 
              'show': preCapture - preShow, 
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')

  display(js)
  
def video_frame(label, bbox):
  data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
  return data

# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array) # numpy array into PIL image object
  iobuf = io.BytesIO()
  
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')

  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes
```
**Open Webcam from colab**
```python
video_stream()
# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = None
count = 0 
while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    image_np = js_to_image(js_reply["img"])


    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=20,
            min_score_thresh=.80,
            agnostic_mode=False)   
    
    
    # Convert numpy image type to Base64 image byte string type
    # for JavaScript Video object 
    bbox_bytes = bbox_to_bytes(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))

    # update bbox so next frame gets new overlay
    bbox = bbox_bytes
```

## Reference
[Object Detection](https://github.com/NyanSwanAung/The-Sparks-Foundation-Intership/tree/main/Task1-Object%20Detection)
