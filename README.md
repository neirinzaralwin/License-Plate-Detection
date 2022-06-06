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

## Reference
[Object Detection](https://github.com/NyanSwanAung/The-Sparks-Foundation-Intership/tree/main/Task1-Object%20Detection)
