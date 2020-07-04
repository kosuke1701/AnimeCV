# AnimeCV

Pretrained computer vision tools for anime style illustrations.

## Getting started

Install AnimeCV by pip:

```
pip install git+https://github.com/kosuke1701/AnimeCV.git
```

## Features

### Character re-identification

The task is to identify which character is depicted in a picture given a set of reference pictures for each character.
[Example code](example/character_re_identification.py)

### Object detection

The task is to detect objects in a picture and return bounding boxes around the objects.

* Face detector. [Example code](example/character_face_detection.py)

## Note

All models are trained on datasets which mainly consists of Japanese anime style illustrations. Thus, it may not perform well on illustrations with other styles.