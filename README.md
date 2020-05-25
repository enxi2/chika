# chika

A simple object tracking video compositor using OpenCV

## Setup

Extract frames as jpegs.

```
mkdir frames
ffmpeg -i "chika.mp4" -q:v 1 "frames/%05d.jpg"
cp frames/00001.jpg frames/00000.jpg
```

Use a ramfs drive for faster processing.

## Process

```
mkdir output
python align_video.py
```

Output will be pngs at scaled resolution.

## Reconstruction

```
ffmpeg -framerate 23.98 -i "output/%05d.png" -i "chika.mp4" -vf "scale=1920:1080" -c:v h264_nvenc -map 0:0 -map 1:1 -shortest chika_clean.mp4
```

## TODO

* Tune AKAZE
* Add a kalman filter or smoothing
* Build a map of the entire environment
