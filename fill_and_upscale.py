import cv2
import glob
import os

# Fill in the output with upscaled frames.
# Make sure to match scale and interpolation mode.

SCALE = 2

if __name__ == "__main__":
  frames = sorted(glob.glob("frames/*.jpg"))

  for frame_index, frame in enumerate(frames):
    output_frame = "output/{:05d}.png".format(frame_index)
    if os.path.exists(output_frame):
      continue

    img = cv2.imread(frame, cv2.IMREAD_COLOR)
    simg = cv2.resize(img, (img.shape[1] * SCALE, img.shape[0] * SCALE), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(output_frame, simg)
