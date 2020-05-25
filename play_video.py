import cv2
import glob
import sys
import time

def play_with_opencv():
  vc = cv2.VideoCapture("chika.mp4")
  if not vc.isOpened():
    print("Error opening video stream")
    sys.exit(1)

  fps = vc.get(cv2.CAP_PROP_FPS)
  mspf = 1000.0 / fps

  key = None
  last_frame_time = time.time()
  while key != 27: # Escape to quit
    ret, frame = vc.read()
    if ret:
      cv2.imshow("player", frame)

    current_time = time.time()
    ms = int(mspf - (current_time - last_frame_time))
    key = cv2.waitKey(ms)

  vc.release()
  cv2.destroyAllWindows()


def play_with_jpg():
  frames = sorted(glob.glob("frames/*.jpg"))

  fps = 23.98
  mspf = 1000.0 / fps

  mx = 0
  my = 0
  cx = 0
  cy = 0
  def capture_mouse(e, x, y, f, p):
    nonlocal mx, my, cx, cy
    mx = x
    my = y

    if e == cv2.EVENT_LBUTTONDOWN:
      cx = x
      cy = y

  cv2.namedWindow("player")
  cv2.setMouseCallback("player", capture_mouse)

  playing = False
  key = None
  current_frame = 0
  last_frame_time = time.time()
  while key != 27: # Escape to quit
    # Read the current frame
    frame = cv2.imread(frames[current_frame])
    # Write some info onto the frame
    t = "{: 5d}".format(current_frame)
    cv2.putText(frame, t, (8, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    t = "{: 5d} {: 5d}     {: 5d} {: 5d}".format(mx, my, cx, cy)
    cv2.putText(frame, t, (512, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("player", frame)
    del frame

    if playing:
      current_frame += 1

    current_time = time.time()
    dt = current_time - last_frame_time
    last_frame_time = current_time
    ms = int(mspf - dt * 1000)
    if ms < 1:
      ms = 1
    key = cv2.waitKey(ms)

    if key == 32: # Space
      playing = not playing
      last_frame_time = time.time()
    elif key == ord('j'): # j
      current_frame -= 1
    elif key == ord('k'): # k
      current_frame += 1
    elif key == ord('h'): # h
      current_frame -= 10
    elif key == ord('l'): # l
      current_frame += 10

    if current_frame < 0:
      current_frame = 0
    if current_frame >= len(frames):
      current_frame = len(frames) - 1

  cv2.destroyAllWindows()

if __name__ == "__main__":
  #play_with_opencv()
  play_with_jpg()
