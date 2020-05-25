import cv2
import glob
import numpy
import os
import skimage.measure
import sys
import time

# ffmpeg -i "chika.mp4" -q:v 1 "frames/%05d.jpg"
# cp frames/00001.jpg frames/00000.jpg

ALIGNMENTS = [
  { # 0
    'mask': [311, 1208, 420, 1730, 622],
    #'tracker': [311, 1010, 616, 1142, 697], too small, few keypoints
    'tracker': [311, 1010, 516, 1242, 797],
    'start': 312,
    'end': 402,
  },
  { # 1
    'mask': [410, 1323, 246, 1830, 823],
    'tracker': [410, 1079, 530, 1322, 874],
    'start': 411,
    'end': 509,
  },
  { # 2
    'mask': [536, 1287, 318, 1844, 739],
    'tracker': [536, 108, 278, 709, 790],
    'start': 537,
    'end': 626,
  },
  { # 3
    'mask': [632, 1271, 185, 1886, 894],
    'tracker': [632, 236, 358, 740, 852],
    'start': 633,
    'end': 722,
  },
  { # 4
    'mask': [728, 1218, 242, 1889, 771],
    'tracker': [728, 188, 389, 655, 708],
    'start': 729,
    'end': 818,
  },
  { # 5
    'mask': [824, 1122, 136, 1814, 942],
    'tracker': [824, ],
    'start': 825,
    'end': 914,
  },
  { # 6
    'mask': [920, 1378, 236, 1818, 822],
    'tracker': [920, ],
    'start': 921,
    'end': 1016,
  },
  { # 7
    'mask': [1087, 101, 137, 706, 943],
    'tracker': [1087, ],
    'start': 1088,
    'end': 1207,
  },
  { # 8
    'mask': [1219, 76, 160, 516, 918],
    'tracker': [1219, ],
    'start': 1220,
    'end': 1339,
  },
  { # 9
    'mask': [1405, 0, 0, 0, 0],
    'tracker': [1405, ],
    'start': 1406,
    'end': 1525,
  },
  { # 10
    'mask': [1539, 93, 222, 566, 860],
    'tracker': [1539, ],
    'start': 1540,
    'end': 1642,
  },
  { # 11
    'mask': [1656, 106, 132, 667, 948],
    'tracker': [1656, ],
    'start': 1657,
    'end': 1752,
  },
  { # 12
    'mask': [1765, 77, 204, 565, 823],
    'tracker': [1765, ],
    'start': 1766,
    'end': 1855,
  },
  { # 13
    'mask': [1862, 1245, 268, 1815, 808],
    'tracker': [1862, ],
    'start': 1863,
    'end': 1958,
  },
  { # 14
    'mask': [1991, 1191, 316, 1825, 790],
    'tracker': [1991, ],
    'start': 1992,
    'end': 2087,
  },
]

# Rescale alignments
for alignment in ALIGNMENTS:
  if len(alignment["mask"]) == 5:
    alignment["mask"][1] *= 2
    alignment["mask"][2] *= 2
    alignment["mask"][3] *= 2
    alignment["mask"][4] *= 2

  if len(alignment["tracker"]) == 5:
    alignment["tracker"][1] *= 2
    alignment["tracker"][2] *= 2
    alignment["tracker"][3] *= 2
    alignment["tracker"][4] *= 2

SCALE = 2

class BruteForceAligner:

  def __init__(self, tracker, tracker_align):
    self.tracker = tracker[tracker_align[2]:tracker_align[4], tracker_align[1]:tracker_align[3]]
    self.ratio = (tracker_align[4] - tracker_align[2]) / (tracker_align[3] - tracker_align[1])
    self.last_transform = ( tracker_align[1], tracker_align[2], tracker_align[3] - tracker_align[1] )
    self.tx = tracker_align[1]
    self.ty = tracker_align[2]
    self.tw = tracker_align[3] - tracker_align[1]
    self.th = tracker_align[4] - tracker_align[2]

  def get_transform(self, frame):
    x, y, w = self.last_transform
    bx, by, bw = x, y, w
    bdiff = None
    btracker = None
    bframe = None
    for sw in range(w - 5, w + 5):
      sh = int(sw * self.ratio)
      stracker = cv2.resize(self.tracker, (sw, sh), interpolation=cv2.INTER_NEAREST)
      for sx in range(x - 10, x + 10):
        for sy in range(y - 10, y + 10):
          sframe = frame[sy:(sy+sh), sx:(sx+sw)]
          diff = numpy.average(numpy.square(sframe.astype(numpy.float) - stracker.astype(numpy.float)))
          #diff = skimage.measure.compare_mse(sframe, stracker)
          #diff = skimage.measure.compare_ssim()
          if bdiff is None or diff < bdiff:
            bdiff = diff
            bx, by, bw = sx, sy, sw
            btracker = stracker
            bframe = sframe

    print(bx, by, bw, bdiff)
    self.bx, self.by, self.bw, self.bdiff = bx, by, bw, bdiff
    self.last_transform = (bx, by, bw)
    self.btracker = btracker
    self.bframe = bframe

    tracker_points = numpy.array([
      self.tx, self.ty,
      self.tx, self.ty + self.th,
      self.tx + self.tw, self.ty + self.th,
      self.tx + self.tw, self.ty,
    ]).reshape((-1, 2))
    bh = int(bw * self.ratio)
    frame_points = numpy.array([
      bx, by,
      bx, by + bh,
      bx + bw, by + bh,
      bx + bw, by,
    ]).reshape((-1, 2))
    matrix, inliers = cv2.estimateAffine2D(tracker_points, frame_points)
    return matrix, inliers

  def debug(self):
    img = numpy.hstack((self.btracker, self.bframe))
    cv2.imshow("best tracker", img)


class AkazeAligner:

  def __init__(self, tracker, tracker_align):
    self.tracker_align = tracker_align

    # Initialize the detector and matcher
    self.detector = cv2.AKAZE_create()
    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Read the tracker and get features
    self.tracker = tracker[tracker_align[2]:tracker_align[4], tracker_align[1]:tracker_align[3]]
    self.tracker_kp, self.tracker_des = self.detector.detectAndCompute(self.tracker, None)


  def get_transform(self, frame):
    self.last_frame = frame

    self.frame_kp, frame_des = self.detector.detectAndCompute(frame, None)

    # Match descriptors.
    self.matches = matches = self.matcher.match(self.tracker_des, frame_des)

    # Estimate the 2d transform
    tracker_points = numpy.zeros((len(matches), 2))
    frame_points = numpy.zeros((len(matches), 2))
    for i, m in enumerate(matches):
      x, y = self.tracker_kp[m.queryIdx].pt
      tracker_points[i][0] = x + self.tracker_align[1]
      tracker_points[i][1] = y + self.tracker_align[2]

      x, y = self.frame_kp[m.trainIdx].pt
      frame_points[i][0] = x
      frame_points[i][1] = y

    # Compute the affine transform
    matrix, inliers = cv2.estimateAffine2D(tracker_points, frame_points)

    return matrix, inliers


class AkazeFullAligner:

  def __init__(self, tracker):
    # Initialize the detector and matcher
    self.detector = cv2.AKAZE_create()
    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Read the tracker and get features
    self.tracker = tracker
    self.tracker_kp, self.tracker_des = self.detector.detectAndCompute(self.tracker, None)


  def get_transform(self, frame):
    self.last_frame = frame

    self.frame_kp, frame_des = self.detector.detectAndCompute(frame, None)

    # Match descriptors.
    self.matches = matches = self.matcher.match(self.tracker_des, frame_des)

    # Estimate the 2d transform
    tracker_points = numpy.zeros((len(matches), 2))
    frame_points = numpy.zeros((len(matches), 2))
    for i, m in enumerate(matches):
      x, y = self.tracker_kp[m.queryIdx].pt
      tracker_points[i][0] = x
      tracker_points[i][1] = y

      x, y = self.frame_kp[m.trainIdx].pt
      frame_points[i][0] = x
      frame_points[i][1] = y

    # Compute the affine transform
    matrix, inliers = cv2.estimateAffine2D(tracker_points, frame_points)

    return matrix, inliers


  def debug(self):
    img = cv2.drawKeypoints(self.tracker, self.tracker_kp, (255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("tracker keypoints", img)

    img = cv2.drawKeypoints(self.last_frame, self.frame_kp, (255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.rectangle(img, (new_mask_rect[0][0], new_mask_rect[1][0]),
            (new_mask_rect[0][1], new_mask_rect[1][1]), (0, 0, 255), 2)
    cv2.imshow("frame keypoints", img)

    img = cv2.drawMatches(self.tracker, self.tracker_kp, self.last_frame, self.frame_kp, self.matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    dt = int((end_time - start_time) * 1000)
    t = "{: 3d}  {: 5d}    {: 4d}ms".format(alignment_index, frame_index, dt)
    cv2.putText(img, t, (512, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("matches", img)


if __name__ == "__main__":
  sys.argv = sys.argv[1:]

  debug = False
  if len(sys.argv) >= 1 and sys.argv[0] == "debug":
    debug = True
    sys.argv = sys.argv[1:]

  alignment_filter = list(map(lambda x: int(x), sys.argv))

  if not os.path.exists("output"):
    os.mkdir("output")

  frames = sorted(glob.glob("frames/*.jpg"))

  for alignment_index, alignment in enumerate(ALIGNMENTS):
    if len(alignment_filter) > 0 and alignment_index not in alignment_filter:
      continue

    print("Alignment #{} [{} - {}]".format(alignment_index, alignment["start"], alignment["end"]))

    # Initialize the aligner
    tracker = cv2.imread(frames[alignment["tracker"][0]], cv2.IMREAD_COLOR)
    tracker = cv2.resize(tracker, (tracker.shape[1] * SCALE, tracker.shape[0] * SCALE), interpolation=cv2.INTER_LANCZOS4)
    #aligner = BruteForceAligner(tracker, alignment["tracker"])
    #aligner = AkazeAligner(tracker, alignment["tracker"])
    aligner = AkazeFullAligner(tracker)

    # Read the mask and create an alpha mask
    mask_align = alignment["mask"]
    mask = cv2.imread(frames[mask_align[0]], cv2.IMREAD_COLOR)
    mask = cv2.resize(mask, (mask.shape[1] * SCALE, mask.shape[0] * SCALE), interpolation=cv2.INTER_LANCZOS4)
    mask_alpha = numpy.zeros(mask.shape, numpy.float64)
    mask_alpha[mask_align[2]:mask_align[4], mask_align[1]:mask_align[3], :] = 1.0

    # Keep rectangle coordinates
    mask_rect = numpy.array([
      [mask_align[1], mask_align[2], 1],
      [mask_align[3], mask_align[4], 1],
      [mask_align[1], mask_align[4], 1],
      [mask_align[3], mask_align[2], 1],
    ]).transpose().astype(numpy.float64)

    for frame_index in range(alignment["start"], alignment["end"] + 1):
      print("Frame #{}".format(frame_index))
      start_time = time.time()

      # Read the frame and get features
      frame = cv2.imread(frames[frame_index], cv2.IMREAD_COLOR)
      frame = cv2.resize(frame, (frame.shape[1] * SCALE, frame.shape[0] * SCALE), interpolation=cv2.INTER_LANCZOS4)

      matrix, inliers = aligner.get_transform(frame)

      matrix3 = numpy.append(matrix, [[0, 0, 1]], axis=0)

      if sum(inliers) < len(inliers) * 0.8:
        print("Warning: too many outliers ({} / {})".format(sum(inliers), len(inliers)))

      # Compute new bounds for the mask
      new_mask_rect = numpy.matmul(matrix3, mask_rect).astype(numpy.int64)

      # Write output
      new_mask = cv2.warpAffine(mask.astype(numpy.float64), matrix, (mask.shape[1], mask.shape[0]))
      new_mask_alpha = cv2.warpAffine(mask_alpha, matrix, (mask_alpha.shape[1], mask_alpha.shape[0]))
      new_frame = cv2.multiply(frame.astype(numpy.float64), 1.0 - new_mask_alpha) + \
              cv2.multiply(new_mask, new_mask_alpha)
      new_frame = new_frame.astype(numpy.uint8)
      cv2.imwrite("output/{:05d}.png".format(frame_index), new_frame)

      end_time = time.time()
      print("Took {} seconds to render".format(end_time - start_time))
      if debug:
        cv2.imshow("output", new_frame)

        aligner.debug()

        if cv2.waitKey() == 27:
          sys.exit(1)
