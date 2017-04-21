# sample usage: python play_video.py g0.avi
# g*.avi is generated video
# t*.avi is truth video

import sys
import skvideo.io
import cv2

# get video name
video_name = sys.argv[1]

# load video
video = skvideo.io.vreader('VideoDataSet/{}'.format(video_name))

# display video
for frame in video:
    # resize frame to 360 by 360
    frame = cv2.resize(frame, (360, 360))
    # display frame
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
