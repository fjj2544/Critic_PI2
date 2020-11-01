import numpy as np
import cv2
CURRENT_EPISODES = 10
sz = (int(640),
      int(480))     # 窗口大小
fps = 20
def load_and_save(videoCapture,vout,name=None):
    # step2:get a frame
    sucess, frame = videoCapture.read()
    # step3:get frames in a loop and do process
    while (sucess):
        sucess, frame = videoCapture.read()
        if (sucess == False):
            cv2.destroyWindow(f"test Video {name}")
            return
        displayImg = cv2.resize(frame, sz, interpolation=cv2.INTER_CUBIC)  # resize it to (1024,768)
        cv2.putText(displayImg, f"After {CURRENT_EPISODES} episodes", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (0, 0, 255), 2)
        cv2.namedWindow(f"test Video {name}")
        vout.write(displayImg)
        cv2.imshow(f"test Video {name}", displayImg)
        keycode = cv2.waitKey(1)
        if keycode == 27:
            cv2.destroyWindow(f"test Video {name}")
            return
if __name__ == "__main__":
    # step1: load in the video file
    videoCapture1 = cv2.VideoCapture('./test.mp4')
    videoCapture2 = cv2.VideoCapture('./test.mp4')



    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    vout1 = cv2.VideoWriter()
    vout1.open('./output.mp4', fourcc, fps, sz, True)
    load_and_save(videoCapture1,vout1, name="1")
    load_and_save(videoCapture2,vout1, name="2")

    vout1.release()
    videoCapture1.release()
    videoCapture2.release()
    cv2.destroyAllWindows()