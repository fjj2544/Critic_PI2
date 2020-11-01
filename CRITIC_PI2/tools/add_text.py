import numpy as np
import cv2
CURRENT_EPISODES = 10
if __name__ == "__main__":
    # step1: load in the video file
    videoCapture = cv2.VideoCapture('./test.mp4')

    # step2:get a frame
    sucess, frame = videoCapture.read()

    # save
    sz = (int(640),
          int(480))     # 窗口大小
    fps = 20
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    vout_1 = cv2.VideoWriter()
    vout_1.open('./output.mp4', fourcc, fps, sz, True)

    # step3:get frames in a loop and do process
    while (sucess):
        sucess, frame = videoCapture.read()
        if(sucess == False):
            cv2.destroyWindow('test Video')
            videoCapture.release()
            vout_1.release()
            break
        displayImg = cv2.resize(frame, sz,interpolation=cv2.INTER_CUBIC)  # resize it to (1024,768)
        cv2.putText(displayImg, f"After {CURRENT_EPISODES} episodes", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        cv2.namedWindow('test Video')
        vout_1.write(displayImg)
        cv2.imshow("test Video", displayImg)
        keycode = cv2.waitKey(1)
        if keycode == 27:
            cv2.destroyWindow('test Video')
            videoCapture.release()
            vout_1.release()
            break
    vout_1.release()
    videoCapture.release()
    cv2.destroyAllWindows()