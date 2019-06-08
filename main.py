# Localização da iris por Hough Cirles e
# indicação da direção de visão.
#
# Trabalho T5
#
# Aluno: Eliel Marcos Romancini - 15205574
#

import cv2
import numpy as np
import math


def nothing(x):
    pass


sizex = 500
sizey = 300

cam = cv2.VideoCapture('Eyes.mp4')
out = cv2.VideoWriter('output.mp4', -1, 30.0, (500, 300))

ret, frame = cam.read()

while ret:

    frame = cv2.resize(frame, (sizex, sizey))

    frameT = frame.copy()

    cv2.putText(frame, 'Iris Positioning - Eliel Romancini', (250, 295), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0))

    frame_pupil = cv2.medianBlur(frame, 19)

    frame_pupil_gray = cv2.cvtColor(frame_pupil, cv2.COLOR_BGR2GRAY)

    frame_eye = cv2.GaussianBlur(frame, (55, 55), sigmaX=0, sigmaY=0)

    frame_eye_hsv = cv2.cvtColor(frame_eye, cv2.COLOR_BGR2HSV)

    frame_eye_bin1 = cv2.inRange(frame_eye_hsv, (0, 45, 4), (15, 255, 255))
    frame_eye_bin2 = cv2.inRange(frame_eye_hsv, (175, 45, 4), (179, 255, 255))
    frame_eye_bin = frame_eye_bin1 + frame_eye_bin2
    frame_eye_bin = cv2.morphologyEx(frame_eye_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    frame_eye_bin = cv2.morphologyEx(frame_eye_bin, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    frame_eye_bin = cv2.bitwise_not(frame_eye_bin)

    #  Encontrar o contorno do olho
    contorns = cv2.findContours(frame_eye_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[0]
    circles = []
    center = 25
    offset = 25

    if not (contorns == []):
        cnt = max(contorns, key=cv2.contourArea)
        cnt_approx = cv2.convexHull(cnt)
        cnt_moments = cv2.moments(cnt_approx)
        cnt_area = cnt_moments['m00']
        if cnt_area > 12000:
            #  Encontrou olho
            EyeX = int(cnt_moments['m10'] / cnt_area)
            EyeY = int(cnt_moments['m01'] / cnt_area)
            cv2.drawContours(frame, [cnt_approx], -1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (EyeX, EyeY), center, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (EyeX, 0 + offset), (EyeX, sizey - offset), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (0 + offset, EyeY), (sizex - offset, EyeY), (255, 0, 0), 1, cv2.LINE_AA)

            #  Encontrar Pupila
            circles = cv2.HoughCircles(frame_pupil_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=90, param2=20,
                                       minRadius=15, maxRadius=38)

            if (circles is None) or (circles == []):
                cv2.imshow('video', frame)
                out.write(frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

                ret, frame = cam.read()
                continue

            #  Encontrou Pupila
            for (PupilX, PupilY, PupilR) in circles[0, :]:
                PupilX = int(PupilX)
                PupilY = int(PupilY)
                PupilR = int(PupilR)
                cv2.circle(frame, (PupilX, PupilY), PupilR, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(frame, (PupilX, PupilY), (EyeX, EyeY), (255, 0, 255), 2, cv2.LINE_AA)

                # Calcula angulo em relação ao eixo X
                vectorX = [1, 0]
                vectorPE = [PupilX - EyeX, PupilY - EyeY]
                if vectorPE == [0, 0]:
                    angle = 0
                else:
                    angle = math.acos(((vectorX[0] * vectorPE[0]) + (vectorX[1] * vectorPE[1])) / (
                            math.hypot(vectorX[0], vectorX[1]) * math.hypot(vectorPE[0], vectorPE[1]))) * 180 / math.pi
                if vectorPE[1] > 0:
                    angle = 360 - angle

                #  Calcula distância entre centro do olho e centro da pupila
                dist = ((PupilX - EyeX) ** 2 + (PupilY - EyeY) ** 2) ** (1 / 2)

                if dist < center:
                    cv2.putText(frame, 'CENTER', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                else:
                    if angle > 15 and angle <= 75:
                        cv2.putText(frame, 'TOP / LEFT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                    if angle > 75 and angle <= 105:
                        cv2.putText(frame, 'TOP', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                    if angle > 105 and angle <= 165:
                        cv2.putText(frame, 'TOP / RIGHT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                    if angle > 165 and angle <= 195:
                        cv2.putText(frame, 'RIGHT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                    if angle > 195 and angle <= 255:
                        cv2.putText(frame, 'DOWN / RIGHT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                    if angle > 255 and angle <= 285:
                        cv2.putText(frame, 'DOWN', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                    if angle > 285 and angle <= 345:
                        cv2.putText(frame, 'DOWN / LEFT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))
                    if angle > 345 or angle <= 15:
                        cv2.putText(frame, 'LEFT', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255))

    cv2.imshow('video', frame)
    out.write(frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    ret, frame = cam.read()

cam.release()
out.release()
cv2.destroyAllWindows()
