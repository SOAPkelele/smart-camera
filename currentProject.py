import time
import cv2
import numpy as np
import sys
import os, psutil
#   В силу специфики задачи   30, 60, 97 стоит заменить на 30,~45, 50. Но тогда зачастую выделяет части тела.
#       distanceAlongY увеличить, а не = distanceAlongX/2
def calculateFrameDifference(firstFrame, secondFrame, fgbg, minWidthObject=30, distanceAlongX=60, minHeightObject=97):
    widthList = []
    xStart = 0
    xEnd = 0
    rectanglesList = []
    distanceAlongY = distanceAlongX/2
    
    fgmask = fgbg.apply(firstFrame)#secondFrame 
    kernelSize = 10
    kernelOpen = np.ones((kernelSize, kernelSize), np.uint8)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernelOpen)
    edges = cv2.Canny(opening, 175, 255)
    
    xList = np.where( np.any(edges, axis=0) )[0]
    
    i = 0
    rectangleNumber = 0
    while i < len(xList):
        if ( (xList[i] > (xList[i - 1] + distanceAlongX) ) or \
            (i + 1 == len(xList) and xList[i]-xStart > distanceAlongX)):
            widthList.append( (xStart, xList[i - 1]) if i + 1 != len(xList) else (xStart, xList[i]) )
            xStart = xList[i]
            xEnd = xList[i]
                
            imagePartX = edges[:, widthList[rectangleNumber][0]:widthList[rectangleNumber][1] + 1]
            rowForRawImageResearch = np.zeros((1, widthList[rectangleNumber][1] - widthList[rectangleNumber][0] + 1),dtype=int)
            rectangleNumber += 1
            imagePartY = np.where(np.any(imagePartX, axis=1))[0]
            yStart = 0
            yEnd = 0
            numberYRawMatrix = 0
            while numberYRawMatrix < len(imagePartY):
                if ( (( (imagePartY[numberYRawMatrix]) > (imagePartY[numberYRawMatrix - 1] + distanceAlongY) ) and
                    numberYRawMatrix > 0) or \
                    (numberYRawMatrix + 1 == len(imagePartY) and imagePartY[numberYRawMatrix]-yStart > distanceAlongY)):
                    yEnd = numberYRawMatrix if numberYRawMatrix + 1 == len(imagePartY) else numberYRawMatrix - 1
                    if((yEnd - yStart) >= minHeightObject):
                            
                        imagePartY2 = imagePartX[imagePartY[yStart]:imagePartY[yEnd], :]
                        imagePartX2 = np.where(np.any(imagePartY2, axis=0))[0]
                            
                        addMinX = imagePartX2[0] if imagePartX2.shape != 0 else 0
                        addMaxX = imagePartX2[-1] if imagePartX2.shape != 0 else 0
                        if((addMaxX - addMinX) >= minWidthObject): 
                            rectanglesList.append((widthList[rectangleNumber - 1][0]+addMinX,
                                                    imagePartY[yStart],
                                                    widthList[rectangleNumber - 1][0]+addMaxX,
                                                    imagePartY[yEnd]) )
                        yStart = numberYRawMatrix
                numberYRawMatrix += 1  
        i += 1
    if (len(rectanglesList) > 0):
        return (rectanglesList)
    else:
        return ([0])
        
        
        
def drawRectangle(frame, rect):
    countVar = 0
    while (len(rect) > countVar):
        cv2.rectangle(frame, (rect[countVar][0], rect[countVar][1]), (rect[countVar][2], rect[countVar][3]), (0, 0, 255), 5)
        countVar += 1

def backgroundDifferenceProcess(input_path = '0'):
    if input_path == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    
    minWidthObject = 10
    distanceAlongX = 60
    minHeightObject = 30
    frameStep = 8# 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_ROI = (0, 0, 255) 
    write_all_val = 1
    i_webcam = 0
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100)

    #--------------------------------------- Video ---------------------------------------
    print('Video')
    save_path = './out.mp4'
    
    while(cap.isOpened() == False): 
        _, _ = cap.read()
    
    exitKey, firstFrame = cap.read()
        
    frame_shape = firstFrame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (frame_shape[1], frame_shape[0]))
    
    # must be turn on, if firstly use CNN fo classification
    # + in that case we need rect = [[],[],..]. So by default can set up [(0,0,0,0)]
    object_detected = False
        
    while(cap.isOpened()):
        # Пропускаем frameStep кадров, чтобы какой-то объект имел возможность заметно переместиться
        for i in range(frameStep):
            exitKey, secondFrame = cap.read()
            if(write_all_val == 1 or object_detected == True):
                cv2.putText(secondFrame, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f"__camera_{i_webcam}__",(30,30), font, .8,(0,0,0),2,cv2.LINE_AA)
                # ЖЕЛАТЕЛЬНО РИСОВАТЬ ОБЛАСТЬ В КОТ.ОБНАРУЖИЛИ ПЕРЕМЕЩЕНИЕ ?
                #ROI_draw(secondFrame, color_ROI)
                
                if object_detected:
                    drawRectangle(secondFrame, rect)
                
                out.write(secondFrame)
                cv2.imshow(f'camera: {i_webcam}',secondFrame)
        exitKey, secondFrame = cap.read()
        cv2.putText(secondFrame, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f"__camera_{i_webcam}__",(30,30), font, .8,(0,0,0),2,cv2.LINE_AA)
        object_detected = False
        if exitKey == False:
            break
                
        rect = calculateFrameDifference(firstFrame, secondFrame,fgbg, minWidthObject,
                                        distanceAlongX, minHeightObject)
        # Уверен, что должны подавать 2 кадра имеющие сдвиг в franeStep кадров? 
        # Надо быть уверенным, прежде чем отправлять
        firstFrame = secondFrame.copy()

        if(rect != [0]):
            drawRectangle(secondFrame, rect)
            with open('time_detection.txt', 'a') as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f'__camera_{i_webcam}\n')
            object_detected = True
        
        cv2.imshow(f'camera: {i_webcam}',secondFrame)
        out.write(secondFrame)
        
        # Get memory usage in Mb
        print('Memory usage: ',psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = (load1/os.cpu_count()) * 100
        print("The CPU usage is : ", cpu_usage)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    
if __name__ == "__main__":
    args = sys.argv[1]
    lock = True
    backgroundDifferenceProcess(args)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
