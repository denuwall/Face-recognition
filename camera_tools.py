import cv2 as cv

def returnCameraIndexes():
    """Получаем индексы возможных камер

    Returns:
        list: Массив индексов камер
    """
    # проверяем первые 10 индексов
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

def openCamera(id):
    capture = cv.VideoCapture(id)
    capture.set(cv.CAP_PROP_BUFFERSIZE, 2)

    # FPS = 1/X
    # X = desired FPS
    FPS = 1/30
    FPS_MS = int(FPS * 1000)

    return capture