import numpy as np
import face_recognition as fr
import cv2 as cv
import os
from datetime import datetime as dt
from playsound import playsound
from PIL import ImageFont, ImageDraw, Image
from random import choices
from string import ascii_letters

# для хранения лиц разработан программый код который использует в качестве обработки массив данных
path = "KnowFaces"  # путь к папке с фото
images = []  # массив для фото
massNames = []  # массив для имен
myList = os.listdir(path)  # получаем список фото

for cls in myList:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)  # добавляем фотографию в массив
    massNames.append(os.path.splitext(cls)[0])  # добавляем имя в массив

print(massNames)


def findEncodings(images):  # добавляем фото в функцию (отвечает за декодинг фото)
    encodeList = []  # инициализируем лист в который в последствии помещаем декодированные фото
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # обрабатываем фото переводя в удобный формат
        encode = fr.face_encodings(img)[0]  # переменная хранит декодинг данные
        encodeList.append(encode)  # добавляем в массив обработанное фото
    return encodeList  # возвращаем массив

# для удобства хранения
def markAttendance(name):  # выводим данные в csv файл
    with open("face.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = dt.now()
            dtString = now.strftime("%d.%m.%y_%H:%M:%S")
            f.writelines(f'\n{name}, {dtString}')


encodeListKnown = findEncodings(images)  # отвечает за обработанные фото
print("Декодирование закончено")

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()  # переменные отвечают за 1 кадр из видео
    imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)  # принимает подготовленный файл для обработки
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS)  # поиск всех лиц
    encodeCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):  # цикл для распознования
        matches = fr.compare_faces(encodeListKnown, encodeFace)  # отвечает за уже известное лицо
        faceDis = fr.face_distance(encodeListKnown, encodeFace)  # вероятность
        print(faceDis)
        matchIndex = np.argmin(faceDis)  # принимаем индексы всех имен

        if matches[matchIndex]:  # проверика на имеющиеся в базе лица
            name = massNames[matchIndex]
            print(f"{name} доступ разрешен")
            y1, x2, y2, x1 = faceLoc  # рисуем рамку
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

        else:
            filename = 'KnowFaces/face.jpg'
            cv.imwrite(filename, img)
            print("Лицо сохранено")
            y1, x2, y2, x1 = faceLoc  # рисуем рамку
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, "Неопознан", (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            playsound('sound.mp3')

            break

    cv.imshow("WebCam", img)  # создаем окно
    cv.waitKey(1)
