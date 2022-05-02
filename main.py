import base64
from genericpath import isfile
from ntpath import join
import pathlib
import numpy as np
import face_recognition as fr
import cv2 as cv
import os
import camera_tools
import time
from datetime import datetime as dt
from playsound import playsound
from flask import Flask, render_template, Response

app = Flask(__name__)
cap = []
cameras = camera_tools.returnCameraIndexes()
for camera in cameras:
    cap.append(camera_tools.openCamera(int(camera)))

# для хранения лиц разработан программый код который использует в качестве обработки массив данных
path = "KnowFaces"  # путь к папке с фото
images = []  # массив для фото
massNames = []  # массив для имен
myList = os.listdir(path)  # получаем список фото
cameras = camera_tools.returnCameraIndexes()
for cls in myList:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)  # добавляем фотографию в массив
    massNames.append(os.path.splitext(cls)[0])  # добавляем имя в массив

print(massNames)


def findEncodings(images):  # добавляем фото в функцию (отвечает за декодинг фото)
    encodeList = []  # инициализируем лист в который в последствии помещаем декодированные фото
    for frame in images:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # обрабатываем фото переводя в удобный формат
        encode = fr.face_encodings(frame)[0]  # переменная хранит декодинг данные
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


def gen_frames(cam):
    process_this_frame = True
    while True:
        success, frame = cap[int(cam)].read()  # переменные отвечают за 1 кадр из видео
        if not success:
            break
        else:
            if process_this_frame:
                small_frame = cv.resize(frame, (0, 0), None, 0.25, 0.25)  # принимает подготовленный файл для обработки
                small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

                facesCurFrame = fr.face_locations(small_frame)  # поиск всех лиц
                encodeCurFrame = fr.face_encodings(small_frame, facesCurFrame)

                if (len(facesCurFrame) == 0):
                    ret, buffer = cv.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):  # цикл для распознования
                    matches = fr.compare_faces(encodeListKnown, encodeFace)  # отвечает за уже известное лицо
                    faceDis = fr.face_distance(encodeListKnown, encodeFace)  # вероятность
                    # print(faceDis)
                    matchIndex = np.argmin(faceDis)  # принимаем индексы всех имен

                    if matches[matchIndex]:  # проверика на имеющиеся в базе лица
                        name = massNames[matchIndex]
                        print(f"{name} доступ разрешен")
                        y1, x2, y2, x1 = faceLoc  # рисуем рамку
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        cv.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
                        cv.putText(frame, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        markAttendance(name)
                        ret, buffer = cv.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        filename = 'KnowFaces/face' + time.strftime + '.jpg'
                        cv.imwrite(filename, frame)
                        print("Лицо сохранено")
                        y1, x2, y2, x1 = faceLoc  # рисуем рамку
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
                        cv.putText(frame, "Неопознан", (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        playsound('sound.mp3')
                        buffer = cv.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                try:
                    frame = buffer.tobytes()
                    ret, buffer = cv.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass
            process_this_frame = not process_this_frame


@app.route('/')
def index():
    return render_template('index.html', cameras=cameras)

@app.route('/archive')
def archive():
    with open("face.csv", "r+") as f:
        archive = f.readlines()
        for i in range(0, len(archive)):
            if(i == 0):
                archive[i] = "<tr><td>Имя</td><td>Время</td></tr>"
                continue
            if(archive[i] == "\n"): 
                continue
            temp = archive[i].split()
            archive[i] = "<tr><td>" + temp[0].replace(",", "") + "</td><td>" + temp[1].replace("_", " - ") + "</td></tr>"
        return render_template('archive.html', archive=archive)

@app.route('/faces')
def faces():
    facesPath = str(pathlib.Path().resolve()) + "\\" + path
    onlyfiles = [f for f in os.listdir(facesPath) if isfile(join(facesPath, f))]
    for i in range(0, len(onlyfiles)):
        with open(facesPath + "\\" + onlyfiles[i], "rb") as image_file:
            onlyfiles[i] = str(base64.b64encode(image_file.read())).replace("b'", "").replace("'", "")
    return render_template('faces.html', files=onlyfiles, as_attachment=True)

@app.route('/video_feed/<cam>')
def video_feed_with_cam_select(cam):
    return Response(gen_frames(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
