import os,sys
import cv2 as cv
import dlib
from imutils import face_utils
from scipy.spatial import distance

# queueモジュールインポート
import queue
# dequeモジュールインポート
from collections import deque



# video or webcamera select
# video:"videoname"
# webcamera:0 or 1 #内臓カメラ，一台目カメラ：0，外部カメラ，二台目以降カメラ：1
cap_name=sys.argv[1]

if cap_name == '0' :
     cap_name=int(cap_name)
     flag = 0
else : 
     pre_cap_name=str(cap_name)
     cap_name=os.path.join("./inputs",pre_cap_name)
     flag = 1

cap = cv.VideoCapture(cap_name)


cap_fps=cap.get(cv.CAP_PROP_FPS)
print(cap_fps)
cap_w=cap.get(cv.CAP_PROP_FRAME_WIDTH)
print(cap_w)
cap_h=cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(cap_h)
cap_c=cap.get(cv.CAP_PROP_FRAME_COUNT)
print(cap_c)
"""
cap_name=sys.argv[1]

if cap_name == '0' :
     cap_name=int(cap_name)
else : 
     cap_name=str(cap_name)

cap = cv2.VideoCapture(cap_name)
"""


fourcc = cv.VideoWriter_fourcc(*'XVID')#OS:windows
if flag == 1 : 
    #output_name=os.path.join("./results",pre_cap_name.split('.')[0]+'_output_'+str(numUpSampling)+'.'+pre_cap_name.split('.')[1])
    output_name=os.path.join("./results",pre_cap_name.split('.')[0]+'_output'+'.'+pre_cap_name.split('.')[1])
elif flag == 0 : 
    output_name=os.path.join("./results","web_camera_output.mp4")
    
    output_name0=os.path.join("./results","web_camera_ori.mp4")
    writer_web_camera = cv.VideoWriter(output_name0 , fourcc, float(cap_fps), (int(cap_w), int(cap_h)) )
    
    output_name1=os.path.join("./results","web_camera_rec.mp4")
    writer_web_camera_rec = cv.VideoWriter(output_name1 , fourcc, float(cap_fps), (int(cap_w), int(cap_h)) )

writer = cv.VideoWriter(output_name, fourcc, float(cap_fps), (int(cap_w), int(cap_h)) )


face_cascade = cv.CascadeClassifier(os.path.join(os.getcwd(),"models/haarcascade_frontalface_alt2.xml"))
face_parts_detector = dlib.shape_predictor(os.path.join(os.getcwd(),"models/shape_predictor_68_face_landmarks.dat"))

eye_cascade = cv.CascadeClassifier(os.path.join(os.getcwd(),"models/haarcascade_eye_tree_eyeglasses.xml"))

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def eye_marker(face_mat, position):
        #要素とインデックス（カウンタ）を同時に取得する
        #l=['Alice', 'Bob', 'Charlie']
        #for i, name in enumerate(l) -> # 0 Alice # 1 Bob # 2 Charlie
    for i, ((x, y)) in enumerate(position):
        cv.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv.putText(face_mat, str(i), (x + 2, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

count=0

# listでキューを作成
q_maxsize=10
q_ear_list=[]
#q_ear = queue.Queue(maxsize=q_maxsize)

while True:
    tick = cv.getTickCount()

    ret, rgb = cap.read()
    rgb_h, rgb_w, rgb_c = rgb.shape[:3]
    ori = rgb.copy()
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    #gray = rgb
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    #eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(50, 50))
    """
    detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flag, Size minSize, Size maxSize)
    image:CV_8U型の行列。ここに格納されていいる画像中から物体が検出されます。
    objects:矩形を要素とするベクトル。それぞれの矩形には、検出した物体を含みます。
    scaleFactor:画像スケールにおける縮小量を表します。
    minNeighbors:物体候補となる矩形は、最低でもこの数だけの近傍矩形を含む必要があります。
    flags:このパラメータは、新しいカスケードでは使用されません。古いカスケードに対しては、cvHaarDetectObjects関数の場合と同じ意味を持ちます。
    minSize:物体が取り得る最小サイズ。これよりも小さい物体は無視されます。
    maxSize:物体が取り得る最大サイズ
    """

    #if len(faces) == 1 and (0<=len(eyes) and len(eyes)<=2):
    if len(faces) == 1:
        print("faces:")
        print(faces)
        x, y, w, h = faces[0, :]#顔がひとつの場合のみ想定しているので0，EX．faces:[[269 129 131 131]]
        cv.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        eyes_gray = gray[y : y + int(h/2), x : x + w]#効率化のためfaces領域の上半分を探索
        eyes = eye_cascade.detectMultiScale(eyes_gray, scaleFactor=1.11, minNeighbors=3, minSize=(8, 8))
        if 0< len(eyes) and len(eyes) <=2:
            print("eye:")
            print(eyes)
            #eyes_gray = gray[y : y + int(h/2), x : x + w]#効率化のためfaces領域の上半分を探索
            #eyes = eye_cascade.detectMultiScale(eyes_gray, scaleFactor=1.11, minNeighbors=3, minSize=(8, 8))
            #eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(8, 8))
            eye_scale=10
            #for ex, ey, ew, eh in eyes:
                 #left_x=x + ex - eye_scale
                 #cv.rectangle(rgb, (x + ex - eye_scale, y + ey - eye_scale), (x + ex + ew + eye_scale, y + ey + eh + eye_scale), (255, 255, 0), 1)############目の矩形領域描画##3/31
                 #cv.rectangle(gray, (0,0), (0,0), (255, 255, 0), 0)
                 #cv2.rectangle(rgb, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 0), 1)

        face_gray = gray[y :(y + h), x :(x + w)]#切り取り
        #scale = 480 / h
        resize_size = 480
        scale = resize_size / h
        #scale = scale*2.0
        face_gray_resized = cv.resize(face_gray, dsize=None, fx=scale, fy=scale)#デフォルト：INTER_LINEAR(バイリニア補間)
        #face_gray_resized =

        face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
        #dlib.rectangle(left: int, top: int, right: int, bottom: int)
        face_parts = face_parts_detector(face_gray_resized, face)
        face_parts = face_utils.shape_to_np(face_parts)#shapeをndarrayに変換


        #re_img=cv.circle(img, (x, y), 1, (255, 255, 255), -1,shift=0)
        #face_parts_detector = dlib.shape_predictor(os.path.join(os.getcwd(),"models/shape_predictor_68_face_landmarks.dat"))
        #dlib.shape_predictor(model_path, image, box(bounding box):検出したい領域の座標)
        #基本bounding box内部にパーツを検出するが，構成パーツの一部はbounding box外部にも検出される．
        face_all_box=dlib.rectangle(x, y, x+w, y+h)
        #face_all_box=cv.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_all = face_parts_detector(rgb, face_all_box)
        face_all = face_utils.shape_to_np(face_all)#shapeをndarrayに変換
        #for ((x, y)) in face_all:
        for i, ((x, y)) in enumerate(face_all):
            if 36<=i and i < 48:
                rgb=cv.circle(rgb, (x, y), 1, (255, 255, 0), -1,shift=0)
            else:
                rgb=cv.circle(rgb, (x, y), 1, (255, 255, 255), -1,shift=0)


        left_eye = face_parts[42:48]
        eye_marker(face_gray_resized, left_eye)
        left_eye_ear = calc_ear(left_eye)
        cv.putText(rgb, "L_eye ear:{} ".format(left_eye_ear), (40, 440), cv.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 0), 1, cv.LINE_AA)
        #cv.putText(face_gray_resized, "LEFT eye EAR:{} ".format(left_eye_ear), (10, 100), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv.LINE_AA)


        right_eye = face_parts[36:42]
        eye_marker(face_gray_resized, right_eye)
        right_eye_ear = calc_ear(right_eye)
        cv.putText(rgb, "R_eye ear:{} ".format(round(right_eye_ear, 3)), (180, 440), cv.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 0), 1, cv.LINE_AA)
        #cv.putText(face_gray_resized, "RIGHT eye EAR:{} ".format(round(right_eye_ear, 3)), (10, 120), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv.LINE_AA)
        
        # ear
        ear=left_eye_ear + right_eye_ear
        cv.putText(rgb, "Eyes ear:{} ".format(round(ear, 3)), (320, 440), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
        
        # キューを作成
        #q_maxsize=10
        #q_ear_list=[]
        #q_ear = queue.Queue(maxsize=q_maxsize)
        
        ############listキューに関する操作#####################
        
        # 30fps(1秒)で、瞬きは0.1~0.15秒程の場合
        # q_maxsizeは最低30/10=3、余裕を持つため、5~10程が適切
        q_maxsize=10
        
        if len(q_ear_list)  == q_maxsize:q_ear_list.pop(0)
        
        q_ear_list.append(ear)
        print(q_ear_list)
        q_ear_ave=sum(q_ear_list)/len(q_ear_list)
        ############listキューに関する操作#####################
        
        #ear_閾値
        ear_threshold=0.45
        cv.putText(rgb, "Threshold:{} ".format(round(ear_threshold, 3)), (460, 440), cv.FONT_HERSHEY_PLAIN, 1, (255, 50, 225), 1, cv.LINE_AA)

        if (q_ear_ave) < ear_threshold:#眠気検知は片目0.2~0.25とされている #####################################################################################################
            #cv.putText(rgb,"Sleepy eyes. Wake up!",(10,180), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, 1)
            #cv.putText(face_gray_resized,"Sleepy eyes. Wake up!",(10,280), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, 1)
            
            cv.putText(rgb,"Drowsiness Detection. ",(40, 350), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, 1)
            cv.putText(rgb,"Wake up! Be careful!",(40, 400), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, 1)
            #cv.putText(face_gray_resized,"Drowsiness Detection. Be careful!",(0,0), cv.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3, 1)
            cv.rectangle(rgb,pt1=(0, 0),pt2=(rgb_w, rgb_h),color=(0, 0, 255),thickness=50,lineType=cv.LINE_4,shift=0)

        cv.imshow('frame_resize', face_gray_resized)###

    fps = cv.getTickFrequency() / (cv.getTickCount() - tick)
    #cv.putText(rgb, "FPS:{} ".format(int(fps)), (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)##############################3/31
    #cv.putText(face_gray_resized, "FPS:{} ".format(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv.imshow('frame', rgb)###
    if flag==0:
        print("flag=0,writer")
        writer_web_camera.write(ori)
        #writer_web_camera_rec.write(face_gray_resized)
    writer.write(rgb)
    if cv.waitKey(1) == 27:
        break  # esc to quit


cap.release()
cv.destroyAllWindows()