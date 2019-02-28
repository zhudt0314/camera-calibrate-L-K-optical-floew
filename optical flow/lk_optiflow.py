import numpy as np
import cv2

cap = cv2.VideoCapture('C:/Users/Dell/Desktop/cvhw-zdt/optical flow/test.mp4')

#角点检测参数
feature_params = dict( maxCorners = 50,
                       qualityLevel=0.3,
                       minDistance = 7,
                       blockSize = 7 )

#LK光流参数，maxLevel为图像金字塔层数
lk_params = dict( winSize = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
color = np.random.randint(0,255,(100,3)) #随机颜色，用于标示特征点

#在第一帧标示角点
ret, f_frame = cap.read()
# cv2.imshow("result",old_frame)
# cv2.waitKey()
f_gray = cv2.cvtColor(f_frame,cv2.COLOR_BGR2GRAY)
# cv2.imshow("result",old_gray)
# cv2.waitKey()
# 获取图像中角点，输出结果为角点的位置数组
p0=cv2.goodFeaturesToTrack(f_gray,mask=None,**feature_params)

#画布，用于画轨迹
mask = np.zeros_like(f_frame) #返回等大小零矩阵

while(1):
    ret, n_frame = cap.read()
    n_gray = cv2.cvtColor(n_frame,cv2.COLOR_BGR2GRAY)
    #获取特征点新位置
    n_cor, status, err = cv2.calcOpticalFlowPyrLK(f_gray, n_gray, p0, None, **lk_params)
    #特征点选择
    good_new = n_cor[status==1]
    good_old = p0[status==1]

    #绘制特征点轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
        frame = cv2.circle(n_frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(100)
    if k == 27:
        break
    #更新上一帧的图像和目标点
    f_gray = n_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

    
 

