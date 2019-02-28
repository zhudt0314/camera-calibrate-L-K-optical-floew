import cv2
import glob
import numpy as np
import os

def getCorner(image):
    #image = "../*.jpg"
    CHECKERBOARD = (8,6) #棋盘格角点数量
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) #找角点
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) 
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space,存储棋盘格角点的世界坐标和图像坐标对
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(image) #读取同一路径下图片
    for fname in images:
        img = cv2.imread(fname) #source image
        # print(_img_shape)
        # if _img_shape == None:
        #     _img_shape = img.shape[:2]
        # else:
        #     assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转灰度
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
    cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            #亚像素级角点
            objpoints.append(objp)
            corners_sub=cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            #在图像上显示角点
            # cv2.drawChessboardCorners(img, (8,6), corners_sub, ret)
            # cv2.imshow('findCorners',img)
            cv2.waitKey(1000)
    cv2.destroyAllWindows()
    img_shape = gray.shape[::-1]
    return objpoints, imgpoints, img_shape

if __name__ == "__main__":
    print(os.getcwd())
    obj_p, img_p, img_shape = getCorner("C:/Users/Dell/Desktop/cvhw-zdt/cam_calibrate/test/*.jpg") 
    '''
    传入所有图片各自角点的三维、二维坐标，相机标定。
    每张图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组。
    mtx，相机内参；dist，畸变系数；revcs，旋转矩阵；tvecs，平移矩阵。
    '''
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, img_shape,None,None)
    img = cv2.imread('C:/Users/Dell/Desktop/cvhw-zdt/cam_calibrate/test/14.jpg') #输入目标图像
    h,w = img.shape[:2]
    '''
    优化相机内参
    参数1表示保留所有像素点，同时可能引入黑色像素，
    设为0表示尽可能裁剪不想要的像素，0-1都可以取。
    '''
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx) #纠正畸变
    #输出纠正畸变以后的图片
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.namedWindow("result",cv2.WINDOW_NORMAL)
    cv2.imshow("result",dst)
    cv2.waitKey()
    #输出内参矩阵和畸变矩阵
    print ("newcameramtx:/n",newcameramtx)
    print ("dist:/n",dist)