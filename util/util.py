import cv2
import numpy as np
import sys


def readkp(filename):
    def is_float(n):
        try:
            float(n)
            return True
        except:
            return False
    kp_des_temp = []
    counter = 0
    num_kp = 0
    des_len = 0
    with open(filename, "r") as file:
        for line in file.readlines():
            counter += 1

            if (counter == 2):
                num_kp = int(line)

            if (counter == 3):
                des_len = int(line)

            if (counter > 3):
                kp_des_temp.append( [float(n) for n in line.split(',') if is_float(n)] )

    kp_des_temp = np.array(kp_des_temp, dtype=np.float32)

    assert (kp_des_temp.shape[0] == num_kp)
    assert (kp_des_temp.shape[1] == (5+des_len) )

    kpts = kp_des_temp[:,0:5]
    des = kp_des_temp[:,5:]

    return kpts, des, num_kp


def KP_opencv(np_kp_matrix):
    radTodeg = 180./np.pi
    np_kp_matrix = np.array(np_kp_matrix, dtype=np.float32)
    keypoints_opencv = []
    for i in range(np_kp_matrix.shape[0]):
        keypoints_opencv.append(cv2.KeyPoint(x=np_kp_matrix[i,0],
            y=np_kp_matrix[i,1],
            _size=np_kp_matrix[i,2],
            _response=np_kp_matrix[i,3],
            _angle=np_kp_matrix[i,4]* radTodeg,
            _octave=0,
            _class_id=0))
    return keypoints_opencv 


def drawKeyPts(im, keyp, color, th, im_save):
    for kp_i in keyp:
        center = (int(np.round(kp_i.pt[0])), int(np.round(kp_i.pt[1])))
        radius = int(np.round(kp_i.size*2.))
        cv2.circle(im, center, radius, color, thickness=th)
        
        orient = (int(np.round(np.cos(kp_i.angle)*radius)), int(np.round(np.sin(kp_i.angle)*radius)))
        cv2.line(im, center, (center[0]+orient[0], center[1]+orient[1]), color, 1)
    
    cv2.imwrite(im_save, im) 



def Matcher_NNDR(IMGs, KPTS_DES):
    img1 = cv2.imread(IMGs[0])  
    img2 = cv2.imread(IMGs[1]) 

    kp1, des1 = KPTS_DES[0]
    kp1 = KP_opencv(kp1)

    kp2, des2 = KPTS_DES[1]
    kp2 = KP_opencv(kp2)
    
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)

    kp_num = np.minimum(KPTS_DES[0][0].shape[0], KPTS_DES[0][1].shape[0])
    print("[+] #Detected keypoints:  %d -> #inliers (by cv2.RANSAC)/#matches: %d / %d"%(kp_num, sum(matchesMask), len(matches)))

    drawKeyPts(img1, kp1, (0,255,0), 2, "kp1.png")
    drawKeyPts(img2, kp2, (0,0,255), 2, "kp2.png")

    cv2.imwrite("matches_NNDR.png", img3) 
