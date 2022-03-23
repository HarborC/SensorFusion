'''
Author: Jiagang Chen
Date: 2021-11-12 02:25:54
LastEditors: Jiagang Chen
LastEditTime: 2021-12-20 04:06:19
Description: ...
Reference: ...
'''
import numpy as np

def printT(T):
    T_list = [T[0,0], T[0,1], T[0,2], T[0,3], T[1,0], T[1,1], T[1,2], T[1,3], T[2,0], T[2,1], T[2,2], T[2,3], T[3,0], T[3,1], T[3,2], T[3,3]]
    print(T_list)

T_car_LeftLidar = np.array([[-0.516377, -0.702254, -0.490096, -0.334623],
                            [0.491997, -0.711704, 0.501414, 0.431973],
                            [-0.700923, 0.0177927, 0.713015, 1.94043],
                            [0.0, 0.0, 0.0, 1.0]], dtype=float)

T_car_RightLidar = np.array([[-0.514521, 0.701075, -0.493723, -0.333596],
                             [-0.492472, -0.712956, -0.499164, -0.373928],
                             [-0.701954, -0.0136853, 0.712091, 1.94377],
                             [0.0, 0.0, 0.0, 1.0]], dtype=float)

T_car_Imu = np.array([[1.0, 0.0, 0.0, -0.07],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.7],
                      [0.0, 0.0, 0.0, 1.0]], dtype=float)
T_Imu_car = np.linalg.inv(T_car_Imu)

T_car_LeftCam = np.array([[-0.00680499, -0.0153215, 0.99985, 1.64239],
                          [-0.999977, 0.000334627, -0.00680066, 0.247401],
                          [-0.000230383, -0.999883, -0.0153234, 1.58411],
                          [0.0, 0.0, 0.0, 1.0]], dtype=float)


T_Imu_LeftLidar = np.matmul(T_Imu_car, T_car_LeftLidar)
T_LeftLidar_Imu = np.linalg.inv(T_Imu_LeftLidar)
print("T_Imu_LeftLidar : ")
print(T_Imu_LeftLidar)
printT(T_Imu_LeftLidar)

T_Imu_RightLidar = np.matmul(T_Imu_car, T_car_RightLidar)
T_RightLidar_Imu = np.linalg.inv(T_Imu_RightLidar)
print("T_Imu_RightLidar : ")
print(T_Imu_RightLidar)
printT(T_Imu_RightLidar)

T_Imu_LeftCam = np.matmul(T_Imu_car, T_car_LeftCam)
print("T_Imu_LeftCam : ")
print(T_Imu_LeftCam)
printT(T_Imu_LeftCam)


T_I_LeftCam_O = np.array([[-0.00413,-0.01966,0.99980,1.73944],
                          [-0.99993,-0.01095,-0.00435,0.27803],
                          [0.01103,-0.99975,-0.01962,-0.08785],
                          [0.0, 0.0, 0.0, 1.0]], dtype=float)

T_I_RightCam_O = np.array([[-0.00768,-0.01509,0.99986,1.73376],
                          [-0.99988,-0.01305,-0.00788,-0.19706],
                          [0.01317,-0.99980,-0.01499,-0.08271],
                          [0.0, 0.0, 0.0, 1.0]], dtype=float)

print("T_Imu_RightCam")
T_LeftCam_RightCam_O = np.matmul(np.linalg.inv(T_I_LeftCam_O), T_I_RightCam_O)
T_Imu_RightCam = np.matmul(T_Imu_LeftCam, T_LeftCam_RightCam_O)
print(T_Imu_RightCam)
printT(T_Imu_RightCam)