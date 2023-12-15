import numpy as np

import cv2
import matplotlib.pyplot as plt

def simulate_squares_1120(radius=50):
    image_border = np.zeros((200, 2100), dtype=np.float32)
    center_coordinates1 = (100, 100)
    center_coordinates2 = (300, 100)
    center_coordinates3 = (500, 100)
    center_coordinates4 = (700, 100)
    center_coordinates5 = (900, 100)
    center_coordinates6 = (1100, 100)
    center_coordinates7 = (1300, 100)
    center_coordinates8 = (1500, 100)
    center_coordinates9 = (1700, 100)
    center_coordinates10 = (1900, 100)

    radius = radius

    color_border = (128, 0, 0)
    color_hole = (255, 0, 0)
    
    thickness_border = -1
    thickness_hole = -1

    # border 3
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius+11), center_coordinates1[1]-(radius+11)), 
                  (center_coordinates1[0] + (radius+11), center_coordinates1[1]+ (radius+11)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius+11), center_coordinates1[1]-(radius+11)), 
                  (center_coordinates1[0] + (radius+11), center_coordinates1[1]+ (radius+11)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius), center_coordinates1[1]-(radius)), 
                  (center_coordinates1[0] + (radius), center_coordinates1[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius), center_coordinates1[1]-(radius)), 
                  (center_coordinates1[0] + (radius), center_coordinates1[1]+ (radius)), color_hole, thickness_hole)


    # # border 4
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius+12), center_coordinates2[1]-(radius+12)), 
                  (center_coordinates2[0] + (radius+12), center_coordinates2[1]+ (radius+12)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius+12), center_coordinates2[1]-(radius+12)), 
                  (center_coordinates2[0] + (radius+12), center_coordinates2[1]+ (radius+12)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius), center_coordinates2[1]-(radius)), 
                  (center_coordinates2[0] + (radius), center_coordinates2[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius), center_coordinates2[1]-(radius)), 
                  (center_coordinates2[0] + (radius), center_coordinates2[1]+ (radius)), color_hole, thickness_hole)

    # # border 5
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius+13), center_coordinates3[1]-(radius+13)), 
                  (center_coordinates3[0] + (radius+13), center_coordinates3[1]+ (radius+13)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius+13), center_coordinates3[1]-(radius+13)), 
                  (center_coordinates3[0] + (radius+13), center_coordinates3[1]+ (radius+13)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius), center_coordinates3[1]-(radius)), 
                  (center_coordinates3[0] + (radius), center_coordinates3[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius), center_coordinates3[1]-(radius)), 
                  (center_coordinates3[0] + (radius), center_coordinates3[1]+ (radius)), color_hole, thickness_hole)

    # # border 6
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius+14), center_coordinates4[1]-(radius+14)), 
                  (center_coordinates4[0] + (radius+14), center_coordinates4[1]+ (radius+14)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius+14), center_coordinates4[1]-(radius+14)), 
                  (center_coordinates4[0] + (radius+14), center_coordinates4[1]+ (radius+14)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius), center_coordinates4[1]-(radius)), 
                  (center_coordinates4[0] + (radius), center_coordinates4[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius), center_coordinates4[1]-(radius)), 
                  (center_coordinates4[0] + (radius), center_coordinates4[1]+ (radius)), color_hole, thickness_hole)

    # # border 15
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius+15), center_coordinates5[1]-(radius+15)), 
                  (center_coordinates5[0] + (radius+15), center_coordinates5[1]+ (radius+15)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius+15), center_coordinates5[1]-(radius+15)), 
                  (center_coordinates5[0] + (radius+15), center_coordinates5[1]+ (radius+15)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius), center_coordinates5[1]-(radius)), 
                  (center_coordinates5[0] + (radius), center_coordinates5[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius), center_coordinates5[1]-(radius)), 
                  (center_coordinates5[0] + (radius), center_coordinates5[1]+ (radius)), color_hole, thickness_hole)

    # # border 16
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius+16), center_coordinates6[1]-(radius+16)), 
                  (center_coordinates6[0] + (radius+16), center_coordinates6[1]+ (radius+16)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius+16), center_coordinates6[1]-(radius+16)), 
                  (center_coordinates6[0] + (radius+16), center_coordinates6[1]+ (radius+16)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius), center_coordinates6[1]-(radius)), 
                  (center_coordinates6[0] + (radius), center_coordinates6[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius), center_coordinates6[1]-(radius)), 
                  (center_coordinates6[0] + (radius), center_coordinates6[1]+ (radius)), color_hole, thickness_hole)

    # # border 17
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius+17), center_coordinates7[1]-(radius+17)), 
                  (center_coordinates7[0] + (radius+17), center_coordinates7[1]+ (radius+17)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius+17), center_coordinates7[1]-(radius+17)), 
                  (center_coordinates7[0] + (radius+17), center_coordinates7[1]+ (radius+17)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius), center_coordinates7[1]-(radius)), 
                  (center_coordinates7[0] + (radius), center_coordinates7[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius), center_coordinates7[1]-(radius)), 
                  (center_coordinates7[0] + (radius), center_coordinates7[1]+ (radius)), color_hole, thickness_hole)

    # # border 18
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius+18), center_coordinates8[1]-(radius+18)), 
                  (center_coordinates8[0] + (radius+18), center_coordinates8[1]+ (radius+18)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius+18), center_coordinates8[1]-(radius+18)), 
                  (center_coordinates8[0] + (radius+18), center_coordinates8[1]+ (radius+18)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius), center_coordinates8[1]-(radius)), 
                  (center_coordinates8[0] + (radius), center_coordinates8[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius), center_coordinates8[1]-(radius)), 
                  (center_coordinates8[0] + (radius), center_coordinates8[1]+ (radius)), color_hole, thickness_hole)
    
    # # border 19
    image_border = cv2.rectangle(image_border, (center_coordinates9[0] - (radius+19), center_coordinates9[1]-(radius+19)), 
                  (center_coordinates9[0] + (radius+19), center_coordinates9[1]+ (radius+19)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates9[0] - (radius+19), center_coordinates9[1]-(radius+19)), 
                  (center_coordinates9[0] + (radius+19), center_coordinates9[1]+ (radius+19)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates9[0] - (radius), center_coordinates9[1]-(radius)), 
                  (center_coordinates9[0] + (radius), center_coordinates9[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates9[0] - (radius), center_coordinates9[1]-(radius)), 
                  (center_coordinates9[0] + (radius), center_coordinates9[1]+ (radius)), color_hole, thickness_hole)
    
    # # border 20
    image_border = cv2.rectangle(image_border, (center_coordinates10[0] - (radius+20), center_coordinates10[1]-(radius+20)), 
                  (center_coordinates10[0] + (radius+20), center_coordinates10[1]+ (radius+20)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates10[0] - (radius+20), center_coordinates10[1]-(radius+20)), 
                  (center_coordinates10[0] + (radius+20), center_coordinates10[1]+ (radius+20)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates10[0] - (radius), center_coordinates10[1]-(radius)), 
                  (center_coordinates10[0] + (radius), center_coordinates10[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates10[0] - (radius), center_coordinates10[1]-(radius)), 
                  (center_coordinates10[0] + (radius), center_coordinates10[1]+ (radius)), color_hole, thickness_hole)
    plt.imshow(image_border, cmap='gray')
    plt.show()

    # cv2.imwrite('./data/test_img_two.png', image_border)
    return image_border

def simulate_circles(radius=20):
    image_hole = np.zeros((200, 1250), dtype=np.float32)
    center_coordinates1 = (100, 100)
    center_coordinates2 = (250, 100)
    center_coordinates3 = (400, 100)
    center_coordinates4 = (550, 100)
    center_coordinates5 = (700, 100)
    center_coordinates6 = (850, 100)
    center_coordinates7 = (1000, 100)
    center_coordinates8 = (1150, 100)



    radius = radius

    color_border = (128, 0, 0)
    color_hole = (255, 0, 0)
    
    thickness_border = -1
    thickness_hole = -1

    # border 3
    image_border = cv2.circle(image_hole, center_coordinates1, radius+3, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates1, radius+3, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates1, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates1, radius, color_hole, thickness_hole)

    # border 4
    image_border = cv2.circle(image_hole, center_coordinates2, radius+4, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates2, radius+4, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates2, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates2, radius, color_hole, thickness_hole)

    # border 5
    image_border = cv2.circle(image_hole, center_coordinates3, radius+5, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates3, radius+5, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates3, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates3, radius, color_hole, thickness_hole)

    # border 6
    image_border = cv2.circle(image_hole, center_coordinates4, radius+6, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates4, radius+6, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates4, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates4, radius, color_hole, thickness_hole)

    # border 7
    image_border = cv2.circle(image_hole, center_coordinates5, radius+7, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates5, radius+7, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates5, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates5, radius, color_hole, thickness_hole)

    # border 8
    image_border = cv2.circle(image_hole, center_coordinates6, radius+8, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates6, radius+8, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates6, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates6, radius, color_hole, thickness_hole)

    # border 9
    image_border = cv2.circle(image_hole, center_coordinates7, radius+9, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates7, radius+9, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates7, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates7, radius, color_hole, thickness_hole)

    # border 10
    image_border = cv2.circle(image_hole, center_coordinates8, radius+10, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates8, radius+10, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates8, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates8, radius, color_hole, thickness_hole)

    plt.imshow(image_hole, cmap='gray')
    plt.show()

    # cv2.imwrite('./data/test_img_two.png', image_border)
    return image_border

def simulate_circles_1120(radius=20):
    image_border = np.zeros((200, 2100), dtype=np.float32)
    center_coordinates1 = (100, 100)
    center_coordinates2 = (300, 100)
    center_coordinates3 = (500, 100)
    center_coordinates4 = (700, 100)
    center_coordinates5 = (900, 100)
    center_coordinates6 = (1100, 100)
    center_coordinates7 = (1300, 100)
    center_coordinates8 = (1500, 100)
    center_coordinates9 = (1700, 100)
    center_coordinates10 = (1900, 100)


    radius = radius

    color_border = (128, 0, 0)
    color_hole = (255, 0, 0)
    
    thickness_border = -1
    thickness_hole = -1

    # border 11
    image_border = cv2.circle(image_border, center_coordinates1, radius+11, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates1, radius+11, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates1, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates1, radius, color_hole, thickness_hole)

    # border 12
    image_border = cv2.circle(image_border, center_coordinates2, radius+12, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates2, radius+12, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates2, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates2, radius, color_hole, thickness_hole)

    # border 13
    image_border = cv2.circle(image_border, center_coordinates3, radius+13, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates3, radius+13, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates3, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates3, radius, color_hole, thickness_hole)

    # border 14
    image_border = cv2.circle(image_border, center_coordinates4, radius+14, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates4, radius+14, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates4, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates4, radius, color_hole, thickness_hole)

    # border 15
    image_border = cv2.circle(image_border, center_coordinates5, radius+15, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates5, radius+15, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates5, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates5, radius, color_hole, thickness_hole)

    # border 16
    image_border = cv2.circle(image_border, center_coordinates6, radius+16, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates6, radius+16, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates6, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates6, radius, color_hole, thickness_hole)

    # border 17
    image_border = cv2.circle(image_border, center_coordinates7, radius+17, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates7, radius+17, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates7, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates7, radius, color_hole, thickness_hole)

    # border 18
    image_border = cv2.circle(image_border, center_coordinates8, radius+18, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates8, radius+18, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates8, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates8, radius, color_hole, thickness_hole)

    image_border = cv2.circle(image_border, center_coordinates9, radius+19, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates9, radius+19, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates9, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates9, radius, color_hole, thickness_hole)

    image_border = cv2.circle(image_border, center_coordinates10, radius+20, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates10, radius+20, color_border, thickness_border)
    image_border = cv2.circle(image_border, center_coordinates10, radius, color_hole,2)
    image_border = cv2.circle(image_border, center_coordinates10, radius, color_hole, thickness_hole)

    plt.imshow(image_border, cmap='gray')
    plt.show()

    # cv2.imwrite('./data/test_img_two.png', image_border)
    return image_border

def simulate_squares(radius=50):
    image_border = np.zeros((200, 1250), dtype=np.float32)
    center_coordinates1 = (100, 100)
    center_coordinates2 = (250, 100)
    center_coordinates3 = (400, 100)
    center_coordinates4 = (550, 100)
    center_coordinates5 = (700, 100)
    center_coordinates6 = (850, 100)
    center_coordinates7 = (1000, 100)
    center_coordinates8 = (1150, 100)



    radius = radius

    color_border = (128, 0, 0)
    color_hole = (255, 0, 0)
    
    thickness_border = -1
    thickness_hole = -1

    # border 3
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius+3), center_coordinates1[1]-(radius+3)), 
                  (center_coordinates1[0] + (radius+3), center_coordinates1[1]+ (radius+3)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius+3), center_coordinates1[1]-(radius+3)), 
                  (center_coordinates1[0] + (radius+3), center_coordinates1[1]+ (radius+3)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius), center_coordinates1[1]-(radius)), 
                  (center_coordinates1[0] + (radius), center_coordinates1[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates1[0] - (radius), center_coordinates1[1]-(radius)), 
                  (center_coordinates1[0] + (radius), center_coordinates1[1]+ (radius)), color_hole, thickness_hole)


    # # border 4
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius+4), center_coordinates2[1]-(radius+4)), 
                  (center_coordinates2[0] + (radius+4), center_coordinates2[1]+ (radius+4)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius+4), center_coordinates2[1]-(radius+4)), 
                  (center_coordinates2[0] + (radius+4), center_coordinates2[1]+ (radius+4)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius), center_coordinates2[1]-(radius)), 
                  (center_coordinates2[0] + (radius), center_coordinates2[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates2[0] - (radius), center_coordinates2[1]-(radius)), 
                  (center_coordinates2[0] + (radius), center_coordinates2[1]+ (radius)), color_hole, thickness_hole)

    # # border 5
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius+5), center_coordinates3[1]-(radius+5)), 
                  (center_coordinates3[0] + (radius+5), center_coordinates3[1]+ (radius+5)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius+5), center_coordinates3[1]-(radius+5)), 
                  (center_coordinates3[0] + (radius+5), center_coordinates3[1]+ (radius+5)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius), center_coordinates3[1]-(radius)), 
                  (center_coordinates3[0] + (radius), center_coordinates3[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates3[0] - (radius), center_coordinates3[1]-(radius)), 
                  (center_coordinates3[0] + (radius), center_coordinates3[1]+ (radius)), color_hole, thickness_hole)

    # # border 6
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius+6), center_coordinates4[1]-(radius+6)), 
                  (center_coordinates4[0] + (radius+6), center_coordinates4[1]+ (radius+6)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius+6), center_coordinates4[1]-(radius+6)), 
                  (center_coordinates4[0] + (radius+6), center_coordinates4[1]+ (radius+6)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius), center_coordinates4[1]-(radius)), 
                  (center_coordinates4[0] + (radius), center_coordinates4[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates4[0] - (radius), center_coordinates4[1]-(radius)), 
                  (center_coordinates4[0] + (radius), center_coordinates4[1]+ (radius)), color_hole, thickness_hole)

    # # border 7
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius+7), center_coordinates5[1]-(radius+7)), 
                  (center_coordinates5[0] + (radius+7), center_coordinates5[1]+ (radius+7)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius+7), center_coordinates5[1]-(radius+7)), 
                  (center_coordinates5[0] + (radius+7), center_coordinates5[1]+ (radius+7)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius), center_coordinates5[1]-(radius)), 
                  (center_coordinates5[0] + (radius), center_coordinates5[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates5[0] - (radius), center_coordinates5[1]-(radius)), 
                  (center_coordinates5[0] + (radius), center_coordinates5[1]+ (radius)), color_hole, thickness_hole)

    # # border 8
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius+8), center_coordinates6[1]-(radius+8)), 
                  (center_coordinates6[0] + (radius+8), center_coordinates6[1]+ (radius+8)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius+8), center_coordinates6[1]-(radius+8)), 
                  (center_coordinates6[0] + (radius+8), center_coordinates6[1]+ (radius+8)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius), center_coordinates6[1]-(radius)), 
                  (center_coordinates6[0] + (radius), center_coordinates6[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates6[0] - (radius), center_coordinates6[1]-(radius)), 
                  (center_coordinates6[0] + (radius), center_coordinates6[1]+ (radius)), color_hole, thickness_hole)

    # # border 9
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius+9), center_coordinates7[1]-(radius+9)), 
                  (center_coordinates7[0] + (radius+9), center_coordinates7[1]+ (radius+9)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius+9), center_coordinates7[1]-(radius+9)), 
                  (center_coordinates7[0] + (radius+9), center_coordinates7[1]+ (radius+9)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius), center_coordinates7[1]-(radius)), 
                  (center_coordinates7[0] + (radius), center_coordinates7[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates7[0] - (radius), center_coordinates7[1]-(radius)), 
                  (center_coordinates7[0] + (radius), center_coordinates7[1]+ (radius)), color_hole, thickness_hole)

    # # border 10
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius+10), center_coordinates8[1]-(radius+10)), 
                  (center_coordinates8[0] + (radius+10), center_coordinates8[1]+ (radius+10)), color_border, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius+10), center_coordinates8[1]-(radius+10)), 
                  (center_coordinates8[0] + (radius+10), center_coordinates8[1]+ (radius+10)), color_border, thickness_border)
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius), center_coordinates8[1]-(radius)), 
                  (center_coordinates8[0] + (radius), center_coordinates8[1]+ (radius)), color_hole, 20) 
    image_border = cv2.rectangle(image_border, (center_coordinates8[0] - (radius), center_coordinates8[1]-(radius)), 
                  (center_coordinates8[0] + (radius), center_coordinates8[1]+ (radius)), color_hole, thickness_hole)
    plt.imshow(image_border, cmap='gray')
    plt.show()

    # cv2.imwrite('./data/test_img_two.png', image_border)
    return image_border


# img_circles = simulate_circles()
# cv2.imwrite('difr_circles.png', np.clip(img_circles, 0, 255))



def simulate_circle(w):
    image_hole = np.zeros((200, 200), dtype=np.float32)
    center_coordinates1 = (100, 100)
    # center_coordinates2 = (110, 60)
    # center_coordinates3 = (200, 60)
    # center_coordinates4 = (300, 60)
    # center_coordinates5 = (400, 60)
    # center_coordinates6 = (500, 60)
    # center_coordinates7 = (600, 60)
    # center_coordinates8 = (700, 60)



    radius = 50

    color_border = (128, 0, 0)
    color_hole = (255, 0, 0)
    
    thickness_border = -1
    thickness_hole = -1

    # border 3
    image_border = cv2.circle(image_hole, center_coordinates1, radius+w, color_border, 2)
    image_border = cv2.circle(image_border, center_coordinates1, radius+w, color_border, thickness_border)
    image_hole = cv2.circle(image_border, center_coordinates1, radius, color_hole,2)
    image_hole = cv2.circle(image_hole, center_coordinates1, radius, color_hole, thickness_hole)

    # # border 4
    # image_border = cv2.circle(image_hole, center_coordinates2, radius+4, color_border, 2)
    # image_border = cv2.circle(image_border, center_coordinates2, radius+4, color_border, thickness_border)
    # image_hole = cv2.circle(image_border, center_coordinates2, radius, color_hole,2)
    # image_hole = cv2.circle(image_hole, center_coordinates2, radius, color_hole, thickness_hole)

    # # border 5
    # image_border = cv2.circle(image_hole, center_coordinates3, radius+5, color_border, 2)
    # image_border = cv2.circle(image_border, center_coordinates3, radius+5, color_border, thickness_border)
    # image_hole = cv2.circle(image_border, center_coordinates3, radius, color_hole,2)
    # image_hole = cv2.circle(image_hole, center_coordinates3, radius, color_hole, thickness_hole)

    # # border 6
    # image_border = cv2.circle(image_hole, center_coordinates4, radius+6, color_border, 2)
    # image_border = cv2.circle(image_border, center_coordinates4, radius+6, color_border, thickness_border)
    # image_hole = cv2.circle(image_border, center_coordinates4, radius, color_hole,2)
    # image_hole = cv2.circle(image_hole, center_coordinates4, radius, color_hole, thickness_hole)

    # # border 7
    # image_border = cv2.circle(image_hole, center_coordinates5, radius+7, color_border, 2)
    # image_border = cv2.circle(image_border, center_coordinates5, radius+7, color_border, thickness_border)
    # image_hole = cv2.circle(image_border, center_coordinates5, radius, color_hole,2)
    # image_hole = cv2.circle(image_hole, center_coordinates5, radius, color_hole, thickness_hole)

    # # border 8
    # image_border = cv2.circle(image_hole, center_coordinates6, radius+8, color_border, 2)
    # image_border = cv2.circle(image_border, center_coordinates6, radius+8, color_border, thickness_border)
    # image_hole = cv2.circle(image_border, center_coordinates6, radius, color_hole,2)
    # image_hole = cv2.circle(image_hole, center_coordinates6, radius, color_hole, thickness_hole)

    # # border 9
    # image_border = cv2.circle(image_hole, center_coordinates7, radius+9, color_border, 2)
    # image_border = cv2.circle(image_border, center_coordinates7, radius+9, color_border, thickness_border)
    # image_hole = cv2.circle(image_border, center_coordinates7, radius, color_hole,2)
    # image_hole = cv2.circle(image_hole, center_coordinates7, radius, color_hole, thickness_hole)

    # # border 10
    # image_border = cv2.circle(image_hole, center_coordinates8, radius+10, color_border, 2)
    # image_border = cv2.circle(image_border, center_coordinates8, radius+10, color_border, thickness_border)
    # image_hole = cv2.circle(image_border, center_coordinates8, radius, color_hole,2)
    # image_hole = cv2.circle(image_hole, center_coordinates8, radius, color_hole, thickness_hole)

    # plt.imshow(image_hole, cmap='gray')
    # plt.show()

    # cv2.imwrite('./data/test_img_two.png', image_border)
    return image_border
