import numpy as np
import cv2
import matplotlib.pyplot as  plt

class Figure():
    def __init__(self, cont):
        self.contour = cont.copy()
        self.border_width = 0
        self.border = cont.copy() # по дефолту сначала заполняется контуром дна


    def getContour(self):
        return self.contour
    
    def getBorderContour(self):
        return self.border
    
    def init_border_contour(self, img, object, width = None):
        """ Метод, ищутся все точки контура границы и пишутся в соответсвующее поле"""
        if width is None:
            width = object.border_width
        temp = np.zeros_like(img)                     # создание временной картинки
        cv2.drawContours(temp, [object.contour], 0, 1, 2*width) # рисование контура с первоначальным offset
        c, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # детектирование контура
        object.border = c[0].reshape(-1, 2).copy()

    
    def try_render(self, img, objects, object, i, width):
        """ Метод, в котором пытаешься отрендерить контур с границей и проверяешь, 
        не пересекается ли он с другими границами или объектами"""
        object.border_width = width
        self.init_border_contour(img, object)
        
        obj_num = None
        flag_render = True
        for k in range(len(objects)):
            set_cur_cont = set(map(tuple, object.border))

            if k != i:
                set_another_cont = set(map(tuple, objects[k].border))

                if set_cur_cont & set_another_cont:
                    obj_num = k
                    flag_render = False
                    break
        
        return obj_num, flag_render

    
    # def render(self, cont, width):
    #     """Метод для рендеринга контура с границей определенной толщины"""
    #     return
    #     # cv2.
    #     # cv2.drawContours(img, [contour], -1, 255, 0)


