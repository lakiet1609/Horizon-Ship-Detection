import cv2
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from src.logger import logging
from src.exception import CustomException

class DetectHorizon:
    def __init__(self, imgs, output_path):
        self.imgs = imgs
        self.output_path = output_path
        logging.info('Initialize horizon detection module ...')
    
    def filter_points(self, points):
        y_values = np.array([y for _, y in points])
        Q1 = np.percentile(y_values, 25)
        Q3 = np.percentile(y_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_points = [point for point in points if lower_bound <= point[1] <= upper_bound]
        return filtered_points
    
    def linear_regression_line(self, points, image_width, img):
        x = np.array([point[0] for point in points]).reshape(-1, 1)
        y = np.array([point[1] for point in points])

        model = LinearRegression()
        model.fit(x, y)

        x_start = np.array([[0]])
        x_end = np.array([[image_width]])
        y_start = model.predict(x_start)[0]
        y_end = model.predict(x_end)[0]

        start_point = (0, int(y_start))
        end_point = (image_width, int(y_end))
        color = (0, 255, 0)  
        thickness = 2  
        cv2.line(img, start_point, end_point, color, thickness)
        return start_point, end_point 

    def horizon_detection(self) -> list:
        try:
            list_points = []
            for img in self.imgs:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
                
                points = []
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if abs(y2 - y1) < 10: 
                            y = (y1 + y2) // 2
                            for x in range(min(x1, x2), max(x1, x2), 10): 
                                points.append((x, y)) 
                
                _, image_width = img.shape[:2]
                filtered_points = self.filter_points(points)
                start_point, end_point = self.linear_regression_line(filtered_points, image_width, img)
                list_points.append([start_point, end_point])
                
                cv2.imwrite(self.output_path, img)
            
            return list_points
        
        except Exception as e:
            raise CustomException(e,sys)