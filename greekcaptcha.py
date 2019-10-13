import cv2
import cv_utils

img_rows, img_cols = 28, 28
LEXICON = "0123456789abcdefghijklmnopqrstuvwxyz"
num_classes = len(LEXICON)
input_shape = (img_rows, img_cols, 1)

def chop_image(image):
    if (image.shape[0] == 60 and image.shape[1] == 240):
        image = cv2.resize(image, (120, 30))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_border = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[0,0,0])
    binarized = cv2.threshold(gray_border, 50, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    contours, hierarchies = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index = 0 
    letters = []
    bounding_boxes = []
    for contour in contours:
        hr = hierarchies[0][index]
        x, y, w, h = cv2.boundingRect(contour)

        if (hr[3] == 0 and w < 100):
            bounding_boxes.append([x, y, w, h])    
        index = index +1
        
    bounding_boxes.sort(key=cv_utils.first_element)        
    bounding_boxes = cv_utils.merge_bounding_boxes(bounding_boxes)
    bounding_boxes = cv_utils.split_large_bounding_boxes(bounding_boxes)
    if len(bounding_boxes) > 6:
        print(bounding_boxes)
    
    for box in bounding_boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        crop_img = binarized[y:y+h, x:x+w]
        letters.append(cv_utils.padded(crop_img, 28, 28))
            
    return letters