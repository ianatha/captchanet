import cv2

def first_element(val):
    return val[0]

def merge_boxes(b1, b2):
    b1_xa, b1_ya, b1_w, b1_h = b1[0], b1[1], b1[2], b1[3]
    b2_xa, b2_ya, b2_w, b2_h = b2[0], b2[1], b2[2], b2[3]
    b1_xb, b1_yb = b1_xa + b1_w, b1_ya + b1_h
    b2_xb, b2_yb = b2_xa + b2_w, b2_ya + b2_h
    
    xa = min(b1_xa, b2_xa)
    ya = min(b1_ya, b2_ya)
    xb = max(b1_xb, b2_xb)
    yb = max(b1_yb, b2_yb)
    w = xb - xa
    h = yb - ya
    return [xa, ya, w, h]
    
def merge_bounding_boxes(bounding_boxes):
    i = 1
    res_boxes = []
    skip_box = False
    for box in bounding_boxes[1:]:
        prev_box = bounding_boxes[i - 1]
        if (prev_box[0] == box[0]) or (prev_box[0] + prev_box[2]) == (box[0] + box[2]):
            res_boxes.append(merge_boxes(prev_box, box))
            skip_box = True
        else:
            if not skip_box:
                res_boxes.append(prev_box)
            skip_box = False
        i = i + 1
    if not skip_box:
        res_boxes.append(bounding_boxes[i - 1])
    return res_boxes

def split_large_bounding_boxes(bounding_boxes, max_width = 23):
    res = []
    for box in bounding_boxes:
        if box[2] > max_width:
            b1 = box.copy()
            b2 = box.copy()
            b1[2] = int(b1[2] / 2)
            b2[2] = int(b2[2] / 2)
            res.append(b1)
            res.append(b2)
        else:
            res.append(box)
    return res

def padded(image, target_w, target_h):
    h, w = image.shape[:2]
    if w < target_w or h < target_h:
        extra_w = int((target_w - w)/2)
        extra_w_1 = target_w - w - extra_w
        extra_h = int((target_h - h)/2)
        extra_h_1 = target_h - h - extra_h
        padded = cv2.copyMakeBorder(image, extra_h, extra_h_1, extra_w, extra_w_1, cv2.BORDER_CONSTANT, value=[255,255,255])        
        return padded
    else:
        return image