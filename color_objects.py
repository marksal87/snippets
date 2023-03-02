# Color-object tracking code


import numpy as np
import cv2
from IPython import display

from PIL import Image, ImageOps
import imutils
from imutils import contours


def load_image(path, resize_dim=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_dim:
        img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_NEAREST)
    img = np.asarray(img).astype(np.uint8)
    return img

def imshow(img, is_bgr=True):
    if len(img.shape) == 2:
        _,ret = cv2.imencode('.jpg', img)
    else:
        # Color
        if is_bgr:
            _,ret = cv2.imencode('.jpg', img[:, :, ::-1])
        else:
            _,ret = cv2.imencode('.jpg', img)
    i = display.Image(data=ret)
    display.display(i)


# A function to fix HSV range
def fixHSVRange(h, s, v):
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * h / 360, 255 * s / 100, 255 * v / 100)

def pad_image_ones(im, size=1, value=1):
    if len(im.shape) == 3:
        new_im = np.ones((im.shape[0]+size*2, im.shape[1]+size*2, im.shape[2]), dtype=im.dtype)
        new_im *= value
        new_im[size:size+im.shape[0], size:size+im.shape[1], :] = im
    elif len(im.shape) == 2:
        new_im = np.ones((im.shape[0]+size*2, im.shape[1]+size*2), dtype=im.dtype)
        new_im *= value
        new_im[size:size+im.shape[0], size:size+im.shape[1]] = im
    return new_im

def find_color_objects(im, 
                       hsv_range_min=(170, 30, 30), 
                       hsv_range_max=(280, 100, 100), 
                       debug_draw=False):
    """
    """
    H, W = im.shape[:2]

    # Make a copy of Image; find the HSV range; convert it to OpenCV's HSV range and make a mask from it
    frm = im.copy()
    frm = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frm, fixHSVRange(*hsv_range_min), fixHSVRange(*hsv_range_max))

    # Remove the noise
    dilate_size = 5 # default 5
    erode_size = 5 # default 5
    blur_size = 47 # default 7
    noise=cv2.dilate(mask,np.ones((dilate_size,dilate_size)))
    noise=cv2.erode(mask,np.ones((erode_size,erode_size)))
    noise=cv2.medianBlur(mask, blur_size)

    # Change image channels
    mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    noise=cv2.cvtColor(noise,cv2.COLOR_GRAY2BGR)
    cleanMask=~noise

    # Make a new mask without noise
    centerMask=cv2.cvtColor(cleanMask.copy(),cv2.COLOR_BGR2GRAY)
    
    # Debug image
    if debug_draw:
        out = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Find contours and sort them by width
    contour_pad_size = 1
    centerMask = pad_image_ones(centerMask, size=contour_pad_size, value=255)
    cnts = cv2.findContours(centerMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    # Find objects
    objects = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        x -= contour_pad_size
        y -= contour_pad_size
        if w<W:
            objects.append({
                'bbox_xywh': (x,y,w,h),
                'center_xy': (x+w//2, y+h//2)
            })
            if debug_draw:
                print(f"Drawing contour x,y={x},{y}   w,h={w},{h}")
                cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.circle(out,(x+w//2,y+h//2),radius=int(min(im.shape[:2])*0.05),thickness=6,color=(127,0,220))
                cv2.rectangle(centerMask, (x, y), (x+w, y+h), 127, 3)

    if debug_draw:
        # Show the output
        centerMask=cv2.cvtColor(centerMask,cv2.COLOR_GRAY2BGR)
        centerMask = centerMask[contour_pad_size:centerMask.shape[0]-contour_pad_size, contour_pad_size:centerMask.shape[1]-contour_pad_size, :] # un-pad the mask
        top=np.hstack((im_rgb,mask,noise))
        btm=np.hstack((cleanMask,centerMask,out))
        imshow(np.vstack((top,btm)))
    
    return objects
  
# Example usage
# im = cv2.imread("test.png")
# find_color_objects(im, debug_draw=True)
