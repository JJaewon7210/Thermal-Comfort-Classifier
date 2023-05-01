from ast import Assert
import numpy as np
import cv2
import tqdm
from PIL import Image, ImageDraw
import os
import pandas as pd
import tifffile as tiff
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
PART_INDICES = {
    # 'nose': [i for i in range(27, 36)],
    'left_nose': [27, 28, 29, 30, 31, 32, 33],
    'right_nose': [27, 28, 29, 30, 33, 34, 35],

    # 'chin': [48, 59, 58, 57, 56, 55, 54, 11, 10, 9, 8, 7, 6, 5],
    'left_chin': [48, 59, 58, 57, 5, 6, 7, 8],
    'right_chin': [57, 56, 55, 54, 11, 10, 9, 8],

    'left_cheek': [1, 2, 3, 4, 31, 39, 40, 41, 36],
    'right_cheek': [12, 13, 14, 15, 45, 46, 47, 42, 35],

    'left_periorbital': [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41],
    'right_periorbital': [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47],

    # 'mouth': [i for i in range(48, 68)],
    'left_mouth': [48, 49, 50, 51, 59, 58, 57],
    'right_mouth': [54, 53, 52, 51, 55, 56, 57],

    # 'face_all': [i for i in range(0, 68)],
    'left_face_all': [0, 1, 2, 3, 4, 5, 6, 7, 8,
                      17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41,
                      27, 28, 29, 30, 31, 32, 33,
                      48, 49, 50, 51, 59, 58, 57],
    'right_face_all':[8, 9, 10, 11, 12, 13, 14 ,15 ,16,
                      22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47,
                      27, 28, 29, 30, 33, 34, 35,
                      54, 53, 52, 51, 55, 56, 57]
}


def make_dir(path):
    if '.' in path:
        dir = os.path.abspath(os.path.join(path, '..'))
    else:
        dir = path

    if not os.path.isdir(dir):
        os.makedirs(dir)

def extract_polygon_points(points, indices):
    polygon = []
    for i in indices:
        x = points[i, 0]
        y = points[i, 1]
        # add `1` pixel bumper to hull points
        if not np.isnan(x) and not np.isnan(y):
            polygon.append((x,y))
            # polygon.append((x+1, y))
            # polygon.append((x, y+1))
            # polygon.append((x-1, y))
            # polygon.append((x, y-1))
        
    hull = ConvexHull(polygon)
    hull_points = [polygon[i] for i in hull.vertices]

    return hull_points


def crop_face_element(image, points, part):

    # create mask
    polygon = extract_polygon_points(points, PART_INDICES[part])

    # create new image
    mask_img = Image.new('1', (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_img)
    
    # apply mask to image
    masked_img = image * mask

    # extract polygon values
    polygon_values = masked_img[mask != 0]

    return polygon_values


def extract_pixel_by_element(image, points, part, threshold=20):
    '''
    image: numpy array (RGB channel)
    points: 68 landmarks shape of (68,2)
    part: one of [nose, chin, cheek, periorbital, mouth, face_all]
    threshold: ignore the pixel intensity under threshold
    '''
    crop_face_array = crop_face_element(image.copy(), points, part)
    # gray_values = [i for i in crop_face_array if i > threshold]
    return crop_face_array

if __name__ == '__main__':
    df = pd.read_excel(
        'D:/ThermalData/Charlotte_ThermalFace/SPIGA_results_charlotte_RAINBOW.xlsx', index_col=0)
    df = df[df['Distance'] <= 3]
    df = df[~df['ID'].duplicated(keep='first')]
    df.dropna(subset=['pred_x1'], inplace=True)
    df = df.reset_index(drop=True)
    
    for i in tqdm.tqdm(range(len(df))):
        # image
        ID = df.loc[i, 'ID']
        Subject = df.loc[i, 'Subject']
        img = tiff.imread(f'D:/ThermalData/Charlotte_ThermalFace/S{Subject}/{ID}.tiff')

        # landmarks
        landmarks_columns = []
        for num in range(68):
            landmarks_columns.append(f'pred_x{num}')
            landmarks_columns.append(f'pred_y{num}')

        landmarks = df.loc[i, landmarks_columns].to_numpy()
        landmarks = np.reshape(landmarks, (68, 2))
        
        try:
            # temp
            for part in ['nose', 'chin', 'cheek', 'periorbital', 'mouth', 'face_all']:
                
                arry1 = extract_pixel_by_element(img, landmarks, f'right_{part}')
                arry2 = extract_pixel_by_element(img, landmarks, f'left_{part}')

                if part in ['nose', 'chin', 'cheek', 'periorbital', 'mouth', 'face_all']:
                    if len(arry1) <= len(arry2):
                        arry = arry2
                    elif len(arry2) <= len(arry1):
                        arry = arry1
                        
                else:
                    arry = np.concatenate((arry1, arry2))
                
                df.loc[i, f'{part}_max'] = round(np.nanmax(arry)/100 - 273.15,            4)
                df.loc[i, f'{part}_min'] = round(np.nanmin(arry)/100 - 273.15,            4)
                df.loc[i, f'{part}_median']  = round(np.nanmedian(arry)/100 - 273.15,     4)
                df.loc[i, f'{part}_average'] = round(np.nanmean(arry)/100 - 273.15,       4)
                df.loc[i, f'{part}_75p'] = round(np.nanpercentile(arry, 75)/100 - 273.15, 4)
                df.loc[i, f'{part}_25p'] = round(np.nanpercentile(arry, 25)/100 - 273.15, 4)
        except:
            print(i, ID)
                
    df.to_csv('D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv')
