import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
import pandas as pd

data_dir = "/home/anicet/Datasets/LINEMOD/hinterstoisser/train/"
test_dir = "/home/anicet/Datasets/LINEMOD/hinterstoisser/test/"
classes = ["01", "02", "03", "04", "05", "06", "07", "08", "09",
            "10", "11", "12", "13", "14", "15"]


lineModDetector = cv.linemod.getDefaultLINEMOD()
mask = np.array([])
#counter = 0
for class_id in classes:
    #counter += 1
    df = pd.read_csv("./" + class_id + ".csv")
    rgb_list = df["rgb"]
    depth_list = df["depth"]
    for i in tqdm(range(len(os.listdir(data_dir+class_id+"/rgb/"))), desc="Adding templates from class {}".format(class_id)):
        rgb_img = cv.imread(data_dir + class_id + "/rgb/" + rgb_list[i], 1)
        depth_img = cv.imread(data_dir + class_id + "/depth/" + depth_list[i], 2)
        templateID, _ = lineModDetector.addTemplate((rgb_img, depth_img), class_id=class_id, object_mask=mask)
    #     if i == 99:
    #         break
    # if counter==2:
    #     break
NumTemplates = lineModDetector.numTemplates()
print("[INFO] Number of templates:", NumTemplates)

class_ids = "" #["02"]
threshold = 70.0
source_rgb = cv.imread(test_dir + "01/" + "rgb/" + "0000.png", 1)
source_d = cv.imread(test_dir + "01/" + "depth/" + "0000.png", 2)
matches, quantized_images = lineModDetector.match(sources=(source_rgb, source_d),
                            threshold=threshold, class_ids=class_ids, masks=mask)


if len(matches) > 0:
    print("[INFO] Number of matches: {}\n".format(len(matches)))
    m = matches[:10]

    for i in range(len(m)):
        print("Match {}\t Similarity: {:.2f}\t x: {}\t y: {}\t class_id: {}\t \
                template_id: {}".format(i+1, m[i].similarity, m[i].x, m[i].y, m[i].class_id, m[i].template_id))
        
        df = pd.read_csv("./" + str(m[i].class_id) + ".csv")
        rgb_list = df["rgb"]
        ref_rgb = cv.imread(data_dir + str(m[i].class_id) + "/rgb/" + str(rgb_list[m[i].template_id]), 1)
        source_rgb_copy = source_rgb.copy()
        #cv.circle(source_rgb_copy, (m[i].x, m[i].y), 10, (255, 0, 0), thickness=2)

        # Find gradient and surface normal feature locations in the template image
        templates = lineModDetector.getTemplates(class_id=matches[i].class_id, template_id=matches[i].template_id)
        gradTemplate = templates[0]                 # choose grad features of the first pyramid level
        surfaceNormalTemplate = templates[1]        # choose surface features of the first pyramid level
        gradFeatures = gradTemplate.features
        surfaceNormalFeatures = surfaceNormalTemplate.features
        gradFeature_locations = [(f.x, f.y) for f in gradFeatures]
        surfaceFeature_locations = [(g.x, g.y) for g in surfaceNormalFeatures]
        
        # Plot gradient feature locations 
        offset = (m[i].x, m[i].y)
        for point in gradFeature_locations:
            x = point[0] + offset[0]
            y = point[1] + offset[1]
            cv.circle(source_rgb_copy, (x, y), 1, (255, 0, 0), thickness=2)

        # Plot surface normal features
        for point in surfaceFeature_locations:
            x = point[0] + offset[0]
            y = point[1] + offset[1]
            cv.circle(source_rgb_copy, (x, y), 1, (0, 255, 0), thickness=2)

        img_i = cv.hconcat([source_rgb_copy, ref_rgb])
        if i == 0:
            img = img_i
        else:
            img = cv.vconcat([img, img_i])

    cv.imwrite("test01.png", img)

else:
    print("No matches found...")
