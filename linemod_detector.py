import cv2 as cv
import numpy as np
from tqdm import tqdm
import pandas as pd
import os



data_dir = "/home/user/Datasets/hinterstoisser/train/"
test_dir = "/home/user/Datasets/hinterstoisser/test/"
classes = ["01", "02", "03", "04", "05", "06", "07", "08", "09",
            "10", "11", "12", "13", "14", "15"]


lineModDetector = cv.linemod.getDefaultLINEMOD()
mask = np.array([])
counter = 0
for class_id in classes:
    counter += 1
    df = pd.read_csv("./" + class_id + ".csv")
    rgb_list = df["rgb"]
    depth_list = df["depth"]
    for i in tqdm(range(len(os.listdir(data_dir+class_id+"/rgb/"))), desc="Adding templates from class {}".format(class_id)):
        rgb_img = cv.imread(data_dir + class_id + "/rgb/" + rgb_list[i], 1)
        depth_img = cv.imread(data_dir + class_id + "/depth/" + depth_list[i], 2)
        templateID, _ = lineModDetector.addTemplate((rgb_img, depth_img), class_id=class_id, object_mask=mask)
    if counter==5:
        break
NumTemplates = lineModDetector.numTemplates()
print("[INFO] Number of templates:", NumTemplates)

class_ids = ["05"]
threshold = 70.0
source_rgb = cv.imread(test_dir + "05/" + "rgb/" + "0504.png", 1)
source_d = cv.imread(test_dir + "05/" + "depth/" + "0504.png", 2)
matches, quantized_images = lineModDetector.match(sources=(source_rgb, source_d), threshold=threshold, class_ids=class_ids, masks=mask)

if len(matches) > 0:
    print("[INFO] Number of matches:{}\n".format(len(matches)))
    m = matches[:10]
    """classes = set([match.class_id for match in m])
    locations = {Class: [] for Class in classes}"""

    #img0 = np.array([])
    for j in range(len(m)):
        print("Match {}\t Similarity: {:.2f}\t x: {}\t y: {}\t class_id: {}\t template_id: {}".format(j+1, m[j].similarity, m[j].x, m[j].y, m[j].class_id, m[j].template_id))
        df = pd.read_csv("./" + str(m[j].class_id) + ".csv")
        rgb_list = df["rgb"]
        ref_rgb = cv.imread(data_dir + str(m[j].class_id) + "/rgb/" + str(rgb_list[m[j].template_id]), 1)
        cv.circle(source_rgb, (m[j].x, m[j].y), 10, (255, 0, 0), thickness=2)
        #source_rgb[m[j].x, m[j].y] = (255, 255, 0)
        img_j = cv.hconcat([source_rgb, ref_rgb])
        if j == 0:
            img = img_j
        else:
            img = cv.vconcat([img, img_j])    
        #locations[m[j].class_id].append((m[j].x, m[j].y))

    """for Class in classes:
        locations[Class] = np.asarray(locations[Class])
        min_x = min(locations[Class][:, 0])
        max_x = max(locations[Class][:, 0])
        min_y = min(locations[Class][:, 1])
        max_y = max(locations[Class][:, 1])
        cv.rectangle(source_rgb, (max_x, max_y), (min_x, min_y), (255, 0, 0), 2)"""


    #img = cv.hconcat([source_rgb, ref_rgb])
    cv.imwrite("matches.png",img)
    #cv.imshow("LINEMOD matching result", img)
    #cv.waitKey(0)

else:
    print("No matches found...")

# similarity, x,y , class_id, template.id
#print("First match:", m.similarity, m.x, m.y, m.class_id, m.template_id)
