import cv2 as cv
import numpy as np

# print(cv2.__file__)
source1_rgb = cv.imread("./02/rgb/0000.png", 1)
source1_d = cv.imread('./02/depth/0000.png', 2)

#source1 = np.stack((source1_rgb[:, :, 0], source1_rgb[:, :, 1], source1_rgb[:, :, 2], source1_d), axis=-1)
source2_rgb = cv.imread("./02/rgb/0001.png", 1)
source2_d = cv.imread("./02/depth/0001.png", 2)

#source2 = np.stack((source2_rgb[:, :, 0], source2_rgb[:,:, 1], source2_rgb[:, :, 2], source2_d), axis=-1)
#lineModDetector = cv.linemod_Detector()

#lineModDetector = cv.linemod.getDefaultLINE()
lineModDetector = cv.linemod.getDefaultLINEMOD()
print("Number of modalities:",len(lineModDetector.getModalities()))

# mask = cv2.bitwise_not(template)[:,:,1]
mask = np.array([])

templateID1, _ = lineModDetector.addTemplate((source1_rgb, source1_d), "circle", mask)
Templates = lineModDetector.getTemplates("circle", 0)
NumTemplates = lineModDetector.numTemplates()


print("Template ID:", templateID1)
print("Template 1:", Templates)
print("Number of templates", NumTemplates)


templateID2, _ = lineModDetector.addTemplate((source2_rgb, source2_d), "circle", mask)
Templates2 = lineModDetector.getTemplates("circle", 1)
NumTemplates = lineModDetector.numTemplates()
print("Template ID:", templateID2)
print("Template 2:", Templates2)
print("Number of templates", NumTemplates)

#print(boundingBox)

#threshold = 8.0
#class_ids = ""
#source = cv.imread('ran.jpg')
#matches, quantized_images = lineModDetector.match(source, threshold, class_ids)
