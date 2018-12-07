import cv2
from PIL import ImageDraw, ImageFont, Image
from googletrans import Translator
import random
from PyDictionary import PyDictionary
import pandas as pd
import pytesseract
import enchant
import numpy as np
import time

class imagePreProcess:

    def __init__(self, imageName=None, lang="eng"):
        self.imagePath = imageName
        self.lang = lang
        self.processing()
    def pixelIntensities(self, xValue, yValue):
        #intensity calculator in RGB
        if xValue >= self.width or yValue >= self.height:
            print("Intensities values are out of bounds")
            return 0
        xyIntensity = self.imageName[yValue][xValue]
        #the multiplication factor that TV's use for B, G, R are the ones used here
        return xyIntensity[0] * 0.11 + xyIntensity[1] * 0.59 + xyIntensity[2] * 0.30
    def connectedContours(self, contourList):
        begin = contourList[0][0]
        length = len(contourList) - 1
        end = contourList[length][0]
        #is contour connected?
        return abs(begin[0] - end[0]) <= 1 and abs(begin[1] - end[1]) <= 1
    def findContour(self, index):
        return self.contours[index]
    def countChildren(self, index, indexList, contourList):
        if indexList[index][2]<0:
            print("no children")
            return 0
        count = 0
        if self.needContour(self.findContour(indexList[index][2])):
            #first child = needed contour
            count = 1
        #count child's sibling and descendants
        count += self.countSiblings(indexList[index][2], indexList, contourList, True)
        return count
    def isChild(self, index, indexList):
        #check if the contour has a parent
        return self.getParent(index, indexList) > 0
    def getParent(self, index, indexList):
        parent = indexList[index][3]
        while not self.needContour(self.findContour(parent)) and parent > 0:
            parent = indexList[parent][3]
        return parent
    def countSiblings(self, index, indexList, contourList, hasChildren = False):
        count = 0
        #if it has children
        if hasChildren:
            count += self.countChildren(index, indexList, contourList)

        #next contour
        nextContour = indexList[index][0]
        while nextContour > 0:
            if self.needContour(self.findContour(nextContour)):
                #if revelant
                count+=1
            if hasChildren:
                #contour has children, so add them
                count+=self.countChildren(nextContour, indexList, contourList)
            nextContour = indexList[nextContour][0]

        #next contour
        previousContour = indexList[index][1]
        while previousContour > 0:
            if self.needContour(findContour(previousContour)):
                #if revelant
                count += 1
            if hasChildren:
                #contour has children, so add them
                count += self.countChildren(previousContour, indexList, contourList)
            previousContour = indexList[previousContour][1]
        return count
    def needContour(self, contour):
        #is this contour needed?
        return self.needBox(contour) and self.connectedContours(contour)
    def needBox(self, contour):
        #x,y intensities; width, height of contour
        xValue, yValue, width, height = cv2.boundingRect(contour)
        #width/height => floats
        width *= 1.0
        height *= 1.0
        #if its shape is very small or very large, not need
        if (width/height < 0.15) or (width/height > 10) :
            print("wrong contour shape")
            return False
        #check the Box size
        if ((width * height) > ((self.width * self.height) /5) or ((width * height) < 15)):
            print("wrong box size")
            return False
        return True
    def addBox(self, index, indexList, contourList):
        #not a revelant 
        if self.isChild(index, indexList) and (self.countChildren(self.getParent(index, indexList),indexList, contourList) <= 2):
            return False
        if self.countChildren(index, indexList, contourList) > 2:
            return False
        return True
    def processing(self):
        #original image and its dimensions
        if (self.imagePath==None):
            return ("Please check your inputs to the function and make sure that they are correct: especially Trained Mdel, and Image Path")
        #image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        #(thresh, imageName) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.originalImg = cv2.imread(self.imagePath)
        self.height, self.width = self.originalImg.shape[0], self.originalImg.shape[1]
        #padding the image
        self.imageName = cv2.copyMakeBorder(self.originalImg, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
        #split RGB into R G B, three-scale 
        blue, green, red = cv2.split(self.imageName)
        #canny edge detection => detect edges on each scale
        blueCanny = cv2.Canny(blue, 200, 250)
        redCanny = cv2.Canny(red, 200, 250)
        greenCanny = cv2.Canny(green, 200, 250)
        #join the result of canny detection back 
        cannyImg = blueCanny | greenCanny | redCanny
        #finding contours
        #returns the image, contours, and their hierarchy
        imageName, self.contours, hierarchy = cv2.findContours(cannyImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #hierarchy = [nextContour, previousContour, firstChild, Parent]
        hierarchy = hierarchy[0]
        #for each contour, determine if its ROI or not
        needBoxes = self.isContourNeeded(hierarchy)
        #create a copy of the image
        whiteImg = cannyImg.copy()
        #all intensity values = 255 = white
        whiteImg.fill(255)
        #distinguish foreground and background
        whiteImg = self.findFgBg(needBoxes, whiteImg)
        #blur the image
        blurredImg = cv2.blur(whiteImg, (2,2))
        self.textResult = self.img2Text(blurredImg)
        return self.textResult
    def img2Text(self, imageName):
        #uses pytesseract to extract the text from
        #the preprocessed image
        return pytesseract.image_to_string(imageName, lang=self.lang)
    def findFgBg(self, needBoxes, whiteImg):
        #distinguish foreground and background
        for index, (contour, box) in enumerate(needBoxes):
            #edge pixels determines foreground
            fGroundInt = 0.0
            for i in contour:
                fGroundInt += self.pixelIntensities(i[0][0], i[0][1])
            fGroundInt /= len(contour)
            x, y, w, h = box
            bGroundInt = \
                [ 
                    #bottom left corner 3 pixels
                    self.pixelIntensities(x - 1, y - 1),
                    self.pixelIntensities(x - 1, y),
                    self.pixelIntensities(x, y - 1),

                    #bottom right corner 3 pixels
                    self.pixelIntensities(x + w + 1, y - 1),
                    self.pixelIntensities(x + w, y - 1),
                    self.pixelIntensities(x + w + 1, y),

                    # top left corner 3 pixels
                    self.pixelIntensities(x - 1, y + h + 1),
                    self.pixelIntensities(x - 1, y + h),
                    self.pixelIntensities(x, y + h + 1),

                    # top right corner 3 pixels
                    self.pixelIntensities(x + w + 1, y + h + 1),
                    self.pixelIntensities(x + w, y + h + 1),
                    self.pixelIntensities(x + w + 1, y + h)
                ]
            #median of bGround intensities
            bGroundInt = np.median(bGroundInt)
            if fGroundInt >= bGroundInt:
                foreGround = 255
                backGround = 0
            else:
                foreGround = 0
                backGround = 255
            #=================================================
            #create bw image by coloring only those pixels
            #whose edges showed up in the canny edge and 
            #their contours were needed
            #=================================================
            for i in range(x, x + w):
                for j in range(y, y + h):
                    if j >= self.height or i >= self.width:
                        continue
                    if self.pixelIntensities(i ,j) > fGroundInt:
                        whiteImg[j][i] = backGround
                    else:
                        whiteImg[j][i] = foreGround
        return whiteImg
    def isContourNeeded(self, hierarchy):
        needBoxes = []
        for index, contour in enumerate(self.contours):
            #return the x,y coordinate and the width, height of the contour
            x, y, w, h = cv2.boundingRect(contour)
            #check contour and it's bounding box
            if self.needContour(contour) and self.addBox(index, hierarchy, contour):
                needBoxes.append([contour, [x,y,w,h]])
        return needBoxes
    def getResult(self):
        print(type(self.textResult))
        return self.textResult
    def __str__(self):
        return self.textResult



def project(fileName, imageName, translate=False, meaning=False, thesa=False, dest="en"):
    fileToWrite = open(fileName, "w+")
    fileToWrite.close()
    new = imagePreProcess(imageName)
    result = new.getResult()
    fileToWrite = open(fileName, "a")
    fileToWrite.write(result)
    if translate==True:
        fileToWrite.write("\n\n\n=======================================================================\n\n\n")
        fileToWrite.write("The above code's translation in " + dest + " is given below:")
        fileToWrite.write("\n\n\n=======================================================================\n\n\n")
        translator = Translator()
        translatedResult = translator.translate(result, dest=dest)
        fileToWrite.write(translatedResult.text)
        fileToWrite.write("\n\n\n=======================================================================\n\n\n")
        fileToWrite.write("The above code's translation in the original language is given below:")
        fileToWrite.write("\n\n\n=======================================================================\n\n\n")
        orig = translator.translate(translatedResult.text, dest="en")
        fileToWrite.write(orig.text)
        img = Image.new('RGB', (600, 300), color = 'white')
        d = ImageDraw.Draw(img)
        d.text((100,100), orig.text, fill=(0,0,0))
        img.save('resulted.png')
    if meaning==True:
        result = result.split(" ")
        print("len result "+ str(len(result)))
        print(result)
        iteratorResult = result
        for i in range(len(iteratorResult)):
            result = iteratorResult[i].split(".")[0]
            dfThes = PyDictionary()
            spellCheck = enchant.Dict("en_US")
            punctuation = [".", "?", "!", ",", "''"]
            if spellCheck.check(result) == False:
              result = spellCheck.suggest(result)
              indexMean = random.randint(0, len(result)-1)
              print(indexMean)
              result = result[indexMean]
            fileToWrite.write(result)
            index = dfThes.meaning(result)
            fileToWrite.write("\nmeaning: \n")
            print(index)
            if index!=None:
                for i in index:
                  meaningWord = index[i]
                  fileToWrite.write("\n")
                  fileToWrite.write(i)
                  fileToWrite.write(": ")
                  img = Image.new('RGB', (600, 300), color = 'white')
                  d = ImageDraw.Draw(img)
                  for j in meaningWord:
                      fileToWrite.write(j)
                      fileToWrite.write(", ")
                      d.text((100,100), j, fill=(0,0,0))
            img.save('resulted.png')
    fileToWrite.close()

project(fileName="newFile.txt", imageName="download.jpeg", meaning=True, dest='la') 




# trans = tlr.translate(g, dest='ko')
# print(trans)
# dfThes = PyDictionary()
# spellCheck = enchant.Dict("en_US")
# punctuation = [".", "?", "!", ",", "''"]
# for text in new:
#     if spellCheck.check(text) == False:
#     	text = spellCheck.suggest(text)
#     	indexMean = random.randint(0, len(text)-1)
#     	print(indexMean)
#     	text = text[indexMean]
#     fileToWrite.write(text)
#     if (operation=="meaning"):
#         index = dfThes.meaning(text)
#     elif (operation=="translate"):
#         index = dfThes.translate(text, lang)
#     fileToWrite.write("\n"+operation+": \n")
    
#     for i in index:
#     	meaningWord = index[i]
#     	fileToWrite.write("\n")
#     	fileToWrite.write(i)
#     	fileToWrite.write(": ")
#     	for j in meaningWord:
#     	    fileToWrite.write(j)
#     	    fileToWrite.write(", ")
# fileToWrite.close()
