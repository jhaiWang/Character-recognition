import urllib.request
import urllib.parse
import difflib
import json
import cv2,os
import time
import numpy as np
import base64
import matplotlib.pyplot as plt
url_request="https://ocrapi-entertainment.taobao.com/ocrservice/entertainment"
headers = {
         'Authorization': 'APPCODE 4c563bbc52454b72927f0c5181576e04',

    }
def posturl(url,data):
  try:
    params=json.dumps(data).encode(encoding='UTF8')
    req = urllib.request.Request(url, params,headers)
    r = urllib.request.urlopen(req)
    html =r.read()
    r.close()
    return html.decode("utf8")
  except urllib.error.HTTPError as e:
      print(e.code)
      print(e.read().decode("utf8"))
  time.sleep(1)

def recongimg(im):
    try:
        img = im[750:, 200:450]
        cv2.imwrite("region.jpg", img)
        image = cv2.imread("region.jpg")
        with open('region.jpg', 'rb') as f:  # 以二进制读取图片
            data = f.read()
            encodestr = str(base64.b64encode(data), 'utf-8')
            dict = {'img': encodestr}
            html = posturl(url_request, dict)
            user_dict = json.loads(html)
            demoresult = user_dict['prism_wordsInfo']
        return demoresult
    except urllib.error.HTTPError as e:
        print(e.code)
        print(e.read().decode("utf8"))

def regionsimilar(demoresult,photoresult):
    similaregion = []
    try:
        for i in range(len(demoresult)):
            demoloaction = demoresult[i]
            demoregionx = demoloaction['pos'][0]['x']
            demoregiony = demoloaction['pos'][0]['y']
            for j in range(len(photoresult)):
                photolocation = photoresult[j]
                photoregionx = photolocation['pos'][0]['x']
                photoregiony = photolocation['pos'][0]['y']
                if abs(demoregionx - photoregionx) < 10 and abs(demoregiony - photoregiony) < 10:
                    photovalue = demoloaction['word']
                    demovalue = photolocation['word']
                    result = difflib.SequenceMatcher(None, photovalue, demovalue).quick_ratio()
                    if result > 0.8:
                        similaregion.append(photovalue)
        num = len(similaregion)/len(demoresult)
        return similaregion,num
    except urllib.error.HTTPError as e:
        print(e.code)
        print(e.read().decode("utf8"))


if __name__=="__main__":
    imgpath_demo = "E://imageproject//Image_python//similar//res//"
    imgpath_photo = "E://imageproject//Image_python//similar//new//"
    for img in os.listdir(imgpath_demo):
        img1_gray = cv2.imread(imgpath_demo + img, 0)
        for image in os.listdir(imgpath_photo):
            img2_gray = cv2.imread(imgpath_photo + image, 0)

            img1 = img1_gray[750:, 200:450]
            img2 = img2_gray[750:, 200:450]

            htitch = np.hstack((img1, img2))
            cv2.imshow("test1", htitch)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            demoresult = recongimg(img1_gray)
            photoresult = recongimg(img2_gray)

            result,num = regionsimilar(demoresult,photoresult)
            print(result,num)
    # for i in range(len(demoresult)):
    #     demoloaction = demoresult[i]
    #     demoregionx = demoloaction['pos'][0]['x']
    #     demoregiony = demoloaction['pos'][0]['y']
    #     for j in range(len(photoresult)):
    #         photolocation = photoresult[j]
    #         photoregionx = photolocation['pos'][0]['x']
    #         photoregiony = photolocation['pos'][0]['y']
    #
    #         if abs(demoregionx - photoregionx) < 10 and abs(demoregiony - photoregiony) < 10:
    #             photovalue = demoloaction['word']
    #             demovalue = photolocation['word']
    #             result = difflib.SequenceMatcher(None, photovalue, demovalue).quick_ratio()
    #             # if result > 0.8:
    #             print(result,photovalue, demovalue)



