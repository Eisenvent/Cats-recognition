# 貓的辨識

## 環境建構
由於個人電腦執行Yolo會需要很長的時間(GTX650)，所以使用**google colab**。

為了在colab上使用個人的雲端硬碟，需先連上雲端，並建立短捷徑。

在登入時會需要個人授權碼，貼上後並登入。

![1](https://user-images.githubusercontent.com/64704410/131439125-305f2b0c-1150-467b-9560-95be40da0a92.png)
```
from google.colab import drive

drive.mount('/content/gdrive')
!ln -fs /content/gdrive/MyDrive /app
```

在colab上執行yolo，需要先將yolo載下來，並修改設定，完成後便可編譯檔案。

```ini
# 抓取yolov4
!git clone https://github.com/AlexeyAB/darknet darknet

# 修改Darknet設定，符合Colab環境
%cd /content/darknet/
!sed -i "s/GPU=0/GPU=1/g" Makefile
!sed -i "s/CUDNN=0/CUDNN=1/g" Makefile
!sed -i "s/OPENCV=0/OPENCV=1/g" Makefile
!apt update
!apt-get install libopencv-dev

# 編譯
!make clean
!make
```
編譯完成後，將權重下載。

下載完成後，為了測試yolo是否安裝成功，進行簡單的測試。
```ini
# 抓取預先訓練完的權重
!wget https://pjreddie.com/media/files/yolov3.weights
!wget https://pjreddie.com/media/files/darknet53.conv.74
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

#測試
!./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg

#展示圖片
import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("predictions.jpg", cv2.IMREAD_UNCHANGED)
cv2_imshow(img)
```
![image](https://user-images.githubusercontent.com/64704410/131439805-76bcf2ec-132c-4ad1-b7ed-c32c3a5215c0.png)

沒有問題的話，將安裝好的環境製作成壓縮檔，以便往後可以快速利用。
```ini
%cd /content/
!zip -r darknet.zip darknet
!cp darknet.zip /app/darknet.zip
```

## 環境下載
在有yolo環境壓縮檔的情況下，可以**直接**連到雲端硬碟下載壓縮檔來使用yolo。
```ini
from google.colab import drive
import os

# 製作捷徑
drive.mount('/content/gdrive')
!ln -fs /content/gdrive/MyDrive /app

#複製darknet壓縮檔
!cp /app/darknet.zip /content/darknet.zip
#解壓縮
!unzip /content/darknet.zip

%cd darknet
```

## 模型訓練
將權重檔案複製到訓練資料夾cat_recognize下，並修改cfg檔案。

由於只有**1**個類別要辨識，所以將參數修改如下。
```ini
# 將cfg檔複製到訓練資料夾下
%cp /content/darknet/cfg/yolov4.cfg /app/cat_recognize/picture/cats_alter/cfg/yolov4_train1.cfg

# 修改cfg檔以符合訓練資料
%cd /app/cat_recognize/picture/cats_alter/cfg
!sed -i "s/width=608/width=416/g" yolov4_train1.cfg
!sed -i "s/height=608/height=416/g" yolov4_train1.cfg
!sed -i "s/subdivisions=8/subdivisions=64/g" yolov4_train1.cfg
!sed -i "s/max_batches = 500500/max_batches = 2000/g" yolov4_train1.cfg #batches = classes*2000
!sed -i "s/steps=400000,450000/steps=1600,1800/g" yolov4_train1.cfg
!sed -i "968c classes=1" yolov4_train1.cfg 
!sed -i "1056c classes=1" yolov4_train1.cfg 
!sed -i "1144c classes=1" yolov4_train1.cfg
#filters = 3*(classes+5)
!sed -i "961c filters=18" yolov4_train1.cfg
!sed -i "1049c filters=18" yolov4_train1.cfg
!sed -i "1137c filters=18" yolov4_train1.cfg
```
修改完畢，記得建立資料夾以儲存訓練產生的權重，之後就可以開始進行訓練。
```ini
#建立儲存訓練權重的資料夾
!mkdir /app/cat_recognize/picture/out2yolo/backup
```
在colab上執行yolo會有個*錯誤*：yolo在偵測完或是訓練完後會展示結果，但在colab上會呈現錯誤，因此需要在程式後加上 **-dont_show**。
```ini
%cd /content/darknet/
!./darknet detector train /app/cat_recognize/picture/out2yolo/obj.data /app/cat_recognize/picture/cats_alter/cfg/yolov4_train1.cfg yolov4.conv.137 -dont_show
```
## 成果辨識
進行辨識前，需要先將cfg函數修改至辨識用的數值，為了方便，複製cfg檔案進行修改。
```ini
%cp /app/cat_recognize/picture/cats_alter/cfg/yolov4_train1.cfg /app/cat_recognize/picture/cats_alter/cfg/yolov4_test1.cfg
# 修改cfg檔以偵測
%cd /app/cat_recognize/picture/cats_alter/cfg
!sed -i "1 s/64/1/g" yolov4_test1.cfg
!sed -i "2 s/64/1/g" yolov4_test1.cfg
!sed -i "4 s/Training/Testing/g" yolov4_test1.cfg
```
修改完後便可以使用訓練結果進行辨識。
```ini
#單張圖片檢測
import cv2
from google.colab.patches import cv2_imshow

def img_show(img_name):
  img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
  cv2_imshow(img)

%cd /content/darknet/
!./darknet detector test /app/cat_recognize/picture/out2yolo/obj.data /app/cat_recognize/picture/cats_alter/cfg/yolov4_test1.cfg /content/gdrive/MyDrive/cat_recognize/picture/out2yolo/backup/yolov4_train1_last.weights /app/cat_recognize/picture/cats/pexels-meruyert-gonullu-7317607.jpg > /content/gdrive/MyDrive/cat_recognize/picture/result/result.txt -dont_show

img_show('predictions.jpg')
```
得到的成果便如下

![下載](https://user-images.githubusercontent.com/64704410/131528212-b17881e0-0700-4934-8441-8b0bf71484ae.jpg)

## 參考資料
