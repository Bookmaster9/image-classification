#DOWNLOAD DATA

# mount google drive
from google.colab import drive

from google.colab import files

drive.mount('/content/drive/')

!ls "/content/drive/My Drive"

#download

! pip install -q kaggle

!mkdir ~/.kaggle

!cp /content/drive/My\ Drive/ai/kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d andrewmvd/pediatric-pneumonia-chest-xray
#https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

!ls

!unzip pediatric-pneumonia-chest-xray.zip

!ls
