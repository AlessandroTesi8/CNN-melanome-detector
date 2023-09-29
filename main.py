from engine import model_predict
import os 
from PIL import Image


if __name__=="__main__":

    quest_path = os.getcwd() + "/quest/"

    elenco_file = os.listdir(quest_path)

    for image in elenco_file:
        try:
            img = Image.open(quest_path + str(image))
            print(f"\nthe class predicted for the image {image} is:")
            model_predict(img)
        except:
            print(f'\n{image} is not an image or has not jpg format')