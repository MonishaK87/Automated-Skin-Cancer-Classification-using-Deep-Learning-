import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')


@app.route('/userlog.html')
def demo():
    return render_template('userlog.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)

        # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Skincancer-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 12, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        
        str_label=" "
        accuracy=""
        Tre=" "
        Tre1=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            

            if np.argmax(model_out) == 0:
                str_label = "actinic keratosis_cancer"
                print("The predicted image of the actinic keratosis is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the actinic keratosis is with a accuracy of {}%".format(model_out[0]*100)
                Tre = "The Treatment for actinic keratosis are:\n\n "
                Tre1 = [" Cryotherapy",  
                "Curettage and Electrosurgery"]
                
            elif np.argmax(model_out) == 1:
                str_label  = "basal cell carcinoma_cancer"
                print("The predicted image of the basal cell carcinoma is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the basal cell carcinoma is with a accuracy of {}%".format(model_out[1]*100)
                Tre = "The Treatment for basal cell carcinoma are:\n\n "
                Tre1 = [" Electrodessication and curettage",  
                "Photodynamic therapy",
                'Cryotherapy or cryosurgery',
                'Laser therapy']
                
            elif np.argmax(model_out) == 2:
                str_label = "dermatofibroma_cancer"
                print("The predicted image of the dermatofibroma is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the dermatofibroma is with a accuracy of {}%".format(model_out[2]*100)
                Tre = "The Treatment for dermatofibroma are:\n\n "
                Tre1 = [" Removal is typically the simplest and most successful option but it requires a surgical procedure",  
                "People may request this treatment if they have a growth that is unsightly or in an embarrassing place."]  

            elif np.argmax(model_out) == 3:
                str_label = "melanoma_cancer"
                print("The predicted image of the melanoma is with a accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the melanoma is with a accuracy of {}%".format(model_out[3]*100)
                Tre = "The Treatment for melanoma are:\n\n "
                Tre1 = [" Immunotherapy",  
                "Targeted therapy",
                'Radiation therapy']

            elif np.argmax(model_out) == 4:
                str_label = "nevus_cancer"
                print("The predicted image of the nevus is with a accuracy of {} %".format(model_out[4]*100))
                accuracy="The predicted image of the nevus is with a accuracy of {}%".format(model_out[4]*100)
                Tre = "The Treatment for nevus are:\n\n "
                Tre1 = [" Most moles are harmless and don’t require treatment",  
                "if you have a mole that’s cancerous or could become cancerous, you’ll likely need to have it removed"]

            elif np.argmax(model_out) == 5:
                str_label = "pigmented benign keratosis_cancer"
                print("The predicted image of the pigmented benign keratosis is with a accuracy of {} %".format(model_out[5]*100))
                accuracy="The predicted image of the pigmented benign keratosis is with a accuracy of {}%".format(model_out[5]*100)
                Tre = "The Treatment for pigmented benign keratosis are:\n\n "
                Tre1 = [" seborrheic keratosis typically doesn't go away on its own, but treatment isn't needed.",  
                "You might choose to have it removed if it becomes irritated or bleeds, or if you don't like how it looks or feels."]

            elif np.argmax(model_out) == 6:
                str_label = "seborrheic keratosis_cancer"
                print("The predicted image of the seborrheic keratosis is with a accuracy of {} %".format(model_out[6]*100))
                accuracy="The predicted image of the seborrheic keratosis is with a accuracy of {}%".format(model_out[6]*100)
                Tre = "The Treatment for basal cell carcinoma are:\n\n "
                Tre1 = [" Cryotherapy",  
                "Electrodessication/Curettage",
                'Laser Therapy']

            elif np.argmax(model_out) == 7:
                str_label = "Skin warts_disease"
                print("The predicted image of the Skin warts_diseaseis with a accuracy of {} %".format(model_out[7]*100))
                accuracy="The predicted image of the Skin warts_disease is with a accuracy of {}%".format(model_out[7]*100)
                Tre = "The Treatment for Skin warts_disease are:\n\n "
                Tre1 = [" Cryotherapy",  
                "Electrodessication/Curettage",
                'Laser Therapy']

            elif np.argmax(model_out) == 8:
                str_label = "Scabies_disease"
                print("The predicted image of the Scabies_disease is with a accuracy of {} %".format(model_out[8]*100))
                accuracy="The predicted image of the Scabies_disease is with a accuracy of {}%".format(model_out[8]*100)
                Tre = "The Treatment for Scabies_disease are:\n\n "
                Tre1 = [" Cryotherapy",  
                "Electrodessication/Curettage",
                'Laser Therapy']

            elif np.argmax(model_out) == 9:
                str_label = "Infectious erythema_disease"
                print("The predicted image of the Infectious erythema_disease is with a accuracy of {} %".format(model_out[9]*100))
                accuracy="The predicted image of the Infectious erythema_disease is with a accuracy of {}%".format(model_out[9]*100)
                Tre = "The Treatment for Infectious erythema_disease are:\n\n "
                Tre1 = [" Cryotherapy",  
                "Electrodessication/Curettage",
                'Laser Therapy']
            elif np.argmax(model_out) == 10:
                str_label = "Impetigo_disease"
                print("The predicted image of the Impetigo_disease is with a accuracy of {} %".format(model_out[10]*100))
                accuracy="The predicted image of the Impetigo_disease is with a accuracy of {}%".format(model_out[10]*100)
                Tre = "The Treatment for Impetigo_disease are:\n\n "
                Tre1 = [" Cryotherapy",  
                "Electrodessication/Curettage",
                'Laser Therapy']
            elif np.argmax(model_out) == 11:
                str_label = "Chickenpox_disease"
                print("The predicted image of the Chickenpox_disease is with a accuracy of {} %".format(model_out[11]*100))
                accuracy="The predicted image of the Chickenpox_disease is with a accuracy of {}%".format(model_out[11]*100)
                Tre = "The Treatment for Chickenpox_disease are:\n\n "
                Tre1 = [" Cryotherapy",  
                "Electrodessication/Curettage",
                'Laser Therapy']    
                            
                            
                            

        return render_template('results.html', status=str_label,accuracy=accuracy,treatment=Tre, treatment1=Tre1,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
