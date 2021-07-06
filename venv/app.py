import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle
from flask import Flask, render_template
from flask import request

app=Flask(__name__)
@app.route('/')
def hi():
  return "hello!"


@app.route('/catego', methods =["GET","POST"])
def hello():
  if request.method == "POST":
    Categories=['Cars','sunflowers','Ice cream cone']
    #print("Type y to give categories or type n to go with classification of Cars, Sunflowers and Ice cream cone");
    while(True):
      check=request.form.get('category')
      if(check=='n' or check=='y'):
        break
      #print("Please give a valid input (y/n)")
    if(check=='y'):
      #print("Enter How many types of images do you want to classify")
      n= (int)(request.form['types'])
      #n=request.form.get('types')
      Categories=[]
      #print(f'Please enter {n} names')
      # for i in range(n):
      name1=request.form.get('name1')
      name2=request.form.get('name2')
      Categories.append(name1)
      Categories.append(name2)
      print(Categories)
      #print(f"Please upload all the {n} category images with the same names as given in categories")
    
      #print(f"Please upload all the {n} category images with the same names as given in categories")
    flat_data_arr=[]
    target_arr=[]
    #datadir= "C:/Users/User/Desktop/fyproj/venv/Cat"
    datadir=request.form.get('url')
    print(datadir)
    for i in Categories:
      print(f'loading... category : {i}')
      path=os.path.join(datadir,i)
      for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
      print(f'loaded category:{i} successfully')
    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(flat_data)
    df['Target']=target
    df
    
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
    print('Splitted Successfully')
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    svc=svm.SVC(probability=True)
    print("The training of the model is started, please wait for while as it may take few minutes to complete")
    model=GridSearchCV(svc,param_grid)
    model.fit(x_train,y_train)
    print('The Model is trained well with the given images')
    model.best_params_
      
    y_pred=model.predict(x_test)
    #print("The predicted Data is :")
    y_pred
    #print("The actual data is:")
    np.array(y_test)
    #classification_report(y_pred,y_test)
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    #confusion_matrix(y_pred,y_test)
    pickle.dump(model,open('img_model.p','wb'))
    #print("Pickle is dumped successfully")
    #print(os.path.abspath(os.getcwd()))
    model=pickle.load(open('img_model.p','rb'))

    url=request.form.get('urlpic')
    img=imread(url)
    #plt.imshow(img)
    #plt.show()
    img_resize=resize(img,(150,150,3))
    l=[img_resize.flatten()]
    probability=model.predict_proba(l)
    for ind,val in enumerate(Categories):
      print(f'{val} = {probability[0][ind]*100}%')

    print("The predicted image is : "+Categories[model.predict(l)[0]])
    print(f'Is the image a {Categories[model.predict(l)[0]]} ?(y/n)')
    while(True):
      b=input()
      if(b=="y" or b=="n"):
        break
      print("please enter either y or n")

    if(b=='n'):
      print("What is the image?")
      for i in range(len(Categories)):
        print(f"Enter {i} for {Categories[i]}")
        k=int(input())
        while(k<0 or k>=len(Categories)):
          print(f"Please enter a valid number between 0-{len(Categories)-1}")
          k=int(input())
        print("Please wait for a while for the model to learn from this image :)")
        flat_arr=flat_data_arr.copy()
        tar_arr=target_arr.copy()
        tar_arr.append(k)
        flat_arr.extend(l)
        tar_arr=np.array(tar_arr)
        flat_df=np.array(flat_arr)
        df1=pd.DataFrame(flat_df)
        df1['Target']=tar_arr
        model1=GridSearchCV(svc,param_grid)
        x1=df1.iloc[:,:-1]
        y1=df1.iloc[:,-1]
        x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.20,random_state=77,stratify=y1)
        d={}
        for i in model.best_params_:
          d[i]=[model.best_params_[i]]
        model1=GridSearchCV(svc,d)
        model1.fit(x_train1,y_train1)
        y_pred1=model.predict(x_test1)
        print(f"The model is now {accuracy_score(y_pred1,y_test1)*100}% accurate")
        pickle.dump(model1,open('img_model.p','wb'))
      print("Thank you for your feedback")
    return render_template("abc.html", Categories0=Categories[model.predict(l)[0]])
  
  return render_template("abc.html")