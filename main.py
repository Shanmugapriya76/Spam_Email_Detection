import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time
import sklearn
import mailbox
import mailbox
import Load_and_Preprocess
from textblob import TextBlob
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,learning_curve
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score

#set target dataset directory
dir=input("Please enter your the location of your spam files, should contain Spam.mobx and Inbox.mbox\n")

#load and parse the spam message box
mbox_spam = mailbox.mbox(dir+'\Spam.mbox')
list_spam = []
for message in mbox_spam:
    list_spam.append(Load_and_Preprocess.preprocess(message))
#convert list to dataframe
df_spam = pd.DataFrame(list_spam, columns=["message"])
df_spam["label"] = "spam"
df_spam['length'] = df_spam['message'].map(lambda text: len(text))

#load and parse the ham message box, set maximum size to 1000 ham emails
mbox_ham = mailbox.mbox(dir+'\Inbox.mbox')
list_ham = []
MaxLength=0
for message in mbox_ham:
    list_ham.append(Load_and_Preprocess.preprocess(message))
    MaxLength+=1
    if MaxLength>1000:
        break
#convert list to dataframe
df_ham = pd.DataFrame(list_ham, columns=["message"])
df_ham["label"] = "ham"
df_ham['length'] = df_ham['message'].map(lambda text: len(text))

#merge spam and ham to a new dataframe
df_merged = pd.concat([df_spam,df_ham])

#split the dataset
mail_train, mail_test, y_train, y_test = train_test_split(df_merged['message'],df_merged['label'],test_size=0.5, random_state=42)

#for each word in the email text, get the base form of the word and return the list of base words
def split_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # return the lemma of each word
    return [word.lemma for word in words]

#function to apply the count vectorizer(BOW) and TF-IDF transforms to a set of input features
def transf(mail):
    #CountVectorier is very costy, consider replace with cached vectors in the future
    bow_transformer = CountVectorizer(analyzer=split_lemmas).fit(mail_train)
    messages_bow = bow_transformer.transform(mail)
    #apply the TF-IDF transform to the output of BOW
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    return messages_tfidf

#function which takes in y test value and y predicted value and prints the associated model performance metrics
def model_assessment(y_test,pred):
    print('confusion matrix:')
    print(confusion_matrix(y_test,pred))
    print('accuracy:')
    print(accuracy_score(y_test,pred))
    print('precision:')
    print(precision_score(y_test,pred,pos_label='spam'))
    print('recall:')
    print(recall_score(y_test,pred,pos_label='spam'))
    print('f-Score:')
    print(f1_score(y_test,pred,pos_label='spam'))
    print('AUC:')
    print(roc_auc_score(np.where(y_test=='spam',1,0),np.where(pred=='spam',1,0)))
    plt.matshow(confusion_matrix(y_test, pred), cmap=plt.cm.binary, interpolation='nearest')
    plt.title('confusion matrix')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')

#transform and build traing and testing features
x_train=transf(mail_train)
x_test=transf(mail_test)    

'''
KNN model
'''
start_time=time.time()
modelKNN=KNeighborsClassifier(n_neighbors=3)
modelKNN.fit(x_train,y_train)
y_pred=modelKNN.predict(x_test)
model_assessment(y_test,y_pred)
print(" KNN with k=3 training took %s seconds" %(time.time()-start_time))


'''
Naive Bayes Model
'''
start_time=time.time()
#create and fit NB model
modelNB=MultinomialNB()
modelNB.fit(x_train,y_train)
#NB predictions
pred_NB=modelNB.predict(x_test)
#assess NB
model_assessment(y_test,pred_NB)
print(" Naive training took %s seconds" %(time.time()-start_time))

'''
Decision Tree Model
'''
start_time=time.time()
#create and fit tree model
model_tree=DecisionTreeClassifier()
model_tree.fit(x_train,y_train)
#run model on test and print metrics
pred_tree=model_tree.predict(x_test)
model_assessment(y_test,pred_tree)
print(" Decision Tree training took %s seconds" %(time.time()-start_time))

'''
Support Vector Machine
'''
start_time=time.time()
#create and fit SVM model
model_svm=SVC()
model_svm.fit(x_train,y_train)
#run model on test and print metrics
pred_svm=model_svm.predict(x_test)
model_assessment(y_test,pred_svm)
print(" SVM training took %s seconds" %(time.time()-start_time))

'''
Random Forest
'''
start_time=time.time()
#create and fit model
model_rf=RandomForestClassifier(n_estimators=20,criterion='entropy')
model_rf.fit(x_train,y_train)
#run model on test and print metrics
pred_rf=model_rf.predict(x_test)
model_assessment(y_test,pred_rf)
print(" Random Forest training took %s seconds" %(time.time()-start_time))

'''
Logistic Regression
'''
start_time=time.time()
#create and fit model
model_lr=LogisticRegression(solver='liblinear', random_state=42)
model_lr.fit(x_train,y_train)
#run model on test and print metrics
pred_lr=model_rf.predict(x_test)
model_assessment(y_test,pred_rf)
print(" Logistic Regression with liblinear solver training took %s seconds" %(time.time()-start_time))

plt.show()
'''
Author:Kai Kang
Description:GUI element for the email spam detection program
'''
class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI,self).__init__()
        self.setGeometry(0,0,900,900)
        self.setWindowTitle("email spam detection")
        self.initUI()
    
    def initUI(self):
        #labels
        self.label1=QtWidgets.QLabel(self)
        self.label1.setText("step1")
        self.label1.move(0,10)
        self.label2=QtWidgets.QLabel(self)
        self.label2.setText("step2")
        self.label2.move(0,50)
        self.label3=QtWidgets.QLabel(self)
        self.label3.setText("step3")
        self.label3.move(0,90)
        self.prompt_label=QtWidgets.QLabel(self)
        self.prompt_label.setText("")
        self.prompt_label.move(500,500)
        self.text_box=QtWidgets.QTextBrowser(self)
        self.text_box.setText("hello world")
        self.text_box.move(100,500)
        #buttons
        self.b1=QtWidgets.QPushButton(self)
        self.b1.setGeometry(50,10,200,20)
        self.b1.setText("load training dataset")
        self.b1.clicked.connect(self.b1_onClick)

    def b1_onClick(self):
        self.text_box.setText("proceed")
        self.update()
        
    def update(self):
        self.text_box.adjustSize()




def window():
    app=QApplication(sys.argv)
    win=MyGUI()
    win.show()
    sys.exit(app.exec_())

    window()    


print('done')'''
Author:Kai Kang
Description:the algorithms for spam email detection

'''
'''
Author:Kai Kang
Description:simple file loader and text preprocessor
'''
#function to get email text from email body
def preprocess(msg): #getting plain text 'email body'
    body = None
    #single body email
    if msg.get_content_type() == 'text/plain':
        body = msg.get_payload(decode=True)   
    #multiboy email       
    elif msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        body = subpart.get_payload(decode=True)
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
    #return mail text which concatenates both mail subject and body
    mailcontent=str(msg['subject'])+" "+str(body)
    return mailcontent



print('done')'''
'''
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(280, 380, 161, 31))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load Training Data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

