import sys
import numpy
import os
import io
import csv, codecs
from PyQt5.QtWidgets import QMainWindow, QTableView,QStatusBar, QInputDialog, QLineEdit, QApplication,QMessageBox, QAction, QFileDialog, QTableWidgetItem
from PyQt5 import QtGui, QtCore, QtWidgets, QtPrintSupport
from PyQt5.QtCore import Qt, QAbstractTableModel, QFile, QTimer ,QThread
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog, QPrintPreviewDialog
from pg import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report, f1_score, accuracy_score, r2_score
from sklearn.preprocessing import  StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import time
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph



        
class MyForm(QMainWindow):

    def __init__(self,flag):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.lab.setText("Status Bar : Not Ready")
        self.flag=flag
        self.ui.statusbar.addWidget(self.ui.lab)
        
        self.ui.actionOpen.triggered.connect(self.openFileD)
        self.ui.actionSave.triggered.connect(self.saveFileD)
        self.ui.actionPrint.triggered.connect(self.printFileD)
        self.ui.actionPrint_PreView.triggered.connect(self.printPreviewD)
        self.ui.actionClear.triggered.connect(self.clearD)
        self.ui.actionStd.triggered.connect(self.Std)
        self.ui.actionVariance.triggered.connect(self.Variance)
        self.ui.actionDescribe.triggered.connect(self.DescribeD)
        self.ui.actionCovarience.triggered.connect(self.Covarience)
        self.ui.actionSum_R.triggered.connect(self.Sum_R)
        self.ui.actionSum_C.triggered.connect(self.Sum_C)
        self.ui.actionMAD.triggered.connect(self.MAD)
        self.ui.actionMean_R.triggered.connect(self.Mean_R)
        self.ui.actionMean_C.triggered.connect(self.Mean_C)
        self.ui.actionMin.triggered.connect(self.MinD)
        self.ui.actionMax.triggered.connect(self.MaxD)
        self.ui.actionHead.triggered.connect(self.HeadD)
        self.ui.actionTail.triggered.connect(self.TailD)
        self.ui.actionDrop_C.triggered.connect(self.DropC)
        self.ui.actionDrop_R.triggered.connect(self.DropR)
        self.ui.actionIsNull.triggered.connect(self.IsNull)
        self.ui.actionDropNa.triggered.connect(self.DropNa)
        self.ui.actionDropAllNa.triggered.connect(self.DropAllNa)
        self.ui.actionFillNa.triggered.connect(self.FillNa)
        self.ui.actionFillNaMin.triggered.connect(self.FillNaMin)
        self.ui.actionFillNaMean.triggered.connect(self.FillNaMean)
        self.ui.actionDropDuplicate.triggered.connect(self.DropDuplicate)
        self.ui.actionInfo.triggered.connect(self.Info)
        self.ui.actionCategorical.triggered.connect(self.CategoricalD)
        self.ui.actionToNum.triggered.connect(self.ToNum)
        self.ui.actionLine.triggered.connect(self.LinePlot)
        self.ui.actionScatter.triggered.connect(self.Scatter)
        self.ui.actionBar.triggered.connect(self.Bar)
        self.ui.actionBarH.triggered.connect(self.BarH)
        self.ui.actionHist.triggered.connect(self.Hist)
        self.ui.actionPie.triggered.connect(self.Pie)
        self.ui.actionLinear_Reg.triggered.connect(self.Linear)
        self.ui.actionLogistic_Regression.triggered.connect(self.Logistic)
        self.ui.actionDecision_Tree.triggered.connect(self.Decision)
        self.ui.actionNaive_Bayes.triggered.connect(self.Bayes)
        self.ui.actionKNN.triggered.connect(self.KNN)
        
                                                       
            

        
    def openFileD(self):
        self.flag=True
        fileName, _= QFileDialog.getOpenFileName(self, 'Open file','/home')
        global df
        df = pd.read_csv(fileName,low_memory=False,error_bad_lines=False,encoding='iso-8859-1')
        self.Construct()

    def worker(self):
        
        count_row=df.shape[0]
        count_col=df.shape[1]
        self.ui.tableWidget.setRowCount(count_row)
        self.ui.tableWidget.setColumnCount(count_col)
        listdf=df.to_numpy().tolist()

        row=0

        for tup in listdf:
            col=0
            for item in tup:
                oneitem=QTableWidgetItem(str(item))
                self.ui.tableWidget.setItem(row,col,oneitem)
                col+=1
            row+=1
        hl=df.columns.values.tolist()
        self.ui.tableWidget.setHorizontalHeaderLabels(hl)
        self.ui.lab.setText("Status Bar : Ready")

    def Construct(self):
            self.ui.lab.setText("Status Bar :Not Ready")
            thread = threading.Thread(target=self.worker)
            thread.start()
            thread.join()
                
        

    def saveFileD(self):
        if self.flag==True:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileNam, _ = QFileDialog.getSaveFileName(self, "save file","","All Files (*)", options=options)
            df.to_csv(fileNam)


    def printFileD(self):
        if self.flag==True:
        
            doc=QtGui.QTextDocument(df.to_string())
            printer=QPrinter(QPrinter.HighResolution)
            dialog=QPrintDialog(printer, self)
            if dialog.exec_()==QPrintDialog.Accepted:
                doc.print_(printer)

    def printPreviewD(self):
        if self.flag==True:
            printer=QPrinter(QPrinter.HighResolution)
            previewDialog=QPrintPreviewDialog(printer,self)
            previewDialog.paintRequested.connect(self.printPreview)
            previewDialog.exec_()

    def printPreview(self,printer):
        if self.flag==True:
            doc=QtGui.QTextDocument(df.to_string())
            doc.print_(printer)


    def clearD(self):
        exit()
        
        
    def Std(self):
        if self.flag==True:
            QMessageBox.about(self,'standerd deviation',str(df.std()))


    def Variance(self):
        if self.flag==True:
            QMessageBox.about(self,'Variance',str(df.var()))


    def DescribeD(self):
        if self.flag==True:
            QMessageBox.about(self,'Describetion',str(df.describe()))

    def Covarience(self):
        if self.flag==True:
            QMessageBox.about(self,'Covarience',str(df.cov()))


    def Sum_R(self):
        if self.flag==True:
            QMessageBox.about(self,'Sum along Rows',str(df.sum(axis='columns')))
        

    def Sum_C(self):
        if self.flag==True:
            QMessageBox.about(self,'Sum along Columns',str(df.sum()))


    def MAD(self):
        if self.flag==True:
            QMessageBox.about(self,'Mean Absolute Deviation',str(df.mad()))


    def Mean_R(self):
        if self.flag==True:
            QMessageBox.about(self,'Mean along Rows',str(df.mean(axis='columns')))


    def Mean_C(self):
        if self.flag==True:
            QMessageBox.about(self,'Mean along Columns',str(df.mean()))


    def MinD(self):
        if self.flag==True:
            QMessageBox.about(self,'Minimum',str(df.min()))


    def MaxD(self):
        if self.flag==True:
            QMessageBox.about(self,'Maximum',str(df.max()))


    def HeadD(self):
        if self.flag==True:
            tx,ty =QInputDialog.getInt(self,'Input Dialog','Enter Number Of Rows')
            QMessageBox.about(self,'Head',str(df.head(tx)))


    def TailD(self):
        if self.flag==True:
            tx,ty =QInputDialog.getInt(self,'Input Dialog','Enter Number Of Rows')
            QMessageBox.about(self,'Tail',str(df.tail(tx)))



    def DropC(self):
        if self.flag==True:
            tx,ty =QInputDialog.getInt(self,'Input Dialog','Enter column Number')
            df.drop(df.columns[tx-1],axis=1,inplace=True)
            self.Construct()

    def DropR(self):
        if self.flag==True:
            tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter column Number')
            df.drop(df.index[tx-1],inplace=True)
            self.Construct()

    def IsNull(self):
        if self.flag==True:
            QMessageBox.about(self,'IsNull',str(df.isnull()))

    def DropNa(self):
        if self.flag==True:
            df.dropna(inplace=True)
            self.Construct()

    def DropAllNa(self):
        if self.flag==True:
            df.dropna(how='all',inplace=True)
            self.Construct()

    def FillNa(self):
        if self.flag==True:
            df.fillna(0,inplace=True)
            self.Construct()

    def FillNaMin(self):
        if self.flag==True:
            df.fillna(method='ffill',inplace=True)
            self.Construct()

    def FillNaMean(self):
        if self.flag==True:
            df.fillna(df.mean(),inplace=True)
            self.Construct()

    def DropDuplicate(self):
        if self.flag==True:
            df.drop_duplicates(inplace=True)
            self.Construct()

    def Info(self):
        if self.flag==True:
            buffer=io.StringIO()
            df.info(buf=buffer)
            s=buffer.getvalue()
            QMessageBox.about(self,'Info About Null Values',s)

    def CategoricalD(self):
        if self.flag==True:
            tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter column Number')
            df[df.columns[tx-1]]=df[df.columns[tx-1]].astype('category')
            self.Construct()

    def ToNum(self):
        if self.flag==True:
            tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter column Number')
            num=LabelEncoder()
            df[df.columns[tx-1]]=num.fit_transform(df[df.columns[tx-1]])
            self.Construct()

    def LinePlot(self):
        if self.flag==True:
            x,ok=QInputDialog.getText(self,'Input Dialog','Enter column Numbers seperated with , ',QLineEdit.Normal,"")
            li=list()
            lis=x.split(",")
            
            for i in lis:
                li.append(int(i))
            if len(li)==1:
                plt.plot(df[df.columns[li[0]]])
                plt.show()
            elif len(li)==2:
                plt.plot(df[df.columns[li[0]]],df[df.columns[li[1]]])
                plt.show()
            elif len(li)==3:
                plt.plot(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]])
                plt.show()
            elif len(li)==4:
                plt.plot(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]])
                plt.show()
            elif len(li)==5:
                plt.plot(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]],df[df.columns[li[4]]])
                plt.show()
            else:
                QMessageBox.about(self,'Worning!','You Entered More columns than 5')
            
        

    def Scatter(self):
        x,ok=QInputDialog.getText(self,'Input Dialog','Enter column Numbers seperated by comma (,)',QLineEdit.Normal,"")
        li=list()
        lis=x.split(",")
        
        for i in lis:
            li.append(int(i))
        if len(li)>1:
            if len(li)==2:
                plt.scatter(df[df.columns[li[0]]],df[df.columns[li[1]]])
                plt.show()
            elif len(li)==3:
                plt.scatter(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]])
                plt.show()
            elif len(li)==4:
                plt.scatter(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]])
                plt.show()
            elif len(li)==5:
                plt.scatter(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]],df[df.columns[li[4]]])
                plt.show()
            else:
                QMessageBox.about(self,'Worning!','You Entered More columns than 5')

    def Bar(self):
        x,ok=QInputDialog.getText(self,'Input Dialog','Enter column Numbers seperated by comma (,)',QLineEdit.Normal,"")
        li=list()
        lis=x.split(",")
        
        for i in lis:
            li.append(int(i))
        if len(li)==1:
            plt.bar(df[df.columns[li[0]]])
            plt.show()
        elif len(li)==2:
            plt.bar(df[df.columns[li[0]]],df[df.columns[li[1]]])
            plt.show()
        elif len(li)==3:
            plt.bar(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]])
            plt.show()
        elif len(li)==4:
            plt.bar(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]])
            plt.show()
        elif len(li)==5:
            plt.bar(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]],df[df.columns[li[4]]])
            plt.show()
        else:
            QMessageBox.about(self,'Worning!','You Entered More columns than 5')

    def BarH(self):
        x,ok=QInputDialog.getText(self,'Input Dialog','Enter column Numbers seperated by comma (,)',QLineEdit.Normal,"")
        li=list()
        lis=x.split(",")
        
        for i in lis:
            li.append(int(i))
        if len(li)==1:
            plt.barh(df[df.columns[li[0]]])
            plt.show()
        elif len(li)==2:
            plt.barh(df[df.columns[li[0]]],df[df.columns[li[1]]])
            plt.show()
        elif len(li)==3:
            plt.barh(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]])
            plt.show()
        elif len(li)==4:
            plt.barh(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]])
            plt.show()
        elif len(li)==5:
            plt.barh(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]],df[df.columns[li[4]]])
            plt.show()
        else:
            QMessageBox.about(self,'Worning!','You Entered More columns than 5')

    def Hist(self):
        x,ok=QInputDialog.getText(self,'Input Dialog','Enter column Numbers seperated by comma (,) ',QLineEdit.Normal,"")
        li=list()
        lis=x.split(",")
        
        for i in lis:
            li.append(int(i))
        if len(li)==1:
            plt.hist(df[df.columns[li[0]]])
            plt.show()
        elif len(li)==2:
            plt.hist(df[df.columns[li[0]]],df[df.columns[li[1]]])
            plt.show()
        elif len(li)==3:
            plt.hist(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]])
            plt.show()
        elif len(li)==4:
            plt.hist(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]])
            plt.show()
        elif len(li)==5:
            plt.hist(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]],df[df.columns[li[4]]])
            plt.show()
        
        else:
            QMessageBox.about(self,'Worning!','You Entered More columns than 5')

    def Pie(self):
        x,ok=QInputDialog.getText(self,'Input Dialog','Enter column Numbers seperated by comma (,) ',QLineEdit.Normal,"")
        li=list()
        lis=x.split(",")
        
        for i in lis:
            li.append(int(i))
        if len(li)==1:
            plt.pie(df[df.columns[li[0]]])
            plt.show()
        elif len(li)==2:
            plt.pie(df[df.columns[li[0]]],df[df.columns[li[1]]])
            plt.show()
        elif len(li)==3:
            plt.pie(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]])
            plt.show()
        elif len(li)==4:
            plt.pie(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]])
            plt.show()
        elif len(li)==5:
            plt.pie(df[df.columns[li[0]]],df[df.columns[li[1]]],df[df.columns[li[2]]],df[df.columns[li[3]]],df[df.columns[li[4]]])
            plt.show()
        else:
            QMessageBox.about(self,'Worning!','You Entered More columns than 5')

    def Linear(self):
        tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter The Dependent column Number')
        x=df.drop(df.columns[tx],axis=1)
        y=df[df.columns[tx]]        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        s=StandardScaler()
        x_train=s.fit_transform(x_train)
        x_test=s.transform(x_test)
        reg=LinearRegression()
        reg.fit(x_train,y_train)
        y_pre=reg.predict(x_test)
        y_test=y_test.tolist()
        y_pre=y_pre.tolist()
        plt.plot(y_test, label='y_test')
        plt.plot(y_pre, label='y_predict')
        plt.legend()
        plt.show()
        r2=r2_score(y_test,y_pre)
        QMessageBox.about(self,'Accuracy By R2 Score',str(r2))

        


    def Logistic(self):
        tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter The Dependent column Number')
        x=df.drop(df.columns[tx],axis=1)
        y=df[df.columns[tx]]        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        s=StandardScaler()
        x_train=s.fit_transform(x_train)
        x_test=s.transform(x_test)
        log=LogisticRegression()
        log.fit(x_train,y_train)
        y_pre=log.predict(x_test)
        y_test=y_test.tolist()
        y_pre=y_pre.tolist()
        plt.plot(y_test, label='y_test')
        plt.plot(y_pre, label='y_predict')
        plt.legend()
        plt.show()
        QMessageBox.about(self,'Accuracy Score',str(log.score(x_test, y_test)))

    def Decision(self):
        tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter The Dependent column Number')
        x=df.drop(df.columns[tx],axis=1)
        y=df[df.columns[tx]]        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        dt=DecisionTreeClassifier()
        dt.fit(x_train,y_train)
        y_pre=dt.predict(x_test)
        y_test=y_test.tolist()
        y_pre=y_pre.tolist()
        QMessageBox.about(self,'Accuracy Score',str(dt.score(x_test, y_test)))
        
        

    def Bayes(self):
        tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter The Dependent column Number')
        y=df.drop(df.columns[tx],axis=1)
        x=df[df.columns[tx]]        
        x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=4)
        tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        x_trainFeat = tfvec.fit_transform(x_train)
        x_testFeat = tfvec.transform(x_test)
        # SVM is used to model
        y_trainSvm = y_train.astype('int')
        classifierModel = LinearSVC()
        classifierModel.fit(x_trainFeat, y_trainSvm)
        predResult = classifierModel.predict(x_testFeat)

        # GNB is used to model
        y_trainGnb = y_train.astype('int')
        classifierModel2 = MultinomialNB()
        classifierModel2.fit(x_trainFeat, y_trainGnb)
        predResult2 = classifierModel2.predict(x_testFeat)

        y_test = y_test.astype('int')
        actual_Y = y_test
        QMessageBox.about(self,"Accuracy Score using SVM: {0:.4f}",str(accuracy_score(actual_Y, predResult)))
        QMessageBox.about(self,'Accuracy Score using MNB: ',str(accuracy_score(actual_Y, predResult2)))
        cmMNb=confusion_matrix(actual_Y, predResult2)
        QMessageBox.about(self,'Confusion matrix using MNB:',str(cmMNb))

        
        
    def KNN(self):
        tx,ty=QInputDialog.getInt(self,'Input Dialog','Enter The Dependent column Number')
        x=df.drop(df.columns[tx],axis=1)
        y=df[df.columns[tx]]        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        k_range=range(1,10)
        scores_list=[]
        for k in k_range:
            knn=KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train,y_train)
            y_pre=knn.predict(x_test)
            scores_list.append(metrics.accuracy_score(y_test,y_pre))
        plt.plot(k_range,scores_list)
        plt.xlabel('value of k for knn')
        plt.ylabel('Testing Accuracy')
        plt.show()
        kno,knos=QInputDialog.getInt(self,'Input Dialog','What is the Value of K?')
        knn=KNeighborsClassifier(n_neighbors=kno)
        knn.fit(x_train,y_train)
        y_pre=knn.predict(x_test)
        QMessageBox.about(self,'Accuracy Score',str(metrics.accuracy_score(y_test,y_pre)))
        
        

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm(flag=False)
    w.setWindowTitle("Predictive Data Analysis")
    w.setWindowIcon(QtGui.QIcon('icon.png'))
    w.show()
    sys.exit(app.exec_())
