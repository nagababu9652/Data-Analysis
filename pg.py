# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pg1.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(800, 600))
        MainWindow.setMouseTracking(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(10, 0, 781, 551))
        self.tableWidget.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.tableWidget.verticalHeader().setStretchLastSection(False)
        self.lab = QtWidgets.QLabel(self.centralwidget)
        self.lab.setGeometry(QtCore.QRect(210, 560, 281, 20))
        self.lab.setObjectName("lab")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuExplore = QtWidgets.QMenu(self.menubar)
        self.menuExplore.setObjectName("menuExplore")
        self.menuWrangling = QtWidgets.QMenu(self.menubar)
        self.menuWrangling.setObjectName("menuWrangling")
        self.menuPlot = QtWidgets.QMenu(self.menubar)
        self.menuPlot.setObjectName("menuPlot")
        self.menuModeling = QtWidgets.QMenu(self.menubar)
        self.menuModeling.setObjectName("menuModeling")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionPrint = QtWidgets.QAction(MainWindow)
        self.actionPrint.setObjectName("actionPrint")
        self.actionPrint_PreView = QtWidgets.QAction(MainWindow)
        self.actionPrint_PreView.setObjectName("actionPrint_PreView")
        self.actionClear = QtWidgets.QAction(MainWindow)
        self.actionClear.setObjectName("actionClear")
        self.actionStd = QtWidgets.QAction(MainWindow)
        self.actionStd.setObjectName("actionStd")
        self.actionVariance = QtWidgets.QAction(MainWindow)
        self.actionVariance.setObjectName("actionVariance")
        self.actionCorrelation = QtWidgets.QAction(MainWindow)
        self.actionCorrelation.setObjectName("actionCorrelation")
        self.actionDescribe = QtWidgets.QAction(MainWindow)
        self.actionDescribe.setObjectName("actionDescribe")
        self.actionCovarience = QtWidgets.QAction(MainWindow)
        self.actionCovarience.setObjectName("actionCovarience")
        self.actionSum_R = QtWidgets.QAction(MainWindow)
        self.actionSum_R.setObjectName("actionSum_R")
        self.actionSum_C = QtWidgets.QAction(MainWindow)
        self.actionSum_C.setObjectName("actionSum_C")
        self.actionMAD = QtWidgets.QAction(MainWindow)
        self.actionMAD.setObjectName("actionMAD")
        self.actionMean_R = QtWidgets.QAction(MainWindow)
        self.actionMean_R.setObjectName("actionMean_R")
        self.actionMean_C = QtWidgets.QAction(MainWindow)
        self.actionMean_C.setObjectName("actionMean_C")
        self.actionMin = QtWidgets.QAction(MainWindow)
        self.actionMin.setObjectName("actionMin")
        self.actionMax = QtWidgets.QAction(MainWindow)
        self.actionMax.setObjectName("actionMax")
        self.actionHead = QtWidgets.QAction(MainWindow)
        self.actionHead.setObjectName("actionHead")
        self.actionTail = QtWidgets.QAction(MainWindow)
        self.actionTail.setObjectName("actionTail")
        self.actionDrop_C = QtWidgets.QAction(MainWindow)
        self.actionDrop_C.setObjectName("actionDrop_C")
        self.actionDrop_R = QtWidgets.QAction(MainWindow)
        self.actionDrop_R.setObjectName("actionDrop_R")
        self.actionIsNull = QtWidgets.QAction(MainWindow)
        self.actionIsNull.setObjectName("actionIsNull")
        self.actionDropNa = QtWidgets.QAction(MainWindow)
        self.actionDropNa.setObjectName("actionDropNa")
        self.actionDropAllNa = QtWidgets.QAction(MainWindow)
        self.actionDropAllNa.setObjectName("actionDropAllNa")
        self.actionFillNa = QtWidgets.QAction(MainWindow)
        self.actionFillNa.setObjectName("actionFillNa")
        self.actionFillNaMin = QtWidgets.QAction(MainWindow)
        self.actionFillNaMin.setObjectName("actionFillNaMin")
        self.actionFillNaMean = QtWidgets.QAction(MainWindow)
        self.actionFillNaMean.setObjectName("actionFillNaMean")
        self.actionDropDuplicate = QtWidgets.QAction(MainWindow)
        self.actionDropDuplicate.setObjectName("actionDropDuplicate")
        self.actionInfo = QtWidgets.QAction(MainWindow)
        self.actionInfo.setObjectName("actionInfo")
        self.actionCategorical = QtWidgets.QAction(MainWindow)
        self.actionCategorical.setObjectName("actionCategorical")
        self.actionLine = QtWidgets.QAction(MainWindow)
        self.actionLine.setObjectName("actionLine")
        self.actionScatter = QtWidgets.QAction(MainWindow)
        self.actionScatter.setObjectName("actionScatter")
        self.actionBar = QtWidgets.QAction(MainWindow)
        self.actionBar.setObjectName("actionBar")
        self.actionBarH = QtWidgets.QAction(MainWindow)
        self.actionBarH.setObjectName("actionBarH")
        self.actionHist = QtWidgets.QAction(MainWindow)
        self.actionHist.setObjectName("actionHist")
        self.actionPie = QtWidgets.QAction(MainWindow)
        self.actionPie.setObjectName("actionPie")
        self.actionArea = QtWidgets.QAction(MainWindow)
        self.actionArea.setObjectName("actionArea")
        self.actionToNum = QtWidgets.QAction(MainWindow)
        self.actionToNum.setObjectName("actionToNum")
        self.actionLinear_Reg = QtWidgets.QAction(MainWindow)
        self.actionLinear_Reg.setObjectName("actionLinear_Reg")
        self.actionLogistic_Regression = QtWidgets.QAction(MainWindow)
        self.actionLogistic_Regression.setObjectName("actionLogistic_Regression")
        self.actionDecision_Tree = QtWidgets.QAction(MainWindow)
        self.actionDecision_Tree.setObjectName("actionDecision_Tree")
        self.actionNaive_Bayes = QtWidgets.QAction(MainWindow)
        self.actionNaive_Bayes.setObjectName("actionNaive_Bayes")
        self.actionKNN = QtWidgets.QAction(MainWindow)
        self.actionKNN.setObjectName("actionKNN")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionPrint)
        self.menuFile.addAction(self.actionPrint_PreView)
        self.menuFile.addAction(self.actionClear)
        self.menuExplore.addAction(self.actionStd)
        self.menuExplore.addAction(self.actionVariance)
        self.menuExplore.addAction(self.actionCorrelation)
        self.menuExplore.addAction(self.actionDescribe)
        self.menuExplore.addAction(self.actionCovarience)
        self.menuExplore.addAction(self.actionSum_R)
        self.menuExplore.addAction(self.actionSum_C)
        self.menuExplore.addAction(self.actionMAD)
        self.menuExplore.addAction(self.actionMean_R)
        self.menuExplore.addAction(self.actionMean_C)
        self.menuExplore.addAction(self.actionIsNull)
        self.menuExplore.addAction(self.actionInfo)
        self.menuExplore.addAction(self.actionMin)
        self.menuExplore.addAction(self.actionMax)
        self.menuExplore.addAction(self.actionHead)
        self.menuExplore.addAction(self.actionTail)
        self.menuWrangling.addAction(self.actionDrop_C)
        self.menuWrangling.addAction(self.actionDrop_R)
        self.menuWrangling.addAction(self.actionDropNa)
        self.menuWrangling.addAction(self.actionDropAllNa)
        self.menuWrangling.addSeparator()
        self.menuWrangling.addSeparator()
        self.menuWrangling.addAction(self.actionFillNa)
        self.menuWrangling.addAction(self.actionFillNaMin)
        self.menuWrangling.addAction(self.actionFillNaMean)
        self.menuWrangling.addSeparator()
        self.menuWrangling.addAction(self.actionDropDuplicate)
        self.menuWrangling.addAction(self.actionCategorical)
        self.menuWrangling.addAction(self.actionToNum)
        self.menuPlot.addAction(self.actionLine)
        self.menuPlot.addAction(self.actionScatter)
        self.menuPlot.addAction(self.actionBar)
        self.menuPlot.addAction(self.actionBarH)
        self.menuPlot.addAction(self.actionHist)
        self.menuPlot.addAction(self.actionPie)
        self.menuModeling.addAction(self.actionLinear_Reg)
        self.menuModeling.addAction(self.actionLogistic_Regression)
        self.menuModeling.addAction(self.actionDecision_Tree)
        self.menuModeling.addAction(self.actionNaive_Bayes)
        self.menuModeling.addAction(self.actionKNN)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuExplore.menuAction())
        self.menubar.addAction(self.menuWrangling.menuAction())
        self.menubar.addAction(self.menuPlot.menuAction())
        self.menubar.addAction(self.menuModeling.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lab.setText(_translate("MainWindow", "TextLabel"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuExplore.setTitle(_translate("MainWindow", "Explore"))
        self.menuWrangling.setTitle(_translate("MainWindow", "Wrangling"))
        self.menuPlot.setTitle(_translate("MainWindow", "Plot"))
        self.menuModeling.setTitle(_translate("MainWindow", "Modeling"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setToolTip(_translate("MainWindow", "Open File "))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setToolTip(_translate("MainWindow", "Save File"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionPrint.setText(_translate("MainWindow", "Print"))
        self.actionPrint.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionPrint_PreView.setText(_translate("MainWindow", "Print PreView"))
        self.actionPrint_PreView.setShortcut(_translate("MainWindow", "Ctrl+Shift+P"))
        self.actionClear.setText(_translate("MainWindow", "Quit"))
        self.actionClear.setShortcut(_translate("MainWindow", "Ctrl+F4"))
        self.actionStd.setText(_translate("MainWindow", "Std"))
        self.actionStd.setToolTip(_translate("MainWindow", "Standered deviation"))
        self.actionStd.setShortcut(_translate("MainWindow", "Alt+S"))
        self.actionVariance.setText(_translate("MainWindow", "Variance"))
        self.actionVariance.setShortcut(_translate("MainWindow", "Alt+V"))
        self.actionCorrelation.setText(_translate("MainWindow", "Correlation"))
        self.actionCorrelation.setShortcut(_translate("MainWindow", "Alt+C"))
        self.actionDescribe.setText(_translate("MainWindow", "Describe"))
        self.actionDescribe.setShortcut(_translate("MainWindow", "Alt+D"))
        self.actionCovarience.setText(_translate("MainWindow", "Covarience"))
        self.actionCovarience.setShortcut(_translate("MainWindow", "Alt+O"))
        self.actionSum_R.setText(_translate("MainWindow", "Sum_R"))
        self.actionSum_R.setToolTip(_translate("MainWindow", "Sum Along Row"))
        self.actionSum_C.setText(_translate("MainWindow", "Sum_C"))
        self.actionSum_C.setToolTip(_translate("MainWindow", "Sum Along Column"))
        self.actionMAD.setText(_translate("MainWindow", "MAD"))
        self.actionMean_R.setText(_translate("MainWindow", "Mean_R"))
        self.actionMean_C.setText(_translate("MainWindow", "Mean_C"))
        self.actionMin.setText(_translate("MainWindow", "Min"))
        self.actionMax.setText(_translate("MainWindow", "Max"))
        self.actionHead.setText(_translate("MainWindow", "Head"))
        self.actionTail.setText(_translate("MainWindow", "Tail"))
        self.actionDrop_C.setText(_translate("MainWindow", "Drop_C"))
        self.actionDrop_R.setText(_translate("MainWindow", "Drop_R"))
        self.actionIsNull.setText(_translate("MainWindow", "IsNull"))
        self.actionDropNa.setText(_translate("MainWindow", "DropNa"))
        self.actionDropAllNa.setText(_translate("MainWindow", "DropAllNa"))
        self.actionFillNa.setText(_translate("MainWindow", "FillNa"))
        self.actionFillNa.setShortcut(_translate("MainWindow", "Alt+F"))
        self.actionFillNaMin.setText(_translate("MainWindow", "FillNaMin"))
        self.actionFillNaMean.setText(_translate("MainWindow", "FillNaMean"))
        self.actionDropDuplicate.setText(_translate("MainWindow", "DropDuplicate"))
        self.actionInfo.setText(_translate("MainWindow", "Info"))
        self.actionInfo.setShortcut(_translate("MainWindow", "Alt+I"))
        self.actionCategorical.setText(_translate("MainWindow", "Categorical"))
        self.actionLine.setText(_translate("MainWindow", "Line"))
        self.actionLine.setShortcut(_translate("MainWindow", "Alt+Shift+L"))
        self.actionScatter.setText(_translate("MainWindow", "Scatter"))
        self.actionScatter.setShortcut(_translate("MainWindow", "Alt+Shift+S"))
        self.actionBar.setText(_translate("MainWindow", "Bar"))
        self.actionBar.setShortcut(_translate("MainWindow", "Alt+Shift+B"))
        self.actionBarH.setText(_translate("MainWindow", "BarH"))
        self.actionBarH.setShortcut(_translate("MainWindow", "Shift+B"))
        self.actionHist.setText(_translate("MainWindow", "Hist"))
        self.actionHist.setShortcut(_translate("MainWindow", "Alt+Shift+H"))
        self.actionPie.setText(_translate("MainWindow", "Pie"))
        self.actionPie.setShortcut(_translate("MainWindow", "Alt+Shift+P"))
        self.actionArea.setText(_translate("MainWindow", "Area"))
        self.actionArea.setShortcut(_translate("MainWindow", "Alt+Shift+A"))
        self.actionToNum.setText(_translate("MainWindow", "ToNum"))
        self.actionLinear_Reg.setText(_translate("MainWindow", "Linear Regression"))
        self.actionLinear_Reg.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionLogistic_Regression.setText(_translate("MainWindow", "Logistic Regression"))
        self.actionLogistic_Regression.setShortcut(_translate("MainWindow", "Shift+L"))
        self.actionDecision_Tree.setText(_translate("MainWindow", "Decision Tree"))
        self.actionDecision_Tree.setShortcut(_translate("MainWindow", "Ctrl+D"))
        self.actionNaive_Bayes.setText(_translate("MainWindow", "Naive Bayes"))
        self.actionNaive_Bayes.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionKNN.setText(_translate("MainWindow", "KNN"))
        self.actionKNN.setShortcut(_translate("MainWindow", "Ctrl+K"))
