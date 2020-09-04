# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, sys, time, datetime, inspect, numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QApplication, QWidget, QAction
from PyQt5.QtCore import QSettings, QPoint, QSize, Qt, QModelIndex, QThread, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QCursor, QPixmap, QImage

from rsc.my_qt import *
from QtImageViewer import *

from Common import *
from LineExtractor import *

class QtLineExtractor(QtWidgets.QMainWindow, Common ):

    paintUiSignal = pyqtSignal()
    plotImageSignal = pyqtSignal( dict )

    def __init__(self):
        QtWidgets.QMainWindow.__init__( self )
        Common.__init__(self)

        uic.loadUi('./QtLineExtractor.ui', self)

        self.lineExtractThread = None
        self.isProcessing = False
        self.statusMessage = None

        self.duration = 0

        self.settings = QSettings('TheOneTech', 'line_extractor')

        self.imageViewers = []

        self.init_tab( self.tabWidgetLeft )
        self.init_tab( self.tabWidgetRight )

        self.progressBar.setValue( 0 )
        self.progressBar.setEnabled( 0 )

        self.lineMatchComboBox.addItems( [ "Matched", "All", "A Only", "B Only"] )

        if 1 :
            headerLabels = [ '그림', 'ID', '유사도', '길이', '좌표(A)', '좌표(B)' ]
            colLen = len(headerLabels)

            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(headerLabels)

            tableView = self.lineTableView
            tableView.setModel(model)

            header = tableView.horizontalHeader()
            header.setDefaultAlignment(Qt.AlignHCenter)

            for i in range( colLen ) :
                #header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
                header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            pass

            # Sets different alignment data just on the first column
            #model.setHeaderData(0, Qt.Horizontal, Qt.AlignJustify, Qt.TextAlignmentRole )
            #model.setHeaderData(0, Qt.Horizontal, Qt.AlignJustify, Qt.TextAlignmentRole)

            for row in range( 4 ):
                items = []
                for col in range( colLen ):
                    data = f"{(row + 1) * (col + 1)}"

                    item = QStandardItem( data )
                    item.setEditable( False )

                    items.append( item )
                pass

                model.insertRow( row, items )
            pass

            tableView.setWindowTitle( "Lines Extracted" )
            tableView.resizeColumnsToContents()
        pass

        # signal -> slot connect
        self.tabWidgetLeft.currentChanged.connect( self.when_tab_widget_current_changed )
        self.exitBtn.clicked.connect( self.when_exitBtn_clicked )
        self.actionExit.triggered.connect(self.close_app)
        self.actionOpen.triggered.connect( self.when_openBtn_clicked )
        self.openBtn.clicked.connect( self.when_openBtn_clicked )
        self.lineExtract.clicked.connect( self.when_lineExtract_clicked )
        self.actionFull_Screen.triggered.connect( self.when_fullScreen_clicked )
        self.prevFileOpen.clicked.connect( self.when_prevFileOpen_clicked )
        self.nextFileOpen.clicked.connect(self.when_nextFileOpen_clicked)

        self.paintUiSignal.connect( self.when_paintUiSignal )
        self.plotImageSignal.connect( self.when_plotImageSignal )
        # -- signal -> slot connect

        self.buildOpenRecentFilesMenuBar()
        self.startEvent()

        self.paintUi()
    pass # -- __init__

    def plot_image(self, image, mode="A", title="", border_color="black" ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        params = {"image": image, "mode": mode, "title": title, "border_color": border_color}

        use_signal = True
        if use_signal :
            self.plotImageSignal.emit( params )
        else :
            self.when_plotImageSignal( params )
        pass
    pass

    def when_plotImageSignal(self, params ) :
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        image = params["image"]
        mode = params["mode"]
        title = params["title"]
        border_color = params["border_color"]

        imageViewers = self.imageViewers
        imageViewer = imageViewers[0] if mode == 'A' else imageViewers[1]

        imageViewer.messageLineEdit.setText( title )

        loadImageFromFile = True
        if loadImageFromFile :
            imageViewer.loadImageFromFile( image.fileName )
        else :
            imageViewer.setImage( self.converToQImage( image ))
        pass

        statusBar = self.statusBar()
        statusBar.showMessage( title )
    pass # -- when_plotImageSignal

    def converToQImage(self, image):
        img = image.img
        img = img.astype(np.uint8)
        shape = img.shape

        log.info(f"shape={img.shape}")

        height = len(img)
        width = len(img[0])

        channel = 1
        fmt = QImage.Format_Indexed8

        if len(shape) > 2:
            channel = shape[2]
            fmt = QImage.Format_RGB888
        pass

        bytesPerLine = channel * width

        qImage = QImage(img, width, height, bytesPerLine, fmt)

        return qImage
    pass # -- converToQImage

    def plot_histogram(self, image, mode="A"):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        # 히스토 그램 표출
        pass
    pass # -- plot_histogram

    def paintUi(self):
        self.paintUiSignal.emit()
    pass # -- paintUi

    def when_paintUiSignal(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        isProcessing = self.isProcessing
        if isProcessing :
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        else :
            QApplication.restoreOverrideCursor()
        pass

        is_file_open = False

        prevFile = None
        nextFile = None

        imageViewers = self.imageViewers

        if imageViewers and not imageViewers[0].isEmpty :
            is_file_open = True

            fileName = imageViewers[0].fileName
            if fileName :
                prevFile = self.prev_file( fileName )
                nextFile = self.next_file( fileName )
            pass
        pass

        log.info( f"prevFile={prevFile}")
        log.info( f"nextFile={nextFile}")

        self.prevFileOpen.setEnabled( prevFile is not None )
        self.nextFileOpen.setEnabled( nextFile is not None )
        self.lineExtract.setEnabled( is_file_open )
        self.saveJsonFile.setEnabled(is_file_open)

        if 1 :
            duration = int( self.duration )
            self.durationLcdNumber.display( str(datetime.timedelta(seconds=duration)) )
        pass

        if self.statusMessage :
            self.statusBar().showMessage( self.statusMessage )
        pass

    pass # -- when_paintUiSignal

    def when_nextFileOpen_clicked(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        imageViewers = self.imageViewers

        if imageViewers :
            imageViewer = imageViewers[0]
            if imageViewer is not None and not imageViewer.isEmpty and imageViewer.fileName :
                fileName = imageViewer.fileName
                nextFile = self.next_file( fileName )
                log.info( f"nextFile = {nextFile}" )
                if nextFile :
                    self.when_openBtn_clicked( fileName = nextFile )
                    self.statusMessage = f"Next File [ {nextFile} ] is opened."

                    self.paintUi()
                pass
            pass
        pass

    pass # -- when_nextFileOpen_clicked

    def when_prevFileOpen_clicked(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        imageViewers = self.imageViewers

        if imageViewers :
            imageViewer = imageViewers[0]
            if imageViewer is not None and not imageViewer.isEmpty and imageViewer.fileName :
                fileName = imageViewer.fileName
                prevFile = self.prev_file( fileName )
                log.info( f"nextFile = {prevFile}" )
                if prevFile :
                    self.when_openBtn_clicked( fileName = prevFile )
                    self.statusBar().showMessage( f"Previous File [ {prevFile} ] is opened.")
                pass
            pass
        pass

    pass # -- when_prevFileOpen_clicked

    def resizeEvent(self, event):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        action = self.actionFull_Screen

        state = self.windowState()

        if state == Qt.WindowFullScreen :
            action.setText( "Exit full screen" )
        else :
            action.setText( "Full Screen" )
        pass
    pass # -- resizeEvent

    def keyPressEvent(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        state = self.windowState()

        key = e.key()

        log.info( f"key = {key}" )

        if key == QtCore.Qt.Key_Escape and state == Qt.WindowFullScreen :
            self.showNormal()
        elif key == QtCore.Qt.Key_F11:
            if self.isFullScreen() :
                self.showNormal()
            else:
                self.showFullScreen()
            pass
        pass
    pass # -- keyPressEvent

    def when_fullScreen_clicked(self, e):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        if self.isFullScreen() :
            self.showNormal()
        else :
            self.showFullScreen()
        pass

    pass # -- when_fullScreen_clicked

    def buildOpenRecentFilesMenuBar(self):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        menuOpen_Recent = self.menuOpen_Recent

        settings = self.settings

        recent_file_list = settings.value( 'recent_file_list', [], str)

        menuOpen_Recent.setEnabled( len( recent_file_list ) > 0 )

        menuOpen_Recent.clear()

        for i, fileName in enumerate( recent_file_list ):
            fileName = fileName.replace ( "\\", "/" )
            log.info( f"fileName = {fileName}" )
            action = QAction( f"[{(i+1):d}] {fileName}", menuOpen_Recent )
            action.fileName = fileName

            if i == 0 :
                action.setShortcut('Ctrl+R')
            pass

            menuOpen_Recent.addAction( action )

            action.triggered.connect( lambda val : self.when_recentFileAction_clicked( f"{fileName}" ) )
        pass

        if len( recent_file_list ) :
            menuOpen_Recent.addSeparator()

            actionClear = QAction("Clear Recently Opened", menuOpen_Recent )
            menuOpen_Recent.addAction(actionClear)

            actionClear.triggered.connect(self.when_clear_recently_opened_clicked)
            actionClear.setEnabled(len(recent_file_list) > 0)
        pass

    pass # -- buildOpenRecentFilesMenuBar

    def when_clear_recently_opened_clicked(self, e):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        settings = self.settings
        settings.setValue("recent_file_list", [])

        self.buildOpenRecentFilesMenuBar()
    pass

    def when_recentFileAction_clicked(self, fileName ):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        self.when_openBtn_clicked( e=None, fileName=fileName )
    pass # -- when_recentFileAction_clicked

    def when_lineExtract_clicked(self, e):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        class LineExtractThread(QThread):
            def __init__(self, qtUi : QtLineExtractor ):
                QThread.__init__(self)
                self.qtUi = qtUi
            pass

            def __del__(self):
                self.wait()

            pass

            def run(self):
                qtUi = self.qtUi

                qtUi.isProcessing = True

                qtUi.paintUi()

                imageViewers = qtUi.imageViewers

                start = time.time()

                qtUi.duration = 0

                for i, imageViewer in enumerate(imageViewers):
                    lineExtractor = LineExtractor()

                    img_path = imageViewer.fileName

                    mode = chr(ord('A') + i)

                    log.info(f"img_path={img_path}, mode={mode}")

                    lineExtractor.my_line_extract(img_path=img_path, qtUi=qtUi, mode=mode)

                    qtUi.duration = time.time() - start

                    open_folder = False
                    if open_folder and i == len(imageViewers) - 1:
                        # 결과창 폴더 열기
                        folder = "c:/temp"
                        lineExtractor.open_file_or_folder(folder)
                    pass

                    qtUi.paintUi()
                pass

                qtUi.isProcessing = False
                qtUi.lineExtractThread = None

                qtUi.paintUi()
            pass

        pass  # LineExtractThread

        if self.lineExtractThread is not None :
            log.info( "There is another thread running." )
        else :
            self.lineExtractThread = LineExtractThread( self )
            self.lineExtractThread.start()
        pass

    pass # -- when_lineExtract_clicked

    def when_openBtn_clicked(self, e = None, fileName="" ):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )
        debug = False

        imageViewers = self.imageViewers

        if imageViewers and imageViewers[0]:
            fileName = imageViewers[0].loadImageFromFile(fileName=fileName, setFileName=True)

            if fileName:
                self.save_recent_file(self.settings, fileName)
                self.buildOpenRecentFilesMenuBar()

                file_next = self.next_file( fileName )

                if file_next and imageViewers[1] :
                    imageViewers[1].loadImageFromFile(fileName=file_next, setFileName=True)
                pass
            pass
        pass

        self.duration = 0
        self.statusMessage = f"File [ {fileName} ] is opened."

        self.paintUi()
    pass # -- when_openBtn_clicked

    def startEvent(self):
        settings = self.settings

        # 마지막 윈도우 크기 및 위치 로딩
        self.resize(self.settings.value("size", QSize(840, 640)))
        self.move(self.settings.value("pos", QPoint(50, 50)))
    pass # -- startEvent

    def showEvent(self, e ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)

        return QtWidgets.QMainWindow.showEvent(self, e)
    pass # -- showEvent

    def init_tab(self, tabWidget):
        settings = self.settings

        tabWidget.setCurrentIndex(0)

        tab = tabWidget.widget( 0 )

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setContentsMargins( 2, 2, 2, 2 )

        row = 0
        col = 0

        imageViewer = QtImageViewer(settings = settings)
        self.imageViewers.append(imageViewer)

        labelWidth = 55

        if 1 :
            fileNameLineEdit = QtWidgets.QLineEdit()
            fileNameLineEdit.setReadOnly( 1 )

            imageViewer.fileNameLineEdit = fileNameLineEdit

            label = QtWidgets.QLabel( "  File: ")
            label.setMinimumWidth( labelWidth )
            label.setMaximumWidth( labelWidth )

            layout = QtWidgets.QHBoxLayout()
            layout.addWidget( label )
            layout.addWidget( fileNameLineEdit )

            gridLayout.addLayout( layout, row, col)
        pass

        if 1 :
            row += 1
            messageLineEdit = QtWidgets.QLineEdit()
            messageLineEdit.setReadOnly(1)

            imageViewer.messageLineEdit = messageLineEdit

            label = QtWidgets.QLabel("  Message: ")
            label.setMinimumWidth(labelWidth)
            label.setMaximumWidth(labelWidth)

            layout = QtWidgets.QHBoxLayout()
            layout.addWidget( label )
            layout.addWidget(messageLineEdit)

            gridLayout.addLayout(layout, row, col)
        pass

        row += 1
        gridLayout.addWidget( imageViewer, row, col )

        if 1 :
            horizontal = QtWidgets.QHBoxLayout()

            zoomSlider = QtWidgets.QSlider( Qt.Horizontal )
            zoomSlider.setMinimum( 10 )
            zoomSlider.setMaximum( 400 )
            zoomSlider.setValue( 100 )

            zoomText = QtWidgets.QLineEdit( "100%" )
            zoomText.setReadOnly(1)
            zoomText.setFixedWidth(40)
            zoomText.setAlignment( Qt.AlignRight )

            fullExt = QtWidgets.QPushButton( "Full Extent")

            horizontal.addWidget( QtWidgets.QLabel( " Zoom ") )
            horizontal.addWidget( zoomSlider )
            horizontal.addWidget( zoomText )
            horizontal.addWidget( fullExt )
            spacer = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            horizontal.addItem( spacer )

            row += 1
            gridLayout.addLayout( horizontal, row, col )
        pass

        tab.setLayout( gridLayout )

    pass #-- init_tab

    def when_exitBtn_clicked(self, e):
        log.info( "when_exitBtn_clicked" )

        self.close()
    pass # -- when_exitBtn_clicked

    def closeEvent(self, e):
        log.info( "closeEvent")
        # 마지막 윈도우 크기 및 위치 저장 
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())

        e.accept()
    pass # -- closeEvent

    def close_app( self ):
        log.info( "close app" )
        self.hide()
        sys.exit()
    pass # -- close_app

    def when_tab_widget_current_changed(self, index):
        log.info("when_tabWidget_currentChanged")
        pass
    pass # -- when_tab_widget_current_changed

    def show_image_on_image_viewer(self , image_viewer, image_path ):
        size = image_viewer.geometry()
        w = size.width()
        h = size.height()

        image_profile = QtGui.QImage(image_path)  # QImage object
        image_profile = image_profile.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation)
        # To scale image for example and keep its Aspect Ration
        image_viewer.setPixmap(QtGui.QPixmap.fromImage(image_profile))
    pass # -- show_image_on_image_viewer
pass

if __name__ == '__main__':
    log.info( f"Pwd 1: {os.getcwd()}" )

    # change working dir to current file
    dir_name = os.path.dirname(__file__)
    if dir_name :
        log.info( f"dir name: {dir_name}" )
        os.chdir(dir_name)
        log.info( f"Pwd 2: {os.getcwd()}" )
    pass

    # Create an instance of QtWidgets.QApplication
    app = QtWidgets.QApplication(sys.argv)
    window = QtLineExtractor() # Create an instance of our class
    window.show()
    app.exec_() # Start the application
pass

