# -*- coding: utf-8 -*-
# QtImageViewer.py:

import sys, os.path

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import os, sys, inspect

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog

from rsc.my_qt import *

class QtImageViewer(QGraphicsView):

    def __init__(self, settings = None, dblClickFileLoad=False):
        QGraphicsView.__init__(self)

        if settings is None :
            settings = QSettings('TheOneTech', 'line_extractor')
        pass

        self.fileName = None

        self.dblClickFileLoad = dblClickFileLoad

        self.settings = settings

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Store a local handle to the scene's current image pixmap.
        self._pixmapHandle = None

        # Image aspect ratio mode.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio
        self.aspectRatioMode = Qt.KeepAspectRatioByExpanding

        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        scrollPolicy = Qt.ScrollBarAlwaysOn
        self.setHorizontalScrollBarPolicy(scrollPolicy)
        self.setVerticalScrollBarPolicy(scrollPolicy)

        # Stack of QRectF zoom boxes in scene coordinates.
        self.zoomStack = []

        self.isEmpty = True

        use_empty_icon = False
        if use_empty_icon :
            image = QPixmap(":/file/empty_grid.png")
            self.setImage( image )
        pass
    pass

    def resizeEvent(self, event):
        log.info( inspect.getframeinfo(inspect.currentframe()).function )

        debug = False

        debug and log.info( f"size={self.size()}" )

        debug = False

        if self.isEmpty :
            size = self.size()
            w = size.width()
            h = size.height()

            image = QPixmap( w, h )

            painter = QPainter( image )

            m = 0
            painter.fillRect( m, m, w - 2*m , h - 2*m, QColor( 'white' ) )

            y = 0
            dy = 20

            row = 0
            while y < h :
                x = 0
                dx = 20
                col = 0
                while x < w :
                    draw = (row % 2 == col%2)

                    if draw :
                        debug and log.info(f"r={row}, c={col}, x={x}, y={y}, dx={dx}, dy={dy}")

                        painter.fillRect( x, y, dx, dy, QColor('gray'))
                    pass

                    x += dx
                    col += 1
                pass

                row += 1
                y += dy
            pass

            if 1 :
                pen = QPen()
                pen.setWidth(1)
                pen.setColor(QColor('green'))
                painter.setPen(pen)

                painter.drawRect( 0, 0, w -1, h - 2 )
            pass

            painter.end()

            self.setImage( image, True )
        pass

        self.updateViewer()
    pass

    def showEvent(self, e ):
        log.info(inspect.getframeinfo(inspect.currentframe()).function)
        debug = False
        debug and log.info( f"size={self.size()}" )

        return QGraphicsView.showEvent(self, e)
    pass

    def hasImage(self):
        return self._pixmapHandle is not None
    pass

    def clearImage(self):
        # Removes the current image pixmap from the scene if it exists.

        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None
        pass
    pass

    def pixmap(self):
        # Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.

        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None
    pass

    def image(self):
        # Returns the scene's current image pixmap as a QImage, or else None if no image exists.

        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None
    pass

    def setImage(self, image, isEmpty = False ):
        self.isEmpty = isEmpty

        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        pass

        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        pass

        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()
    pass

    def loadImageFromFile(self, fileName="", setFileName=False):
        # Load an image from file.

        if len(fileName) == 0:
            settings = self.settings

            home = os.path.expanduser('~')
            directory = os.path.join(home, 'Documents')

            directory = self.settings.value( "open_dir", directory )

            file_filter = "Image Files (*.png *.jpg *.bmp)"
            fileName, dummy = QFileDialog.getOpenFileName(self, "Open image file.", directory=directory, filter=file_filter)

            log.info(f"fileName = {fileName}")

            if fileName :
                directory = os.path.dirname( fileName )
                log.info( f"directory = {directory}")
                settings.setValue("open_dir", directory )
            pass
        pass

        if len(fileName) and os.path.isfile(fileName):
            image = QImage(fileName)
            self.setImage(image)
        pass

        if setFileName and hasattr(self, "fileNameLineEdit" ) :
            self.fileName = fileName
            self.fileNameLineEdit.setText(fileName)
            self.messageLineEdit.setText( "Opened." )
        pass

        return fileName
    pass # -- loadImageFromFile

    def updateViewer(self):
        if not self.hasImage():
            return
        if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
            self.fitInView(self.zoomStack[-1], Qt.IgnoreAspectRatio)
        else:
            self.zoomStack = []
            self.fitInView(self.sceneRect(), self.aspectRatioMode)
        pass
    pass

    def isCtrl(self):
        isCtrl = False

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            log.info('Shift+Click')
        elif modifiers == QtCore.Qt.ControlModifier:
            isCtrl = True

            log.info('Control+Click')
        elif modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
            log.info('Control+Shift+Click')
        else:
            log.info('Click')
        pass

        return isCtrl
    pass

    def mousePressEvent(self, event):
        scenePos = self.mapToScene(event.pos())

        isCtrl = self.isCtrl()

        if not isCtrl :
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        pass

        QGraphicsView.mousePressEvent(self, event)
    pass

    def mouseReleaseEvent(self, event):
        QGraphicsView.mouseReleaseEvent(self, event)
        scenePos = self.mapToScene(event.pos())

        isCtrl = self.isCtrl()

        dragMode = self.dragMode()

        if dragMode == QGraphicsView.ScrollHandDrag :
            self.setDragMode(QGraphicsView.NoDrag)
        elif dragMode == QGraphicsView.RubberBandDrag :
            # image zie
            sceneRect = self.sceneRect()
            viewBox = self.zoomStack[-1] if len(self.zoomStack) else sceneRect
            selectionArea = self.scene.selectionArea().boundingRect()
            selectionBox = selectionArea.intersected(viewBox)

            log.info(f"sceneRect={sceneRect}")
            log.info(f"viewBox={viewBox}")
            log.info(f"selectionArea={selectionArea}")
            log.info(f"selectionBox={selectionBox}")

            # Clear current selection area.
            self.scene.setSelectionArea(QPainterPath())

            if selectionBox.isValid() and (selectionBox != viewBox):
                self.zoomStack.append(selectionBox)
                self.updateViewer()
            pass

            self.setDragMode(QGraphicsView.NoDrag)
        pass
    pass

    def mouseDoubleClickEvent(self, event):
        scenePos = self.mapToScene(event.pos())

        isCtrl = self.isCtrl()

        dragMode = self.dragMode()

        if self.dblClickFileLoad :
            self.loadImageFromFile()
        else :
            self.setZoom( 1.0 )
        pass

        QGraphicsView.mouseDoubleClickEvent(self, event)
    pass # -- mouseDoubleClickEvent

    def setZoom(self, ratio):
        if ratio == 1 :
            self.zoomStack = []  # Clear zoom stack.
            self.updateViewer()
        pass
    pass

    def wheelEvent(self, e):
        self.mouseWheelEvent( e )
    pass

    def mouseWheelEvent(self, e):
        # 마우스 휠 이벤트

        angleDelta = e.angleDelta()
        log.info(f"angleDelta = {angleDelta}")

        isCtrl = self.isCtrl()

        if isCtrl :
            self.setDragMode(QGraphicsView.NoDrag)
        else :
            super(QGraphicsView, self).wheelEvent(e)
        pass
    pass # -- wheelEvent

pass

if __name__ == '__main__':
    app = QApplication(sys.argv)

    viewer = QtImageViewer( dblClickFileLoad = True )

    # Show viewer and run application.
    viewer.show()

    sys.exit(app.exec_())
pass