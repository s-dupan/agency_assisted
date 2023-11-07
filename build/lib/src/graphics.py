"""
Widget for plotting calibration information during abstract control.

"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QComboBox, QDesktopWidget, QMessageBox, QLabel, QTreeWidget, QTreeWidgetItem
from axopy.gui.canvas import Item
from pyqtgraph.dockarea import *

class Basket(Item):
    """Collection of two lines oriented as a "V sign".

    The coordinates of this item correspond to the bottom of the V. This
    item's ``qitem`` attribute is a ``QGraphicsItemGroup`` (a group of two
    lines).

    Parameters
    ----------
    x : float
        x-coordinate of bottom basket
    y : float
        y-coordinate of bottom basket
    size : float
        The size is the length of each line making up the basket.
    linewidth : float
        Thickness of each line making up the basket.
    color : str
        Color of the lines making up the basket.
    """

    def __init__(self, xy_origin, xy_rotate=45, x=0, y=0, size=0.2, linewidth=0.01, color='white'):
        path = QtGui.QPainterPath()
        path.moveTo(x,y)
        path.arcTo(-size,size,2*size,-2*size,0,90)
        path.closeSubpath()
        qitem = QtWidgets.QGraphicsPathItem(path)
        self.br = QtGui.QBrush(QtGui.QColor(color))
        # qitem.setBrush(br)        # fill color
        qitem.setPen(QtGui.QPen(self.br, linewidth))
        qitem.rotate(xy_rotate)
        qitem.setPos(xy_origin[0], xy_origin[1])
        super(Basket, self).__init__(qitem)

class Target(Item):
    """Collection of lines and arches that form a target in the V-shaped task.

    Parameters
    ----------
    target_number : float
        The target number that has to be displayed
    linewidth : float
        Thickness of each line that makes up the target
    color = str
        Color of the lines making up the target
    """

    def __init__(self, xy_origin, theta_target, r1=0.5, r2=0.8, rotation=90, linewidth=0.01, color='white'):
        path = QtGui.QPainterPath()
        path.moveTo(r1,0)
        path.arcTo(-r1,r1,2*r1,-2*r1,0,theta_target)
        path.arcTo(-r2,r2,2*r2,-2*r2,theta_target,-theta_target)
        path.closeSubpath()
        qitem = QtWidgets.QGraphicsPathItem(path)
        self.br = QtGui.QBrush(QtGui.QColor(color))
        # qitem.setBrush(br)        # fill color
        qitem.setPen(QtGui.QPen(self.br, linewidth))
        qitem.rotate(rotation)
        qitem.setPos(xy_origin[0], xy_origin[1])
        super(Target, self).__init__(qitem)

    # TO DO --- does not work at the moment
    @property
    def color(self):
        return self.br().color().getRgb()
    # properties - to do

    @color.setter
    def color(self, color):
        self.br.setColor(QtGui.QColor(color))

class Cursor(Item):
    """Collection of lines and arches that form a target in the V-shaped task.

    Parameters
    ----------
    target_number : float
        The target number that has to be displayed
    linewidth : float
        Thickness of each line that makes up the target
    color = str
        Color of the lines making up the target
    """

    def __init__(self, x=0, y=0, r=0.05, color = 'green'):
        path = QtGui.QPainterPath()
        path.moveTo(r,0)
        path.arcTo(-r, r, 2*r, -2*r, 0, 360)
        path.closeSubpath()
        path.translate(x,y)
        qitem = QtWidgets.QGraphicsPathItem(path)
        self.br = QtGui.QBrush(QtGui.QColor(color))
        qitem.setBrush(self.br)        # fill color
        qitem.setPen(QtGui.QPen(self.br, 0))        # linewidth 0
        qitem.rotate(45)
        qitem.setPos(0, -0.5)   # bring to middle screen
        super(Cursor, self).__init__(qitem)

class CalibWidget(QWidget):
    """ Visualising calibration information for 1 channel

    A window is opened that includes raw data, bar, and buttons to allow selecting the channel for calibration, and calculating rest/contraction.
    """

    def __init__(self, channel_names=None, c_min=None, c_max=None, c_std=None, c_select=None, channels_task=None):
        super(CalibWidget, self).__init__()

        self.plot_items = []
        self.plot_data_items = []

        self.n_channels = 0
        self.channel_names = channel_names
        self.c_min = c_min
        self.c_max = c_max
        self.c_std = c_std
        self.c_select = c_select
        self.channels_task = channels_task

    def plot(self, data, data_mav, data_calib):
        """
        Adds new data to the widget.

        Previous data are scrolled to the left, and the new data is added to
        the end.

        Parameters
        ----------
        data : ndarray, shape = (n_channels, n_samples)
            Window of data to add to the end of the currently-shown data.
        """
        nsamp, nch  = data.shape
        if nch != self.n_channels:
            self.n_channels = nch

            if self.channel_names is None:
                self.channel_names = range(self.n_channels)

            self._update_num_channels(data_calib)


        for i, pdi in enumerate(self.plot_items):
            pdi.data_calib = data_calib
            pdi.emg.setData(data[:,i])
            if (self.c_min[i] != 0 and self.c_max[i] != 1):
                pdi.bar.setOpts(height=data_mav[:,i])

    def _update_num_channels(self, data_calib):
        """
        Adds a dock for each channels

        """
        for i, name in zip(range(self.n_channels), self.channel_names):
            plot_item = NewChannel(i, self.channel_names, self.n_channels, self.c_min, self.c_max, self.c_std, self.c_select, data_calib, self.plot_items, self.channels_task)
            plot_item.show()
            self.plot_items.append(plot_item)


###############################################################
### layout of the window + all functionalities of buttons/lists
###############################################################
# class NewChannel(QWidget):
class NewChannel(QtGui.QWidget):
################


    def __init__(self, ch_number, ch_names, n_channels, c_min, c_max, c_std, c_select, data_calib, plot_items, channels_task):
        # open new window and set properties
        QWidget.__init__(self)

        self.ch_number = ch_number
        self.ch_names = ch_names
        self.n_channels = n_channels

        self.c_min = c_min
        self.c_max = c_max
        self.c_std = c_std
        self.c_select = c_select
        self.data_calib = data_calib
        self.plot_items = plot_items
        self.channels_task = channels_task

        self.setWindowTitle(self.ch_names[self.ch_number])

        # self.statusBar = QStatusBar()
        # self.setStatusBar(self.statusBar)
        # self.statusBar.showMessage('Rest: r;    Rest all: z;    Contract: c;    Close all: esc;    Resize windows: double-click')

        layout = QGridLayout()
        layout.setSpacing(20)
        self.setLayout(layout)

        # Widgets
        self.emgWidget = pg.PlotWidget(background=None)
        self.emg = self.emgWidget.plot(pen='b')
        self.emgWidget.hideAxis('left')
        self.emgWidget.hideAxis('bottom')

        self.barWidget = pg.PlotWidget(background=None)
        self.bar = pg.BarGraphItem(x=[1.],height=[0.], width=1, brush='b')
        self.barWidget.addItem(self.bar)
        self.barWidget.setYRange(0, 1.3)
        self.barWidget.hideAxis('bottom')
        self.barWidget.showGrid(y=True, alpha=0.5)

        self.select = QComboBox()
        self.select.addItem('Select')
        self.select.addItem('Ignore')
        for i in range(self.channels_task):
            self.select.addItem(str(i))
        self.select.currentIndexChanged[str].connect(self.selectActivated)
        self.reset_button = QPushButton('Reset')
        self.reset_button.resize(self.reset_button.sizeHint())
        self.reset_button.clicked.connect(self.resetButtonClicked)
        self.max_button = QPushButton('max')
        self.max_button.resize(self.max_button.sizeHint())
        self.max_button.clicked.connect(self.maxButtonClicked)
        self.min_button = QPushButton('min')
        self.min_button.resize(self.min_button.sizeHint())
        self.min_button.clicked.connect(self.minButtonClicked)

        self.tree = QTreeWidget()
        self.ch_active = []
        for ch in range(len(ch_names)):
            ch_active = QTreeWidgetItem(self.tree)
            ch_active.setText(0, self.ch_names[ch])
            ch_active.setFlags(ch_active.flags() | Qt.ItemIsUserCheckable)
            if ch == self.ch_number:
                ch_active.setCheckState(0, Qt.Checked)
            else:
                ch_active.setCheckState(0, Qt.Unchecked)
            self.ch_active.append(ch_active)

        layout.addWidget(self.emgWidget, 0, 0, 4, 1)
        layout.addWidget(self.barWidget, 0, 1, 4, 1)
        layout.addWidget(self.select, 0, 2)
        layout.addWidget(self.reset_button, 1, 2)
        layout.addWidget(self.max_button, 2, 2)
        layout.addWidget(self.min_button, 3, 2)
        layout.addWidget(self.tree,0,3,4,1)

        # determine layout window
        layout.setColumnStretch(1,10)
        layout.setColumnStretch(2,2)
        layout.setColumnStretch(3,2)

        # determine where on screen the window will be positioned
        screen = QDesktopWidget().screenGeometry()
        # define positions, with a max of 2 rows
        if self.n_channels == 1:
            positions = [(0,0)]
        elif self.n_channels == 2:
            positions = [(0,0), (0,1)]
        else:
            positions = [(i,j) for i in range(2) for j in range(int(np.ceil(self.n_channels/2)))]

        max_row = max(positions)[0]
        max_col = max(positions)[1]
        w_w = screen.width()/(max_col+1)
        w_h = min([w_w/3, screen.height()/(max_row+1)])

        self.resize(w_w, w_h)
        self.move(w_w*positions[self.ch_number][1], w_h*positions[self.ch_number][0])

        self.installEventFilter(self)

    def maxButtonClicked(self):
        for i in range(len(self.ch_active)):
            if self.ch_active[i].checkState(0) == Qt.Checked:
                self.c_max[i] = np.max(self.data_calib[i])

        # print('data_calib: ', self.data_calib)
        print('New c_max ', self.c_max)

    def minButtonClicked(self):
        for i in range(len(self.ch_active)):
            if self.ch_active[i].checkState(0) == Qt.Checked:
                self.c_min[i] = np.min(self.data_calib[i])

        # print('data_calib: ', self.data_calib)
        print('New c_min ', self.c_min)

    def restAll(self):
        for i in range(len(self.ch_active)):
            self.c_min[i] = np.min(self.data_calib[i])

        print('New c_min ', self.c_min)

    def resetButtonClicked(self):
        self.c_min[self.ch_number] = 0
        self.c_max[self.ch_number] = 1
        self.select.setCurrentText('Select')
        print('New c_min and c_max ', self.c_min, self.c_max)

    def selectActivated(self, text):
        if text == 'Ignore':
            self.c_select[self.ch_number] = np.nan
        elif text == 'Select':
            self.c_select[self.ch_number] = np.nan
        else:
            if int(float(text)) in self.c_select:
                if self.overwriteSelect():
                    ch_overwrite = np.where(self.c_select == int(float(text)))[0][0]
                    print('channel to overwrite: ', ch_overwrite)
                    self.c_select[self.ch_number] = int(float(text))
                    self.plot_items[ch_overwrite].select.setCurrentText('Ignore')
                else:
                    self.plot_items[self.ch_number].select.setCurrentText('Ignore')
            else:
                self.c_select[self.ch_number] = int(float(text))
        print('New selections are: ', self.c_select)

    def overwriteSelect(self):
        reply = QMessageBox.question(self, 'Message', "A different EMG channel is already selected to be this control signal. Are you sure you want to overwrite?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            overwrite = True
        else:
            overwrite = False
        return overwrite

    def eventFilter(self, obj, event):
        if event.type() == QEvent.WindowActivate:
            for i in range(len(self.ch_active)):
                if self.ch_active[i].checkState(0) == Qt.Checked:
                    self.plot_items[i].emg.setPen('b')
                else:
                    self.plot_items[i].emg.setPen(120,120,120)

            return True
        else:
            return False

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
        if e.key() == QtCore.Qt.Key_R:
            self.minButtonClicked()
        if e.key() == QtCore.Qt.Key_C:
            self.maxButtonClicked()
        if e.key() == QtCore.Qt.Key_Z:
            self.restAll()

    def closeEvent(self, event):
        n_sel_chan = np.count_nonzero(~np.isnan(self.c_select))
        print('Nr selected channels: ', n_sel_chan)
        n_windows_open = 0
        for i in range(len(self.plot_items)):
            if self.plot_items[i].isVisible():
                n_windows_open =n_windows_open+1

        text = str(self.select.currentText())
        if text == 'Ignore' or text == 'Select':
            event.accept()
        else:
            if self.c_min[self.ch_number] == 0 or self.c_max[self.ch_number] == 1:
                reply = QMessageBox.question(self, 'Message',
                    "This channel is selected for control, but is not calibrated. The selection will be canceled. Are you sure you want to close channel?", QMessageBox.Yes |
                    QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.plot_items[self.ch_number].select.setCurrentText('Ignore')
                    event.accept()
                else:
                    event.ignore()

        if (n_windows_open <= self.channels_task) and (n_sel_chan < self.channels_task):
            reply = QMessageBox.question(self, 'Message',
                "Ignoring this channel will leave you with less channels than you need for the task. Are you sure you want to close the channel?", QMessageBox.Yes |
                QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
