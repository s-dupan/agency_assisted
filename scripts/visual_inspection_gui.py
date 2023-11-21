"""
GUI to inspect the individual trials of abstract control experiments.

The computed MAVs and raw data are presented, together with extra information (BN, TN, percent hold). People can choose to accept/reject a trial, and can run through trials with either arrow keys or Prev/Next buttons
"""

import sys
import os
import h5py
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QApplication, QLabel, QFileDialog, QMainWindow
from pyqtgraph.dockarea import *


class Inspection(QMainWindow):
    def __init__(self):
        super().__init__()
        # prepare saving
        self.save_dir = None

        self.initUI()

    def initUI(self):
        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1000,500)
        self.setWindowTitle('visual_inspection_gui')

        # create 3 docks: mav, emg, and info
        mav_dock = Dock('D1', size=(700,200))
        mav_dock.hideTitleBar()
        emg_dock = Dock('D2', size=(700,200))
        emg_dock.hideTitleBar()
        info_dock = Dock('D3', size=(300,400))
        info_dock.hideTitleBar()

        area.addDock(mav_dock, 'left')
        area.addDock(emg_dock, 'bottom', mav_dock)
        area.addDock(info_dock, 'right')

        # add mav plots
        mav = pg.PlotWidget(title='MAV', background=None)
        mav.setYRange(0, 1.3, padding=0)
        self.mav1 = mav.plot()
        self.mav1.setPen(pg.mkPen(color=(255,0,0), width=3))
        self.mav2 = mav.plot()
        self.mav2.setPen(pg.mkPen(color=(0,0,255), width=3))
        mav_dock.addWidget(mav)
        # add vertical lines for start-end reach and hold phase
        self.reach_start = pg.InfiniteLine()
        self.reach_start.setPen(pen=(120,120,120), width=5)
        self.hold_start = pg.InfiniteLine()
        self.hold_start.setPen(pen=(120,120,120), width=5)
        self.hold_end = pg.InfiniteLine()
        self.hold_end.setPen(pen=(120,120,120), width=5)
        mav.addItem(self.reach_start)
        mav.addItem(self.hold_start)
        mav.addItem(self.hold_end)

        # add raw plots
        emg = pg.PlotWidget(title='raw EMG', background=None)
        self.emg1 = emg.plot(pen='r')
        self.emg2 = emg.plot(pen='b')
        emg_dock.addWidget(emg)

        # get pushbuttons and labels in dock 3
        w1 = pg.LayoutWidget()
        self.loadBtn = QPushButton('Load')
        self.loadBtn.setStyleSheet('font-size: 20px')
        self.loadBtn.clicked.connect(self.loadBtnClicked)
        self.acceptBtn = QPushButton('Accept/Reject')
        self.acceptBtn.setStyleSheet('background-color: green; font-size: 20px')
        self.acceptBtn.clicked.connect(self.acceptBtnClicked)
        self.prevBtn = QPushButton('Prev')
        self.prevBtn.setStyleSheet('font-size: 20px')
        self.prevBtn.clicked.connect(self.prevBtnClicked)
        self.nextBtn = QPushButton('Next')
        self.nextBtn.setStyleSheet('font-size: 20px')
        self.nextBtn.clicked.connect(self.nextBtnClicked)

        block = QLabel('Block num:')
        block.setStyleSheet('font-size: 20px')
        self.bn = QLabel('-')
        self.bn.setStyleSheet('font-size: 20px')
        trial = QLabel('Trial num:')
        trial.setStyleSheet('font-size: 20px')
        self.tn = QLabel('-')
        self.tn.setStyleSheet('font-size: 20px')
        score = QLabel('Percent hold:')
        score.setStyleSheet('font-size: 20px')
        self.ph = QLabel('-')
        self.ph.setStyleSheet('font-size: 20px')

        w1.addWidget(self.loadBtn, row=0, col=0, colspan=2)
        w1.addWidget(block, row=1, col=0)
        w1.addWidget(self.bn, row=1, col=1)
        w1.addWidget(trial, row=2, col=0)
        w1.addWidget(self.tn, row=2, col=1)
        w1.addWidget(score, row=3, col=0)
        w1.addWidget(self.ph, row=3, col=1)
        w1.addWidget(self.acceptBtn, row=4, col=0, colspan=2)
        w1.addWidget(self.prevBtn, row=5, col=0)
        w1.addWidget(self.nextBtn, row=5, col=1)

        info_dock.addWidget(w1)

        self.show()

    def loadBtnClicked(self):
        # if other file is open, save this one first
        if self.save_dir is not None:
            self.trials.to_csv(self.save_dir, index = None, header=True)
        # prepare loading
        self.file_dir = QFileDialog.getExistingDirectory(self, "Select subject directory")
        self.loadBtn.setText(self.file_dir.split('/')[-1])

        # make list of subfolders
        self.subfolders = [f.path for f in os.scandir(self.file_dir) if f.is_dir()]
        self.control_ind = np.array([])
        for i in range(len(self.subfolders)):
            if 'control_20' in self.subfolders[i]:
                self.control_ind = np.append(self.control_ind, i)
        self.current_file_ind = 0
        self.total_files = self.control_ind.size
        if self.total_files == 0:
            print('There are no control files in this directory')

        self.loadFile()


    def loadFile(self):
        current_file = self.subfolders[int(self.control_ind[self.current_file_ind])]

        print(current_file)
        # open first control folder, load the existing files
        # for trial data: check if inspected file is already generated
        try:
            self.trials = pd.read_csv(current_file + '/trials.csv')
            self.data_raw = h5py.File(current_file + '/data_raw.hdf5')
            self.data_mav = h5py.File(current_file + '/data_proc.hdf5')
            self.state = h5py.File(current_file + '/state.hdf5')

            # if no accept column for trials, add one. Set accepted as standard
            if 'accept' not in self.trials.keys():
                self.trials['accept'] = 1

            # prepare saving - path to save
            self.save_dir = current_file + '/trials.csv'
            print('Save dir: ', self.save_dir)

            # determine control channels
            self.ch_0 = self.trials['control_ch_0'][0]
            self.ch_1 = self.trials['control_ch_1'][0]

            # update the windows
            self.current_trial = 0
            self.updateGui(self.current_trial)

        except FileNotFoundError:
            # recursion in the code. Not ideal!
            self.current_file_ind = self.current_file_ind + 1
            self.loadFile()

    def acceptBtnClicked(self):
        if QColor(self.acceptBtn.palette().color(1)) == QColor('green'):
            self.acceptBtn.setStyleSheet('background-color: red; font-size: 20px')
            self.trials['accept'][self.current_trial] = 0
        else:
            self.acceptBtn.setStyleSheet('background-color: green; font-size: 20px')
            self.trials['accept'][self.current_trial] = 1

    def prevBtnClicked(self):
        if self.current_trial != 0:
            self.current_trial = self.current_trial - 1
            self.updateGui(self.current_trial)
        elif self.current_file_ind > 0:
            self.current_file_ind = self.current_file_ind - 1
            # save current file
            self.trials.to_csv(self.save_dir, index = None, header=True)
            self.loadFile()

    def nextBtnClicked(self):
        if self.current_trial != self.trials.shape[0]-1:
            self.current_trial = self.current_trial + 1
            self.updateGui(self.current_trial)
        elif self.current_file_ind < self.total_files -1:
            self.current_file_ind = self.current_file_ind + 1
            # saving current file
            self.trials.to_csv(self.save_dir, index = None, header=True)
            self.loadFile()


    def updateGui(self, tn):
        # update gui based on tn
        self.mav1.setData(self.data_mav[str(tn)][0:][self.ch_0])
        self.mav2.setData(self.data_mav[str(tn)][0:][self.ch_1])
        self.emg1.setData(self.data_raw[str(tn)][0:][self.ch_0])
        self.emg2.setData(self.data_raw[str(tn)][0:][self.ch_1])
        self.bn.setText(str(self.trials['block'][tn]))
        self.tn.setText(str(self.trials['trial'][tn]))
        self.ph.setText('{0:.2f}'.format(self.trials['percent_hold'][tn]))
        if self.trials['accept'][tn] == 1:
            self.acceptBtn.setStyleSheet('background-color: green; font-size: 20px')
        else:
            self.acceptBtn.setStyleSheet('background-color: red; font-size: 20px')

        # determine start-end reach and hold based on state information
        state = self.state[str(tn)][0:]
        reach_start = np.nonzero(state == 0)[0][0]
        hold_start = np.nonzero(state == 1)[0][0]
        state_nan = np.argwhere(np.isnan(state))
        hold_end_ind = np.nonzero(state_nan > hold_start)[0][0]
        hold_end = state_nan[hold_end_ind][0]

        self.reach_start.setValue(reach_start)
        self.hold_start.setValue(hold_start)
        self.hold_end.setValue(hold_end)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up:
            self.acceptBtnClicked()
        elif e.key() == Qt.Key_Right:
            self.nextBtnClicked()
        elif e.key() == Qt.Key_Left:
            self.prevBtnClicked()
        elif e.key() == Qt.Key_Escape:
            # saving
            self.trials.to_csv(self.save_dir, index = None, header=True)
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Inspection()
    sys.exit(app.exec_())
