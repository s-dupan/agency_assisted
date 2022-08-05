"""
Tracking based on error signal

During calibration phase, the participant is asked to reproduce a series of
muscle contractions to determine noise level and maximum contraction during the
control. Data are collected and stored so that they can be used later to train
models.

Input devices:
trigno
    Delsys Trigno EMG system.
myo
    Myo armband.
noise
    Noise generator
quattro
    Delsys quattro EMG system


All configuration settings are stored and loaded from an external configuration
file (``config.ini``).
"""

import os
import numpy as np
import h5py
from time import localtime, strftime
import winsound

from argparse import ArgumentParser
from configparser import ConfigParser
from scipy.signal import butter

from axopy.experiment import Experiment
from axopy.task import Task
from axopy import util
from axopy.timing import Counter
from axopy.gui.canvas import Canvas, Text, Circle
from axopy.pipeline import (Windower, Pipeline, Filter,
                            FeatureExtractor, Ensure2D, Block)
from PyQt5.QtWidgets import QDesktopWidget
from src.graphics import CalibWidget, Basket, Target
from axopy.features import MeanAbsoluteValue


class Normalize(Block):
    def process(self, data):
        data_norm = (data - c_min)/(c_max-c_min)
        data_norm[data_norm < 0] = 0.0
        return data_norm

class _BaseTask(Task):
    """Base experimental task.

    Implements the processing pipeline, the daqstream and the trial counter.
    """
    def __init__(self):
        super(_BaseTask, self).__init__()
        self.pipeline = self.make_pipeline()

    def make_pipeline(self):
        b,a = butter(FILTER_ORDER,
                      (LOWCUT / S_RATE / 2., HIGHCUT / S_RATE / 2.),
                      'bandpass')
        pipeline = Pipeline([
            Windower(int(S_RATE * WIN_SIZE)),
            Filter(b, a=a,
                   overlap=(int(S_RATE * WIN_SIZE) -
                            int(S_RATE * READ_LENGTH))),
            FeatureExtractor([('MAV', MeanAbsoluteValue())],
                             n_channels),
            Ensure2D(orientation='row'),
            Normalize()
        ])

        print('Pipeline ok...')
        return pipeline

    def prepare_daq(self, daqstream):
        self.daqstream = daqstream
        self.daqstream.start()

    def key_press(self, key):
        super(_BaseTask, self).key_press(key)
        if key == util.key_escape:
            self.finish()

    def finish(self):
        self.daqstream.stop()
        self.finished.emit()

class DataCollection(_BaseTask):
    """Data collection task for callibration of the system.

    Collects minimum and maximum values for interface. Raw EMG activity is
    shown on screeen. Experimenter decides through key presses when muscles
    at rest and when they represent full activity.
    """

    def __init__(self):
        super(DataCollection, self).__init__()
        (self.pipeline_raw, self.pipeline_calib) = self.make_pipeline_calib()

    def make_pipeline_calib(self):
        print('Pipeline calib ok...')
        b,a = butter(FILTER_ORDER,
                      (LOWCUT / S_RATE / 2., HIGHCUT / S_RATE / 2.),
                      'bandpass')
        pipeline_raw = Pipeline([
            Windower(int(S_RATE * WIN_SIZE_CALIB)),
            Filter(b, a=a,
                   overlap=(int(S_RATE * WIN_SIZE_CALIB) -
                            int(S_RATE * READ_LENGTH))),
            Ensure2D(orientation='col')
        ])

        pipeline_calib = Pipeline([
            Windower(int(S_RATE * WIN_SIZE)),
            Filter(b, a=a,
                   overlap=(int(S_RATE * WIN_SIZE) -
                            int(S_RATE * READ_LENGTH))),
            FeatureExtractor([('MAV', MeanAbsoluteValue())],
                    n_channels),
            Ensure2D(orientation='col'),
            Windower(int((1/READ_LENGTH) * WIN_SIZE_CALIB)),
        ])

        return pipeline_raw, pipeline_calib


    def prepare_design(self, design):
        for b in range(N_BLOCKS):
            block = design.add_block()
            for t in range(N_TRIALS):
                block.add_trial()

    def prepare_graphics(self, container):
        self.scope = CalibWidget(channel_names, c_min, c_max, c_std, c_select, channels_task)

    def prepare_storage(self, storage):
        time = strftime('%Y%m%d%H%M%S', localtime())
        self.writer = storage.create_task('calibration' + '_' + time)
        self.data_dir = os.path.join(os.path.dirname(
                            os.path.realpath(__file__)),
                            'data')
        
        # Save Config File
        block_savedir = os.path.join(self.data_dir, exp.subject, 
                    'calibration' + '_' + time)
        config = ConfigParser()
        config.read(CONFIG)
        with open(block_savedir+"\\config.ini", 'w') as f:
            config.write(f)

    def run_trial(self, trial):
        # saving data
        trial.add_array('data_raw', stack_axis=1)
        trial.add_array('data_proc', stack_axis=1)
        trial.add_array('c_min', stack_axis=1)
        trial.add_array('c_max', stack_axis=1)
        trial.add_array('c_std', stack_axis=1)
        trial.add_array('c_select', stack_axis=1)

        self.pipeline.clear()
        self.connect(self.daqstream.updated, self.update)

    def update(self, data):
        if self.pipeline_raw is not None:
            data_raw = self.pipeline_raw.process(data)
        if self.pipeline is not None:
            data_proc = self.pipeline.process(data)
        if self.pipeline_calib is not None:
            data_calib = self.pipeline_calib.process(data)

        self.scope.plot(data_raw, data_proc, data_calib)
        # Update Arrays
        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(data_proc)

    def key_press(self, key):
        super(_BaseTask, self).key_press(key)
        if key == util.key_escape:
            self.finish()

    def finish(self):
        self.trial.arrays['c_min'].stack(c_min)
        self.trial.arrays['c_max'].stack(c_max)
        self.trial.arrays['c_std'].stack(c_std)
        self.trial.arrays['c_select'].stack(c_select)
        self.writer.write(self.trial)
        self.disconnect(self.daqstream.updated, self.update)
        self.daqstream.stop()
        self.finished.emit()

class RealTimeControl(_BaseTask):
    """Real time abstract control
    """

    def __init__(self):
        super(RealTimeControl, self).__init__()
        self.advance_block_key = util.key_return
        self.prepare_targets()
        self.prepare_timers()

    def prepare_targets(self):
        if NUM_TARGETS == 12:
            self.radii = 0.303, 0.451, 0.673, 1
        elif NUM_TARGETS == 8:
            self.radii = 0.303, 0.550, 1
        elif NUM_TARGETS == 4:
            self.radii = 0.303, 1

        self.target_var = [(self.radii[i]*UI_XY_SCALE, self.radii[i+1]*UI_XY_SCALE, UI_ROTATION+j*UI_THETA_TARGET) for i in range(NUM_TARGETS//4) for j in range(4)]

    def prepare_timers(self):
        self.iti_timer = Counter(int(TRIAL_INTERVAL / READ_LENGTH))
        self.iti_timer.timeout.connect(self.finish_iti)
        self.reach_timer = Counter(int(REACH_LENGTH/ READ_LENGTH))
        self.reach_timer.timeout.connect(self.finish_reach)
        self.hold_timer = Counter(int(HOLD_LENGTH/ READ_LENGTH))
        self.hold_timer.timeout.connect(self.finish_hold)
        self.score_timer = Counter(int(SCORE_LENGTH / READ_LENGTH))
        self.score_timer.timeout.connect(self.finish_trial)

    def prepare_design(self, design):
        for b in range(N_BLOCKS):
            block = design.add_block()
            for target in TARGETS:
                    block.add_trial(attrs={'target' : target})
            block.shuffle()

    def prepare_graphics(self, container):
        self.canvas = Canvas()
        container.set_widget(self.canvas)

        self.task_canvas = Canvas(border_color='black',bg_color = '#000000', draw_border=False)
        monitor = QDesktopWidget().screenGeometry(DISPLAY_MONITOR)

        # add canvas elements
        self.cursor = Circle(diameter = 0.0625*UI_XY_SCALE, color = 'green')
        self.cursor.hide()
        self.basket = Basket(xy_origin=UI_XY_ORIGIN, size = 0.2*UI_XY_SCALE, xy_rotate = UI_ROTATION)
        self.basket.hide()
        self.text_score = Text(text='test', color='white')
        self.text_score.pos = (0,0)
        self.text_score.hide()

        self.task_canvas.add_item(self.basket)
        self.task_canvas.add_item(self.cursor)
        self.task_canvas.add_item(self.text_score)

        self.task_canvas.move(monitor.left(), monitor.top())
        self.task_canvas.showFullScreen()

    def prepare_storage(self, storage):
        time = strftime('%Y%m%d%H%M%S', localtime())
        self.writer = storage.create_task('control' + '_' + time)
        self.data_dir = os.path.join(os.path.dirname(
                            os.path.realpath(__file__)),
                            'data')
        
        # Save Config File
        block_savedir = os.path.join(self.data_dir, exp.subject, 
                            'control' + '_' + time)
        config = ConfigParser()
        config.read(CONFIG)
        with open(block_savedir+"\\config.ini", 'w') as f:
            config.write(f)

    def run_trial(self, trial):
        self.iti_timer.reset()

        # add target to canvas
        self.target = Target(xy_origin=UI_XY_ORIGIN, theta_target = UI_THETA_TARGET, r1=self.target_var[self.trial.attrs['target']][0], r2=self.target_var[self.trial.attrs['target']][1], rotation=self.target_var[self.trial.attrs['target']][2])
        self.target.hide()
        self.task_canvas.add_item(self.target)

        trial.add_array('data_raw', stack_axis=1)
        trial.add_array('data_proc', stack_axis=1)
        trial.add_array('hold', stack_axis=1)
        trial.add_array('state', stack_axis=1)
        self.rest_array = np.array([])

        self.pipeline.clear()
        self.connect(self.daqstream.updated, self.update_iti)

    def update_iti(self, data):
        data_proc = self.pipeline.process(data)
        self.cursor.pos = self.transform_data((data_proc[0][CONTROL_CHANNELS]))

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        self.trial.arrays['hold'].stack(np.nan)
        self.trial.arrays['state'].stack(np.nan)
        self.iti_timer.increment()
        if self.cursor.collides_with(self.basket):
            self.rest_array = np.append(self.rest_array, 1)
        else:
            self.rest_array = np.append(self.rest_array, 0)

    def update_rest(self, data):
        data_proc = self.pipeline.process(data)
        self.cursor.pos = self.transform_data((data_proc[0][CONTROL_CHANNELS]))

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        self.trial.arrays['hold'].stack(np.nan)
        self.trial.arrays['state'].stack(np.nan)
        if self.cursor.collides_with(self.basket):
            self.rest_array = np.append(self.rest_array, [1])
        else:
            self.rest_array = np.append(self.rest_array, [0])

        if np.all(self.rest_array[-int(WIN_SIZE_REST/READ_LENGTH):] == 1):
            self.finish_rest()

    def update_reach(self, data):
        data_proc = self.pipeline.process(data)
        self.cursor.pos = self.transform_data((data_proc[0][CONTROL_CHANNELS]))

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        if self.cursor.collides_with(self.target):
            self.trial.arrays['hold'].stack(1)
        else:
            self.trial.arrays['hold'].stack(0)
        self.trial.arrays['state'].stack(0)
        self.reach_timer.increment()

    def update_hold(self, data):
        data_proc = self.pipeline.process(data)
        self.cursor.pos = self.transform_data((data_proc[0][CONTROL_CHANNELS]))

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        if self.cursor.collides_with(self.target):
            self.trial.arrays['hold'].stack(1)
        else:
            self.trial.arrays['hold'].stack(0)
        self.trial.arrays['state'].stack(1)
        self.hold_timer.increment()

    def update_score(self, data):
        data_proc = self.pipeline.process(data)
        self.cursor.pos = self.transform_data((data_proc[0][CONTROL_CHANNELS]))

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        self.trial.arrays['hold'].stack(np.nan)
        self.trial.arrays['state'].stack(np.nan)
        self.score_timer.increment()

    def finish_iti(self):
        self.basket.show()
        self.cursor.show()
        self.disconnect(self.daqstream.updated, self.update_iti)
        self.connect(self.daqstream.updated, self.update_rest)

    def finish_rest(self):
        self.target.show()
        self.cursor.show()
        self.reach_timer.reset()
        winsound.PlaySound('beep_1000Hz_200ms.wav',1)
        self.disconnect(self.daqstream.updated, self.update_rest)
        self.connect(self.daqstream.updated, self.update_reach)

    def finish_reach(self):
        self.hold_timer.reset()
        winsound.PlaySound('beep_1000Hz_200ms.wav',1)
        self.disconnect(self.daqstream.updated, self.update_reach)
        self.connect(self.daqstream.updated, self.update_hold)

    def finish_hold(self):
        self.basket.hide()
        self.target.hide()
        self.cursor.hide()
        # calculate score
        hold_period = np.where(self.trial.arrays['state'].data == 1)[0]
        self.score = np.mean(self.trial.arrays['hold'].data[hold_period])
        self.text_score.qitem.setText("{:.0f} %".format(self.score*100))
        self.text_score.show()
        self.score_timer.reset()
        self.disconnect(self.daqstream.updated, self.update_hold)
        self.connect(self.daqstream.updated, self.update_score)

    def finish_trial(self):
        # Display prompt if that was the last trial in the block
        if self.trial.attrs['trial'] == N_TRIALS - 1:
            self.text_score.qitem.setText(" End of block {}. \n Press enter to continue.".format(
                self.trial.attrs['block'] + 1
            ))
        else:
            self.text_score.hide()
        self.trial.attrs['percent_hold'] = self.score
        self.trial.attrs['boundaries'] = self.radii
        self.trial.attrs['target_var'] = self.target_var[self.trial.attrs['target']]
        self.trial.attrs['control_ch_0'] = CONTROL_CHANNELS[0]
        self.trial.attrs['control_ch_1'] = CONTROL_CHANNELS[1]
        self.writer.write(self.trial)
        # self.disconnect(self.daqstream.updated, self.update)
        self.disconnect(self.daqstream.updated, self.update_score)
        self.next_trial()

    def finish(self):
        self.daqstream.stop()
        self.finished.emit()

    def key_press(self, key):
        if key == util.key_escape:
            self.finish()
        else:
            super().key_press(key)

    def transform_data(self, pos):
        "New position after rotation and translation. Based on interface"
        x = pos[0]
        y = pos[1]

        x_new = x*UI_XY_SCALE*np.cos(np.radians(UI_ROTATION)) - y*UI_XY_SCALE*np.sin(np.radians(UI_ROTATION)) + UI_XY_ORIGIN[0]
        y_new = x*UI_XY_SCALE*np.sin(np.radians(UI_ROTATION)) + y*UI_XY_SCALE*np.cos(np.radians(UI_ROTATION)) + UI_XY_ORIGIN[1]

        return x_new, y_new

if __name__ == '__main__':
    parser = ArgumentParser()
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument('--train', action='store_true')
    task.add_argument('--test', action='store_true')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--trigno', action='store_true')
    source.add_argument('--myo', action='store_true')
    source.add_argument('--noise', action='store_true')
    source.add_argument('--quattro', action='store_true')
    args = parser.parse_args()

    CONFIG = 'config.ini'
    cp = ConfigParser()
    cp.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'config.ini'))
    
    SUBJECT = cp.get('participant', 'subject')
    READ_LENGTH = cp.getfloat('hardware', 'read_length')
    CHANNELS = list(map(int, (cp.get('hardware', 'channels').split(','))))
    n_channels = len(CHANNELS)
    WIN_SIZE = cp.getfloat('processing', 'win_size')
    LOWCUT = cp.getfloat('processing', 'lowcut')
    HIGHCUT = cp.getfloat('processing', 'highcut')
    FILTER_ORDER = cp.getfloat('processing', 'filter_order')
    NUM_TARGETS = cp.getint('experiment', 'targets')
    channels_task = cp.getint('control', 'channels_task')

    if args.trigno:
        from pytrigno import TrignoEMG
        S_RATE = 2000.
        dev = TrignoEMG(channels=CHANNELS, zero_based=False,
                        samples_per_read=int(S_RATE * READ_LENGTH))
    elif args.myo:
        import myo
        from pydaqs.myo import MyoEMG
        # CHANNELS = list(range(1,9))   # could decide to only use 2
        S_RATE = 200.
        if HIGHCUT > 100:
            # due to low sampling rate, code won't work if highcut filter is too high
            HIGHCUT = 100
        myo.init(
            sdk_path=r'C:\Users\user\coding\myo-python\myo-sdk-win-0.9.0')
        dev = MyoEMG(channels=CHANNELS, zero_based=False,
                     samples_per_read=int(S_RATE * READ_LENGTH))
    elif args.noise:
        from axopy.daq import NoiseGenerator
        S_RATE = 2000.
        dev = NoiseGenerator(rate=S_RATE, num_channels=n_channels, amplitude=10.0, read_size=int(S_RATE * READ_LENGTH))

    elif args.quattro:
        from pytrigno import QuattroEMG
        S_RATE = 2000.
        dev = QuattroEMG(sensors=range(1,3), 
                        samples_per_read=int(S_RATE * READ_LENGTH),
                        zero_based=False,
                        mode=313,
                        units='normalized',
                        data_port=50043)

    exp = Experiment(daq=dev, subject=SUBJECT, allow_overwrite=False)

    if args.train:
        N_TRIALS = cp.getint('calibration', 'n_trials')
        N_BLOCKS = cp.getint('calibration', 'n_blocks')
        WIN_SIZE_CALIB = cp.getfloat('calibration', 'win_size')
        channel_names = ['EMG ' + str(i) for i in range(1, n_channels+1)]
        c_min = np.zeros(n_channels)
        c_max = np.ones(n_channels)
        c_std = np.zeros(n_channels)
        c_select = np.full(n_channels, np.nan)

        # download existing calibration info
        use_previous_calib = cp.getint('calibration', 'use_previous_calib')
        root_subject = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data', exp.subject)
        if os.path.exists(root_subject) & use_previous_calib:
            subfolders = [f.path for f in os.scandir(root_subject) 
                                if f.is_dir()]
            if subfolders:
                calib_ind = np.array([])
                for i in range(len(subfolders)):
                    # check if existing calib & if there's data
                    if 'calibration_20' in subfolders[i]:
                        if len(os.listdir(subfolders[i])):
                            calib_ind = np.append(calib_ind, i)
                print(calib_ind)
                if calib_ind.size != 0:
                    last_calib = subfolders[int(calib_ind[-1:])]

                    f = h5py.File([last_calib + '\\c_min.hdf5'][0], 'r')
                    c_min = f['0'][0:]
                    print('c_min: ', c_min)
                    f = h5py.File([last_calib + '\\c_max.hdf5'][0], 'r')
                    c_max = f['0'][0:]
                    print('c_max: ', c_max)
                    f = h5py.File([last_calib + '\\c_select.hdf5'][0], 'r')
                    c_select = f['0'][0:]
                    print('previously existing calibration loaded')

        print('Running calibration...')
        exp.run(DataCollection())
    elif args.test:
        N_TRIALS = cp.getint('control', 'n_trials')
        N_BLOCKS = cp.getint('control', 'n_blocks')
        TARGETS = [i for i in range(NUM_TARGETS)]
        INTERFACE = cp.get('experiment', 'interface')
        WIN_SIZE_REST = cp.getfloat('control', 'win_size_rest')

        TRIAL_INTERVAL = cp.getfloat('control', 'trial_interval')
        REACH_LENGTH = cp.getfloat('control', 'reach')
        HOLD_LENGTH = cp.getfloat('control', 'hold')
        SCORE_LENGTH = cp.getfloat('control', 'score_present')
        DISPLAY_MONITOR = cp.getint('control', 'display_monitor')

        # download calibration info
        root_subject = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data', exp.subject)
        subfolders = [f.path for f in os.scandir(root_subject) if f.is_dir()]
        calib_ind = np.array([])
        for i in range(len(subfolders)):
            if 'calibration_20' in subfolders[i]:
                calib_ind = np.append(calib_ind, i)
        last_calib = subfolders[int(calib_ind[-1:])]

        f = h5py.File([last_calib + '\\c_min.hdf5'][0], 'r')
        c_min = f['0'][0:]
        print('c_min: ', c_min)
        f = h5py.File([last_calib + '\\c_max.hdf5'][0], 'r')
        c_max = f['0'][0:]
        print('c_max: ', c_max)
        f = h5py.File([last_calib + '\\c_select.hdf5'][0], 'r')
        c_select = f['0'][0:]
        print('c_select: ',c_select)

        # update targets based on number of n_trials
        if N_TRIALS%len(TARGETS) != 0:
            raise ValueError('Please make sure number of trials is a multiple of number of targets.')
        TARGETS = TARGETS*int(N_TRIALS/len(TARGETS))

        # determine control channels
        CONTROL_CHANNELS = np.array([])
        for i in range(channels_task):
            CONTROL_CHANNELS = np.append(CONTROL_CHANNELS, np.where(c_select == i)[0][0])
        CONTROL_CHANNELS = CONTROL_CHANNELS.astype(int)
        control_channels_overwrite = cp.getint('control', 'control_channels_overwrite')
        if control_channels_overwrite:
            CONTROL_CHANNELS_NEW = np.array([])
            new_channel_list = list(map(int, (cp.get('control', 'control_channels').split(','))))

            print('CHANNELS: ', CHANNELS)
            print('new_channel_list ', new_channel_list)

            for i in range(len(new_channel_list)):
                print(CHANNELS.index(new_channel_list[i]))
                CONTROL_CHANNELS_NEW = np.append(CONTROL_CHANNELS_NEW, CHANNELS.index(new_channel_list[i]))
                CONTROL_CHANNELS_NEW = CONTROL_CHANNELS_NEW.astype(int)
            if len(CONTROL_CHANNELS_NEW) != channels_task:
                raise ValueError('Trying to overwrite the wrong amount of control channels.')
            elif np.any([i > n_channels-1 for i in CONTROL_CHANNELS_NEW]):
                raise ValueError('The value for one of the control channels is higher than the amount of channels in the experiment.')
            elif np.any([[c_min[i] == 0 for i in CONTROL_CHANNELS_NEW], [c_max[i] == 1 for i in CONTROL_CHANNELS_NEW]]):
                raise ValueError('One of the control channels is not calibrated.')
            else:
                CONTROL_CHANNELS = CONTROL_CHANNELS_NEW

        # exit()

        # set interface variables based on control task
        if INTERFACE == 'V-shape':
            print('Interface: V-shape')
            UI_XY_ORIGIN = 0., -0.9
            UI_XY_SCALE = 1.5
            UI_ROTATION = 45
            UI_THETA_TARGET = 22.5
        elif INTERFACE == 'centre-around':
            print('Interface: centre-around')
            UI_XY_ORIGIN = 0., 0.
            UI_XY_SCALE = 1
            UI_ROTATION = 22.5
            UI_THETA_TARGET = 45
        else:
            raise ValueError('Unknown interface. Please choose either V-shape or centre-around.')

        exp.run(RealTimeControl())

