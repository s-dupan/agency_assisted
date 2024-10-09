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
from axopy.gui.canvas import Canvas, Text, Circle, Line
from axopy.pipeline import (Windower, Pipeline, Filter,
                            FeatureExtractor, Ensure2D, Block)
from PyQt5.QtWidgets import QDesktopWidget
from src.graphics import CalibWidget, Sinusoid
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
        if args.stick:
            pipeline = Pipeline(
                [Windower(1),
                 Ensure2D(orientation='row')
            ])
        else:
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
        self.prepare_timers()

    def prepare_timers(self):
        self.iti_timer = Counter(int(TRIAL_INTERVAL / READ_LENGTH))
        self.iti_timer.timeout.connect(self.finish_iti)
        self.trial_timer = Counter(int(TRIAL_LENGTH / READ_LENGTH))
        self.trial_timer.timeout.connect(self.finish_trial)
        self.score_timer = Counter(int(SCORE_LENGTH / READ_LENGTH))
        self.score_timer.timeout.connect(self.finish_score)

    def prepare_design(self, design):
        for b in range(N_BLOCKS):
            block = design.add_block()
            for noise in NOISE:
                    block.add_trial(attrs={'noise': noise})
            block.shuffle()                    

    def prepare_graphics(self, container):
        self.canvas = Canvas()
        container.set_widget(self.canvas)

        self.task_canvas = Canvas(border_color='black',bg_color = '#000000', draw_border=False)
        monitor = QDesktopWidget().screenGeometry(DISPLAY_MONITOR)

        # add canvas elements
        self.cursor = Circle(diameter=.1, color= 'green')
        self.cursor.hide()
        self.text_score = Text(text='test', color='white')
        self.text_score.pos = (-1,0)
        self.text_score.hide()
        
        # self.timepoints = np.arange(1, -1, -2*READ_LENGTH/TRIAL_LENGTH)
        # theta = 0       # phase
        # self.wave = WAVE_AMPL * np.sin(2 * np.pi * WAVE_FREQ * (self.timepoints * TRIAL_LENGTH/2) + theta)
        # self.wave_line = Sinusoid(x=self.wave, y=self.timepoints, color='white', linewidth=0.01)
        # # self.wave.hide()
        # self.task_canvas.add_item(self.wave_line)
        
        # wave that we want to follow
        self.timepoints = np.arange(1, -1, -2*READ_LENGTH/TRIAL_LENGTH)
        theta = np.pi     # phase
        self.wave = WAVE_AMPL * np.sin(2 * np.pi * WAVE_FREQ * (self.timepoints * TRIAL_LENGTH) + theta)
        # make wave twice as long, as we will move it upwards on the screen
        # you need to take off 1 element of timepoints as you will generate 1 point less for wave
        self.time_double = np.arange(1, -3, -2*READ_LENGTH/TRIAL_LENGTH)
        self.time_double = self.time_double[:-1]
        # we need to interpolate the wave to make it twice as long
        middle_points = self.wave[:-1] + np.diff(self.wave)/2
        self.wave_double = np.empty(2*self.wave.size - 1)
        self.wave_double[0::2] = self.wave
        self.wave_double[1::2] = middle_points
        self.wave_line = Sinusoid(x=self.wave_double, y=self.time_double, color='white', linewidth=0.01)
        self.wave_line.hide()
        self.task_canvas.add_item(self.wave_line)
        # prepare how much the sinusoid will move each step
        self.move_step = np.arange(0, 2, 2*READ_LENGTH/TRIAL_LENGTH)
    
        # self.task_canvas.add_item(self.basket)
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
            
    # def noise_level(self):
    #     noise_levels = np.random.choice([0, 0.1, 0.2, 0.3], size=20)

    #     for noise_level in noise_levels:
    #         self.run_trial(noise_level=noise_level)

    def run_trial(self, trial):
        self.iti_timer.reset()
        
        # create wave form for this trial
        # theta = 0
        # self.wave_iter = iter(self.wave)         # changed name of variable so that it doesn't overwrite the standard self.wave
        # self.timepoints_iter = iter(self.timepoints)        # changed name of variable so that it doesn't overwrite the standard self.timepoints
        
        self.wave_iter = iter(self.wave_double)   # watch out! does not work for score anymore if point of wave 
                                            # at top of screen does not align with point at y = 0
        # iterate over steps for wave to move
        self.move_iter = iter(self.move_step) 
        
        # self.noise = np.random.normal(0, scale=self.trial.attrs['noise'], size=len(self.wave))
        noise_ampl = self.trial.attrs['noise']
        self.noise =  noise_ampl * np.sin(2 * np.pi * WAVE_FREQ * (self.timepoints * TRIAL_LENGTH/2) + np.pi/2)
        self.noise[:int(1/READ_LENGTH)] = 0      # first second no noise
        self.noise[int(-1/READ_LENGTH):] = 0     # last second no noise
        self.noise_iter = iter(self.noise)

        trial.add_array('data_raw', stack_axis=1)
        trial.add_array('data_proc', stack_axis=1)
        trial.add_array('error', stack_axis=1)
        trial.add_array('wave', stack_axis=1)
        trial.add_array('noise', stack_axis=1)
        trial.add_array('cursor_position', stack_axis=1)
 
        self.pipeline.clear()
        self.connect(self.daqstream.updated, self.update_iti)

    def update_iti(self, data):
        data_proc = self.pipeline.process(data)

        if args.stick:
            # channels opposite of muscles, as we want the movement to the right be positive 
            muscle_t = 2*(data_proc[1][0] - data_proc[0][0])
        else:
            muscle_t = data_proc[0][CONTROL_CHANNELS[0]] - data_proc[0][CONTROL_CHANNELS[1]]   # muscle position at this time
        wave_t = 0         # wave position at this time 
        error = muscle_t - wave_t
        noise_t = 0
        self.cursor.pos = muscle_t, 0 #change to plot muscle_t , y = initial poit (top of screen)

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        self.trial.arrays['error'].stack(error)
        self.trial.arrays['wave'].stack(wave_t)
        self.trial.arrays['noise'].stack(noise_t)
        self.trial.arrays['cursor_position'].stack(muscle_t)

        self.iti_timer.increment()

    def update_trial(self,data, noise_level=0):
        data_proc = self.pipeline.process(data)
        if args.stick:
            muscle_t = 2*(data_proc[1][0] - data_proc[0][0])
        else:
            muscle_t = data_proc[0][CONTROL_CHANNELS[0]] - data_proc[0][CONTROL_CHANNELS[1]]   # muscle position at this time
        wave_t = next(self.wave_iter)
        time_t = next(self.move_iter)
        noise_t = next(self.noise_iter)
        
        self.wave_line.pos = 0,time_t
        
        cursor_position = muscle_t + noise_t
        self.cursor.pos = cursor_position, 0 #change to plot muscle_t
        
        error = muscle_t - wave_t
        # error = cursor_position - wave_t
        
        self.text_score.hide()

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        self.trial.arrays['error'].stack(error)
        self.trial.arrays['wave'].stack(wave_t)
        self.trial.arrays['noise'].stack(noise_t)
        self.trial.arrays['cursor_position'].stack(cursor_position)

        self.trial_timer.increment()

    def update_score(self, data):
        data_proc = self.pipeline.process(data)
        if args.stick:
            muscle_t = 2*(data_proc[1][0] - data_proc[0][0])
        else:
            muscle_t = data_proc[0][CONTROL_CHANNELS[0]] - data_proc[0][CONTROL_CHANNELS[1]]   # muscle position at this time
        wave_t = 0         # wave position at this time
        noise_t = 0
        error = muscle_t - wave_t
        self.cursor.pos = error, 0

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(np.transpose(data_proc))
        self.trial.arrays['error'].stack(error)
        self.trial.arrays['wave'].stack(wave_t)
        self.trial.arrays['noise'].stack(noise_t)
        self.trial.arrays['cursor_position'].stack(muscle_t)

        self.score_timer.increment()

    def finish_iti(self):
        self.wave_line.show()
        self.cursor.show()
        winsound.PlaySound('beep_1000Hz_200ms.wav',1)
        self.disconnect(self.daqstream.updated, self.update_iti)
        self.connect(self.daqstream.updated, self.update_trial)

    def finish_trial(self):
        self.cursor.hide()
        self.wave_line.hide()
        # calculate score
        self.score = 1. - np.mean(np.absolute(self.trial.arrays['error'].data))
        # self.text_score.qitem.setText("{:.0f} %".format(self.score*100))
        # self.text_score.show()
        # Save control type
        if args.trigno:
            self.trial.attrs['emg'] = 1
        else:
            self.trial.attrs['emg'] = 0
        self.text_score.qitem.setText("Rate your control over the completed trial: \n 1-9")
        self.text_score.show()
        self.score_timer.reset()
        self.disconnect(self.daqstream.updated, self.update_trial)
        self.connect(self.daqstream.updated, self.update_score)

    def finish_score(self):
        # Display prompt if that was the last trial in the block
        if self.trial.attrs['trial'] == N_TRIALS - 1:
            self.text_score.qitem.setText(" End of block {}. \n Press enter to continue.".format(
                self.trial.attrs['block'] + 1
            ))
        else:
            self.text_score.hide()
        self.trial.attrs['score'] = self.score
        self.writer.write(self.trial)
        self.disconnect(self.daqstream.updated, self.update_score)
        self.next_trial()

    def finish(self):
        self.daqstream.stop()
        self.finished.emit()
        self.text_score.hide()

    def key_press(self, key):
        if key == util.key_escape:
            self.finish()
        else:
            super().key_press(key)

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
    source.add_argument('--stick', action='store_true')
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
    channels_task = cp.getint('control', 'channels_task')

    if args.trigno:
        from pytrigno import TrignoEMG
        S_RATE = 2000.
        dev = TrignoEMG(channels=CHANNELS, zero_based=False,
                        samples_per_read=int(S_RATE * READ_LENGTH))
    elif args.myo:
        import myo
        from pydaqs.myo import MyoEMG
        S_RATE = 200.
        if HIGHCUT > 100:
            # due to low sampling rate, code won't work if highcut filter is too high
            HIGHCUT = 100
        myo.init(
            sdk_path=r'C:\Users\Sigrid\coding\myo-python\myo-sdk-win-0.9.0')
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
    elif args.stick:
        from pydaqs.stick import Stick
        S_RATE = 1/READ_LENGTH
        dev = Stick(rate=S_RATE, dev_id=0, mode='divaxis')

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

        TRIAL_INTERVAL = cp.getfloat('control', 'trial_interval')
        TRIAL_LENGTH = cp.getfloat('control', 'trial_length')
        SCORE_LENGTH = cp.getfloat('control', 'score_present')
        DISPLAY_MONITOR = cp.getint('control', 'display_monitor')
        NOISE_LEVELS = list(map(float, (cp.get('experiment', 'noise_levels').split(','))))
        print(NOISE_LEVELS)
        # List of noise levels = length of trials
        # NOISE = np.array(NOISE_LEVELS*(N_TRIALS/len(NOISE_LEVELS))) 
        NOISE = np.tile(NOISE_LEVELS, int(N_TRIALS/len(NOISE_LEVELS)))
        
        # download calibration info
        # no calibration for stick
        if ~args.stick:
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

        # read in wave information
        WAVE_FREQ = cp.getfloat('experiment', 'wave_frequency')
        WAVE_AMPL = cp.getfloat('experiment', 'wave_amplitude')

        exp.run(RealTimeControl())

