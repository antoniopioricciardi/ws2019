from pathlib import Path
import os

RESOURCES_DIR_PATH = Path('../resources')
GRAPH_PLOTS_DIR_PATH = RESOURCES_DIR_PATH / 'graph_plots'
DS1_PATH = RESOURCES_DIR_PATH / 'ds-1.tsv'
DS2_PATH = RESOURCES_DIR_PATH / 'ds-2.tsv'

'''Path to plot resulting merged topics as graphs'''
RES_PATH = os.getcwd() + '/../results/'
TASK_1_PATH = RES_PATH + 'TASK_1/'
TASK_2_PATH = RES_PATH + 'TASK_2/'
DOTS_PATH = TASK_2_PATH + 'dots/'
IMAGES_PATH = TASK_2_PATH + 'images/'
