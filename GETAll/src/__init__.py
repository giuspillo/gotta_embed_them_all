import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
