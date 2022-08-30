import json
import logging
from os import path
from datetime import datetime

def experiment_config(filename="config.json"):
    """
    Load experiment configuration file
    :param filename:
    :return: dict with configuration
    """
    try:
        with open(filename) as config_file:
            return json.load(config_file)
    except IOError:
        logging.error(filename + " doesn't exist, an existing file is needed like config file")
        return -1

def update_experiment(self, config, filename):
    """

    :param config:
    :param experiment: new experiment configuration
    :param filename:
    :return:
    """
    try:
        with open(filename, 'w') as config_file:
            json.dump(config, config_file, indent=2)
            return 0
    except IOError:
        logging.error(filename + " doesn't exist, an existing file is needed like config file")
        return -1

features_names = ["mean_abs_val(L)"
    , "np.var(L)"
    , "root_mean_square(L)"
    , "log_detector(L)"
    , "wl(L)"
    , "np.nanstd(L)"
    , "diff_abs_std(L)"
    , "higuchi(L)"
    , "mfl(L)"
    , "myo(L)"
    , "iemg(L)"
    , "ssi(L)"
    , "zc(L)"
    , "ssc(L)"
    , "wamp(L)"
    , "p_max(Pxx, L)"
    , "f_max(Pxx)"
    , "mp(Pxx)"
    , "tp(Pxx)"
    , "meanfreq(L)"
    , "medfreq(L)"
    , "std_psd(Pxx)"
    , "mmnt(Pxx, order=1)"
    , "mmnt(Pxx, order=2)"
    , "mmnt(Pxx, order=3)"
    , "kurt(Pxx)"
    , "skw(Pxx)"]
rdo_schema = ['r', 'rd']

root_path = "/data/elekin"
doc_path = path.join(root_path, "doc/")
rdo_root_path = path.join(root_path,"data/results/handwriting")
rdo_log_path = path.join(rdo_root_path, 'log')

datasources_path = "data/origin/ethw"
ct_root_path = path.join(root_path, datasources_path, "Controles30jun14/")
et_root_path = path.join(root_path, datasources_path, "protocolo_temblor")
file_list_path = path.join(doc_path, "ETONA.txt")
metadata_path = path.join(doc_path, "metadata-202106-v1.csv")

controls = 27
et = 23
coefficients = [10] + list(range(14, 26, 1)) + [30, 50]
coefficients = [10]
h5file = path.join(rdo_root_path, "archimedean-")
filename_ds = path.join(rdo_root_path, "archimedean_ds-")
extension = ".h5"

r_ct = 'r_ct'  # radius
r_et = 'r_et'  # radius
rd_ct = 'rd_ct'  # residues
rd_et = 'rd_et'  # residues

rd_ct_fe = 'rd_ct_fe'  # residues based features ct_rd
rd_et_fe = 'rd_et_fe'  # et_rd

r_ct_fe = 'r_ct_fe'  # radius based features ct_r
r_et_fe = 'r_et_fe'  # et_r

train_rd = 'train_rd'
train_r = 'train_r'
X = train_rd
labels = 'labels'
subject_id = 'subject_id'

mode = 'r'

log_conf_path = './conf/logging.conf'
log_filename = 'archimedean-{:%Y-%m-%d}.log'.format(datetime.now())
log_file_path = path.join(rdo_log_path, log_filename)

schema = ['x', 'y', 'timestamp', 'pen_up', 'azimuth', 'altitude', 'pressure']
resample = 4096

