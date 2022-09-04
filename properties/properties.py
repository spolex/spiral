import json
import logging
from os import path
from datetime import datetime, date

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

metadata_path = path.join(doc_path, "metadata-202208-v1.csv")

coefficients = [17] + list(range(14, 26, 1)) + [30, 50]
coefficients = [17]

file = path.join(rdo_root_path, "biodarw-")
filename_ds = path.join(rdo_root_path, "biodarw_ds-")
extension = ".csv"

rd_filename= path.join(rdo_root_path, 'residues_{}_{}.csv')
rd_feat_filename=path.join(rdo_root_path, 'residues_feat_{}_{}.csv')
rd_feat_norm_filename=path.join(rdo_root_path, 'residues_feat_norm_{}_{}.csv')
rd_feat_relief_filename = path.join(rdo_root_path, 'residues_feat_relief_{}_{}.csv')


r_filename = path.join(rdo_root_path, 'radius_{}.csv')
r_feat_filename = path.join(rdo_root_path, path.join(rdo_root_path, 'radius_feat_{}.csv'))
r_feat_norm_filename = path.join(rdo_root_path, 'radius_feat_norm_{}.csv')
r_feat_relief_filename = path.join(rdo_root_path, 'radius_feat_relief_{}.csv')

label_filename =path.join(rdo_root_path,'binary_labels_{}.csv')
level_filename =path.join(rdo_root_path, 'level_{}.csv')


log_conf_path = './conf/logging.conf'
log_filename = 'archimedean-{:%Y-%m-%d}.log'.format(datetime.now())
log_file_path = path.join(rdo_log_path, log_filename)

schema = ['x', 'y', 'timestamp', 'pen_up', 'azimuth', 'altitude', 'pressure']
resample = 4096

seed = 38

