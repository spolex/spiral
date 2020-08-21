import json
import logging
from os import path
from datetime import datetime


class Properties:

    schema = ['x', 'y', 'timestamp', 'pen_up', 'azimuth', 'altitude', 'pressure']
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
    subject_id = 'subject_id'

    root_path = "Z:/elekin"
    rdo_root_path = path.join(root_path,"02-RESULTADOS/03-HANDWRITTING")
    rdo_log_path = path.join(rdo_root_path, '01-LOG')

    datasources_path = "00-DATASOURCES/02-ETHW"
    ct_root_path = path.join(root_path, datasources_path, "Controles30jun14/")
    et_root_path = path.join(root_path, datasources_path, "Protocolo temblor")
    file_list_path = path.join(root_path, datasources_path, "ETONA.txt")

    controls = 27
    et = 23
    coefficients = [10] + list(range(14, 26, 1)) + [30, 50]
    # coefficients = [17]
    h5file = path.join(rdo_root_path, "00-OUTPUT/archimedean-")
    filename_ds = path.join(rdo_root_path, "00-OUTPUT/archimedean_ds-")
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

    mode = 'r'

    resample = 4096

    log_conf_path = 'conf/logging.conf'
    log_filename = 'archimedean-{:%Y-%m-%d}.log'.format(datetime.now())
    log_file_path = path.join(rdo_log_path, log_filename)

    # param_grid = {"svc__kernel": ["rbf"],
    #              "svc__C": np.logspace(-5, 5, num=25, base=10),
    #              "svc__gamma": np.logspace(-9, 9, num=25, base=10)}

    def experiment_config(filename="conf/config.json"):
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