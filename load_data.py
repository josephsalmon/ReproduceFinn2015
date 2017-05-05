"""creation of connectivity (parietal retreat)."""
from nilearn import connectome, signal
import numpy as np
from joblib import load

# import seaborn as sns
import os
# from fnmatch import fnmatch
import pandas as pd

from tempfile import mkdtemp
from joblib import Memory

# cache memory with joblib
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

individual_connectivity_matrices = {}
indiv_connect_mat = {}
mean_connect_mat = {}

dirname = '/home/jo/Documents/Mes_papiers/Retraite2017/Data/HCP_timeseries'


def connnect_creation(df_int, kind='correlation'):
    """Create connectivity."""
    order = df_int[1]
    session_nb = str(df_int[2])
    filename_id = df_int[3][:-9]
    ts_dirty = load(filename_id + 'basc064' + '/' + 'rfMRI_REST' +
                    session_nb + '_' + order + '_raw')

    ts_ortho = np.loadtxt(filename_id + 'confounds' + '/' +
                          'rfMRI_REST' + session_nb + '_' +
                          order + '_Movement_Regressors.txt')
    ts = signal.clean(ts_dirty, detrend=True, standardize=True,
                      confounds=ts_ortho, low_pass=None,
                      high_pass=None, t_r=0.72, ensure_finite=False)

    conn_measure = connectome.ConnectivityMeasure(kind=kind)
    indiv_connect_mat[kind] = conn_measure.fit_transform([ts])
    mean_connect_mat[kind] = indiv_connect_mat[kind].mean(axis=0)
    connectivity_coefs = connectome.sym_to_vec(indiv_connect_mat[kind],
                                               discard_diagonal=True)
    return connectivity_coefs, ts


@memory.cache
def load_data_all_subjects(kind='correlation'):
    """Loading."""
    data = []
    features_lst = []
    nb_subjects = 0
    ts_all = []
    for path, subdirs, files in os.walk(dirname):
        subject_id = path[len(dirname) + 1::]
        filename_id = dirname + '/' + subject_id + '/'
        # print(filename_id + 'rfMRI_REST1_LR_raw')
        if ('basc064' in subdirs):
            filename = dirname + '/' + subject_id + '/basc064/'
            d1 = os.path.exists(filename + 'rfMRI_REST1_LR_raw')
            d2 = os.path.exists(filename + 'rfMRI_REST2_LR_raw')
            d3 = os.path.exists(filename + 'rfMRI_REST1_RL_raw')
            d4 = os.path.exists(filename + 'rfMRI_REST2_RL_raw')
            if d1 and d2 and d3 and d4:
                nb_subjects += 1

                df_int = [subject_id, 'LR', 1, filename_id + '/basc064/']
                data.append(df_int)
                connectivity_coefs, ts = connnect_creation(df_int,
                                                           kind='correlation')
                features_lst.append(connectivity_coefs)
                ts_all.append(ts)

                df_int = [subject_id, 'LR', 2, filename_id + '/basc064/']
                data.append(df_int)
                connectivity_coefs, ts = connnect_creation(df_int,
                                                           kind='correlation')
                features_lst.append(connectivity_coefs)
                ts_all.append(ts)

                df_int = [subject_id, 'RL', 1, filename_id + '/basc064/']
                data.append(df_int)
                connectivity_coefs, ts = connnect_creation(df_int,
                                                           kind='correlation')
                features_lst.append(connectivity_coefs)
                ts_all.append(ts)

                df_int = ([subject_id, 'RL', 2, filename_id + '/basc064/'])
                data.append(df_int)
                connectivity_coefs, ts = connnect_creation(df_int,
                                                           kind='correlation')
                features_lst.append(connectivity_coefs)
                ts_all.append(ts)

    df = pd.DataFrame(data, columns=['id', 'order', 'session_nb', 'filename'])
    features = np.array(features_lst).squeeze()
    return df, features, ts_all
