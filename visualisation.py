"""creation of connectivity (parietal retreat)."""
# from nilearn import connectome, signal
import numpy as np
# from joblib import load

# import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import connectome
# import os
# from fnmatch import fnmatch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import time
from load_data import load_data_all_subjects
import seaborn as sns

sns.set_context('poster')
sns.set_style(style='white')
start = time.time()
df, features, ts_all = load_data_all_subjects(kind="correlation")
end = time.time()

print('Loading time was ' + str(end - start) + 's')
print(features.shape)
print(df.shape)

plt.close('all')

orders = ['LR', 'RL']
whoistrain = [1, 2]
y_hats = []
scores = []

for j, session_nb in enumerate(whoistrain):
    for i, order in enumerate(orders):

        # Mask creation
        first_sess_msk = np.array((df['session_nb'] == session_nb) &
                                  (df['order'] == order))
        secnd_sess_msk = np.array((df['session_nb'] != session_nb) &
                                  (df['order'] == order))

        X_train = features[first_sess_msk, :]
        X_test = features[secnd_sess_msk, :]

        y = np.array(df['id'])
        y_train = y[first_sess_msk]
        y_test = y[secnd_sess_msk]

        pca_visu = PCA(0.9)
        X_train_pca = pca_visu.fit_transform(X_train)
        X_test_pca = pca_visu.transform(X_test)

        # Visualisation:
        fig, ax = plt.subplots()
        plt.title('PCA of connectoms (first two axis) \n intersession '
                  'displacement for a given subject ')
        X1 = X_train_pca[:, [0, 1]]
        X2 = X_test_pca[:, [0, 1]]
        dX = X2 - X1

        ax.scatter(X1[:, 0], X1[:, 1], c='k', marker='.', label='Session 1')
        ax.scatter(X2[:, 0], X2[:, 1], c='b', marker='.', label='Session 2')
        Q = plt.quiver(X1[:, 0], X1[:, 1], dX[:, 0], dX[:, 1],
                       scale_units='xy', scale=1, angles='xy', alpha=0.5)
        plt.legend()
        plt.xlabel('First PCA axis')
        plt.ylabel('Second PCA axis')
        plt.show()
        plt.savefig('PCA.png')
        dX.mean(axis=0)

        # Density visualization
        # plt.figure()
        # sns.kdeplot(np.linalg.norm(X_train - X_test, axis=1), clip=[0, 25])
        # plt.show()

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        y_hats.append(y_hat)
        scores.append(accuracy_score(y_hat, y_test))

print(scores)

orders = ['LR', 'RL']
whoistrain = [1, 2]
kinds = ["correlation", "partial correlation",
         "tangent", "covariance", "precision"]
performances = []

# test 1, 2, 4, 8
subs_fact = 8


for i, kind in enumerate(kinds):
    for j, session_nb in enumerate(whoistrain):
        for k, order in enumerate(orders):

            first_all_sess_msk = np.array((df['session_nb'] == session_nb) &
                                          (df['order'] == order))
            secnd_all_sess_msk = np.array((df['session_nb'] != session_nb) &
                                          (df['order'] == order))

            ts_all_array = np.array(ts_all, dtype=object)
            ts_fst = list(ts_all_array[first_all_sess_msk][::subs_fact])
            ts_secnd = list(ts_all_array[secnd_all_sess_msk][::subs_fact])

            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            indiv_connect_mat = conn_measure.fit_transform(ts_fst)
            connectivity_fst = connectome.sym_to_vec(indiv_connect_mat,
                                                     discard_diagonal=True)

            indiv_connect_mat = conn_measure.transform(ts_secnd)
            connectivity_secnd = connectome.sym_to_vec(indiv_connect_mat,
                                                       discard_diagonal=True)

            X_train_all = connectivity_fst
            y_train = y[first_all_sess_msk][::subs_fact]

            X_test_all = connectivity_secnd
            y_test = y[secnd_all_sess_msk][::subs_fact]

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X_train_all, y_train)
            y_hat = clf.predict(X_test_all)
            print(accuracy_score(y_hat, y_test))

            curent_df = [accuracy_score(y_hat, y_test), order,
                         session_nb, kind]
            performances.append(curent_df)


df_barplot = pd.DataFrame(performances, columns=['Accuracy', 'order',
                                                 'session_nb', 'Estimator'])

df_barplot['Train Session'] = df_barplot['order'] + \
    df_barplot['session_nb'].astype(str)
replacements = {'Train Session': {'RL1': 'Session 1 - RL',
                                  'RL2': 'Session 2 - RL',
                                  'LR1': 'Session 1 - LR',
                                  'LR2': 'Session 2 - LR'}}
df_barplot = df_barplot.replace(to_replace=replacements)


# Visualisation
plt.figure()
ax = sns.barplot(x="Estimator", y="Accuracy", hue="Train Session", ci=None,
                 data=df_barplot)
plt.title('Accuracy : identify patient from one session to another ' +
          '(sub=%s)' % subs_fact)
plt.show()

plt.savefig('barplot_perf_subsampled' + str(subs_fact) + '.png')
