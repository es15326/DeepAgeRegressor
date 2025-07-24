import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def mae(y_true : np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] == 0:
        return 0
    return np.abs(y_true - y_pred).mean()

def aar(y_true : np.ndarray, y_pred: np.ndarray) -> float:
    true_age_groups = np.clip(y_pred // 10, 0, 7)
    mae_score = mae(y_true, y_pred)
    
    # MAE per age group
    sigmas = []
    maes = []
    for i in range(8):
        idx = true_age_groups == i
        mae_age_group = mae(y_true[idx], y_pred[idx])
        maes.append(mae_age_group)
        sigmas.append((mae_age_group - mae_score) ** 2)

    sigma = np.sqrt(np.array(sigmas).mean())
    
    aar_score = max(0, 7 - mae_score) + max(0, 3 - sigma)
    
    return aar_score, mae_score, sigma, sigmas, maes


def main():
    print('aligned dlib only features')
    df = pd.read_csv('data/training_caip_contest.csv', 
                    header=None)
    
    # x = np.load('data/features_dlib_aligned.npy')
    # x = np.load('data/features_cr.npy')
    # x = np.load('data/features_vggface.npy')
    x = np.load('data/features_pred_resnext_aar.npy')
    # x = np.concatenate([np.load('data/features_dlib_non_aligned.npy'),
    #             np.load('data/features_non_aligned.npy')], axis=1)
    y = np.array(df.iloc[:, 1])
    
    train_index = np.loadtxt('data/train_index.txt', delimiter=',').astype('int')
    test_index = np.loadtxt('data/test_index.txt', delimiter=',').astype('int')
    
    x_train = x[train_index]
    y_train = y[train_index]
    
    x_test = x[test_index]
    y_test = y[test_index]
    
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, 
    #                                                     random_state=1)

    n_estimators = 100
    min_samples_leaf = 5
    max_features = 128
    random_state = 1
    rf = RandomForestRegressor(n_estimators=n_estimators, 
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, 
                                random_state=random_state,
                                n_jobs=64,
                                verbose=2)
    
    t0 = time.time()
    rf.fit(x_train, y_train)
    print(f'Time it took for training: {time.time() - t0:.3f} ms.')
    # dump(rf, f'data/rf_vggface2_{n_estimators}_{min_samples_leaf}_{max_features}_{random_state}.joblib') 
    out = np.clip(rf.predict(x_test).round(), 1, 81)
    print(f'rf MAE on validation: {np.abs(out - y_test).mean()}')
    AAR, MAE, *_, sigmas, maes = aar(y_test, out)

    print(f'ResNext: AAR on valdiation: {AAR}')
    maes_str = '\t'.join([f'{m:.03f}' for m in maes])
    sigmas_str = '\t'.join([f'{np.sqrt(m):.03f}' for m in sigmas])

    print("Summary: MAE\t" + "\t".join(["MAE" + str(i) for i in range(1, 9)]) + "\tAAR")
    print(f'Summary: {MAE:.03f}\t{maes_str}\t{AAR:.03f}')
    print(f'Summary: {0:.03f}\t{sigmas_str}\t{0:.03f}')

    
    # rrf = RefinedRandomForest(clf, C = 0.01, n_prunings = 0)
    # rrf.fit(x_train, y_train)

    # out = rrf.predict_proba(x_test).argmax(axis=1)
    # print(f'rrf MAE on validation: {np.abs(out - y_test).mean()}')
    # ARR, *_ = aar(y_test, out)
    # print(f'rrf AAR on validation: {ARR}')

if __name__ == '__main__':
    main()
