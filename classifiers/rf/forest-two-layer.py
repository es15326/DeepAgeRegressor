import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

plt.style.use('ggplot')

def ind2onehot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b


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


if __name__ == '__main__':

    df = pd.read_csv('data/training_caip_contest.csv', 
                    header=None)
    

    x = np.load('data/features_pred_resnext_aar.npy')
    y = np.array(df.iloc[:, 1])
    
    train_index = np.loadtxt('data/train_index.txt', delimiter=',').astype('int')
    test_index = np.loadtxt('data/test_index.txt', delimiter=',').astype('int')
    
    x_train = x[train_index]
    y_train = y[train_index]
    y_train_group = np.clip(y_train // 10, 0, 7)
    
    x_test = x[test_index]
    y_test = y[test_index]
    y_test_group = np.clip(y_test // 10, 0, 7)

    n_estimators = 100
    min_samples_leaf = 5
    max_features = 128
    random_state = 1
    rfc = RandomForestClassifier(n_estimators=n_estimators, 
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, 
                                random_state=random_state,
                                n_jobs=64,
                                verbose=2)
    
    rfc.fit(x_train, y_train_group)
    
    # rfc = joblib.load('data/two_layer_rf_rfr_100_5_128_1.joblib')
    
    y_train_prob = rfc.predict_proba(x_train)
    # x_train = np.concatenate([x_train, y_train_group.reshape(-1, 1)], axis=1)
    x_train = np.concatenate([x_train, y_train_prob], axis=1)
    # x_train = np.concatenate([x_train, ind2onehot(y_train_group)], axis=1)
    

    rfr = RandomForestRegressor(n_estimators=n_estimators, 
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, 
                                random_state=random_state,
                                n_jobs=64,
                                verbose=2)
    
    # t0 = time.time()

    rfr.fit(x_train, y_train)
    # rfr = joblib.load('data/two_layer_rf_rfc_100_5_128_1.joblib')
    # print(f'Time it took for training: {time.time() - t0:.3f} ms.')
    # dump(rfc, f'data/two_layer_rf_rfc_{n_estimators}_{min_samples_leaf}_{max_features}_{random_state}.joblib') 
    # dump(rfr, f'data/two_layer_rf_rfr_{n_estimators}_{min_samples_leaf}_{max_features}_{random_state}.joblib') 
    outc = rfc.predict_proba(x_test)
    y_pred_group = outc.argmax(axis=1)
    conf_mat = confusion_matrix(y_test_group, y_pred_group, normalize='true')
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='.2f', cbar=False,
                xticklabels=["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"],
                yticklabels=["< 10", "10-19", "20-29", "30-39", "40-24", "50-59", "60-69", "> 70"])
    plt.savefig('heaptmap.png')
    for i in range(8):
        y_pred_i = y_pred_group == i
        y_true_i = y_test_group == i
        print(f"Accuracy for group {i}: {accuracy_score(y_true_i, y_pred_i)}")

    print(f"Classifier accuracy: {accuracy_score(y_test_group, y_pred_group):.3f}")
    print(f"Classifier confusion matrix: \n {conf_mat}")
    print(f"Report: \n {classification_report(y_test_group, y_pred_group)}")
    # x_test = np.concatenate([x_test, y_test_group.reshape(-1, 1)], axis=1)
    x_test = np.concatenate([x_test, outc], axis=1)
    out = np.clip(rfr.predict(x_test).round(), 1, 81)

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
