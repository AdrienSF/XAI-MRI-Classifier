import pathlib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def resample_balanced(volume_filename: str):
    volumes = pd.read_csv(volume_filename)

    # add other label format
    volumes["Target"] = volumes["Target"].astype('category')
    volumes['Target_cat'] = volumes['Target'].cat.codes 


    # Subsampling from the dataset 
    # Creating the Minority
    volumes['Target'].value_counts()
    df_minority = volumes[volumes.Target=='AD']
    df_minority

    #Creating a Majority for CN 
    df_CN = volumes[volumes.Target=='CN']
    # Downsample majority class
    cn_sampled = resample(df_CN, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results


    #Creating a Majority for MCI 
    df_MCI = volumes[volumes.Target=='MCI']
    # Downsample majority class
    mci_sampled = resample(df_MCI, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results

    #Creating a Majority for MCI 
    df_SPR = volumes[volumes.Target=='SPR']
    # Downsample majority class
    spr_sampled = resample(df_SPR, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results

    

    # creating a new df with subsampled dfs
    pdList = [cn_sampled, mci_sampled, spr_sampled]  # List of your dataframes
    df_majority = pd.concat(pdList)
    df_majority["Target"].value_counts()


    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_minority, df_majority])


    limited = df_downsampled
    limited = limited.drop(columns=['Target', 'Target_cat'])


    return limited, df_downsampled




def save_shap_plots(clf, data, labels, plot_labels, clf_name):
    classmap = {0: 'AD', 1: 'CN', 2: 'MCI', 3:'SPR'}
    clf_name = clf_name
    # make sure directories exist
    pathlib.Path(clf_name+'_plots').mkdir(parents=True, exist_ok=True)
    start_path = clf_name+'_plots/'+clf_name

    # train on all data, we are not interested in test accuracy but rather which features the classifier finds important.
    # Thus we should train it on as much data as we can in order to more accurately measure feature salience 
    clf.fit(data, labels)
    # explain predictions
    kexplainer = shap.KernelExplainer(clf.predict_proba, data)
    shap_values = kexplainer.shap_values(data)


    # save plot of overall salience
    plt.title(clf_name + ' All classes')
    shap.summary_plot(shap_values, data, show=False, class_names=['AD', 'CN', 'MCI', 'SPR'])
    plt.savefig(start_path+'mean(|SHAP_val|)', bbox_inches='tight')
    plt.close()
    # save plots of impact on output for each class
    for i in range(4):
        plt.title(clf_name + ' class: ' + classmap[i])
        shap.summary_plot(shap_values[i], data, class_names=plot_labels, show=False)
        plt.savefig(start_path+'SHAP_val_class_'+classmap[i], bbox_inches='tight')
        plt.close()
        plt.title(clf_name + ' class: ' + classmap[i])
        shap.summary_plot(shap_values[i], data, class_names=plot_labels, show=False, plot_type='bar')
        plt.savefig(start_path+'mean(|SHAP_val|)'+classmap[i], bbox_inches='tight')
        plt.close()


# load in scaled or unscaled data
data, data_and_labels = resample_balanced('scaled_volumes.csv')
# data, data_and_labels = resample_balanced('volumes.csv')


classifiers = [
    # RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0),
    # KNeighborsClassifier(),
   SVC(kernel='linear', probability=True),
#    SVC(kernel='sigmoid', probability=True),
#    SVC(kernel='rbf', probability=True),
#    GradientBoostingClassifier(),
#    AdaBoostClassifier(),
    # LinearDiscriminantAnalysis(),
    # MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
]


for clf in classifiers:
   try:
    save_shap_plots(clf, data, data_and_labels['Target_cat'], data_and_labels['Target'], str(clf))
   except Exception as e:
       print(str(clf)+' failed:')
       print(e)
       continue

