import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut

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






# load volume data
limited, full = resample_balanced('volumes.csv')
data, labels = limited.to_numpy(), full['Target_cat'].to_numpy()

# run leave-one-out cross-validation
accs = []
clf = SVC(kernel='linear', probability=True)
loo = LeaveOneOut()
for train_index, test_index in loo.split(data):
    clf.fit(data[train_index], labels[train_index])
    accs.append(clf.score(data[test_index], labels[test_index]))

acc = np.mean(accs)
print("linear svm leave-one-out cross-validation accuracy (balanced unscaled dataset):", acc)


# load scaled data
limited, full = resample_balanced('scaled_volumes.csv')
data, labels = limited.to_numpy(), full['Target_cat'].to_numpy()

# run leave-one-out cross-validation
accs = []
clf = SVC(kernel='linear', probability=True)
loo = LeaveOneOut()
for train_index, test_index in loo.split(data):
    clf.fit(data[train_index], labels[train_index])
    accs.append(clf.score(data[test_index], labels[test_index]))

acc = np.mean(accs)
print("linear svm leave-one-out cross-validation accuracy (balanced scaled dataset):", acc)