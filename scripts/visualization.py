from common import *



#------------------------------------------------------------

def show_feature_stats(train_df, test_df, feat_train, feat_test, feature_name):
    '''
    Inputs:
    =======
    feat_train, feat_test : vectors with shape (len, );
    feature_name : string
    '''
    train_df[f"{feature_name}"] = feat_train.reshape(-1, 1)
    test_df[f"{feature_name}"] = feat_test.reshape(-1, 1)
#     print(ks_2samp(train_df[f"{feature_name}"].values, test_df[f"{feature_name}"].values))
    
    fig, ax = plt.subplots(1,2, figsize=(14, 6))
    
    sns.distplot(feat_train, kde=False, label='Train', ax=ax[0]);
    sns.distplot(feat_test, kde=False, label='Test', ax=ax[0]);
    ax[0].set_yscale('log')
    ax[0].set_title(f'{feature_name} Distr Train / Test')
    ax[0].legend();
    ax[0].set_xlabel(f'{feature_name}')
    fig.tight_layout()
    
    sns.distplot(train_df[train_df['target'] == 1][f"{feature_name}"], label='Alice', kde=False)
    sns.distplot(train_df[train_df['target'] == 0][f"{feature_name}"], label='Other', kde=False)
    ax[1].set_yscale('log')
    ax[1].set_title(f'{feature_name} Distr Train / Test')
    ax[1].legend();
    ax[1].set_xlabel(f'{feature_name}')
    
    fig.tight_layout()