from common import *
from global_common import *


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()       
        model = models.resnet18(True)
        self.bone = list(model.children())[:-2]
        self.bottle = self.bone[-1][-1] # Bottleneck
        self.n_ft = list(self.bottle.children())[-1].num_features
        self.head = nn.Sequential( 
            *self.bone,
            AdaptiveConcatPool2d(sz=1),
            Flatten(),
            nn.BatchNorm1d(num_features=self.n_ft*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=self.n_ft*2, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=102, bias=True),
        )                    
    def forward(self, x):
        output = self.head(x)        
        return output
    
    
    
def run_tree(lgb_params, train, test, folds=10, early_stop=200, verbose_eval=100):
    # Prepare dataset for training
    cols_to_drop = [
        'id',
        'target',
    ]

    X = train.drop(cols_to_drop, axis=1, errors='ignore')
    y = train.target.values
    
    cat_features = ['os_category', 'device_type', 'service_7_flag_m1', 
                       'service_7_flag_m2', 'service_7_flag_m3', 'sim_count'] 
    
    

    id_test = test.id.values
    X_test = test.drop(cols_to_drop[0], axis=1, errors='ignore')

    print('train.shape = {}, test.shape = {}'.format(train.shape, test.shape))

    # Build the model
    cnt = 0
    p_buf = []
    n_splits = folds
    n_repeats = 1
    kf = StratifiedKFold(
        n_splits=n_splits, 
        )
    err_buf = []   

    n_features = X.shape[1]

    for train_index, valid_index in kf.split(X, y):
        print('Fold {}/{}*{}'.format(cnt + 1, n_splits, n_repeats))
        params = lgb_params.copy() 

        lgb_train = lgb.Dataset(
            X.iloc[train_index], 
            y[train_index], 
            categorical_feature=cat_features,

            )
        lgb_train.raw_data = None

        lgb_valid = lgb.Dataset(
            X.iloc[valid_index], 
            y[valid_index],
            categorical_feature=cat_features,
            )
        lgb_valid.raw_data = None

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100000,
            valid_sets=[lgb_train, lgb_valid],
            early_stopping_rounds=early_stop, 
            verbose_eval=verbose_eval,
        )

        if cnt == 0:
            importance = model.feature_importance()
            model_fnames = model.feature_name()
            tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
            tuples = [x for x in tuples if x[1] > 0]
            print('Important features:')
            for i in range(60):
                if i < len(tuples):
                    print(tuples[i])
                else:
                    break

            del importance, model_fnames, tuples

        p = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)
        err = roc_auc_score(y[valid_index], p)

        print('{} auc: {}'.format(cnt + 1, err))

        p = model.predict(X_test, num_iteration=model.best_iteration)
        if len(p_buf) == 0:
            p_buf = np.array(p, dtype=np.float16)
        else:
            p_buf += np.array(p, dtype=np.float16)
        err_buf.append(err)


        cnt += 1

        del model, lgb_train, lgb_valid, p
        gc.collect

    err_mean = np.mean(err_buf)
    err_std = np.std(err_buf)
    print('auc = {:.6f} +/- {:.6f}'.format(err_mean, err_std))

    preds = p_buf/cnt
    return preds
    
    
def run_tree_pseudo(lgb_params, train, add, test, folds=10, early_stop=200, verbose_eval=100):
    # Prepare dataset for training
    cols_to_drop = [
        'id',
        'target',
    ]

    X = train.drop(cols_to_drop, axis=1, errors='ignore')
    y = train.target.values
    X_add = add.drop(cols_to_drop, axis=1, errors='ignore')

    
    cat_features = ['os_category', 'device_type']

    id_test = test.id.values
    X_test = test.drop(cols_to_drop[0], axis=1, errors='ignore')

    print('train.shape = {}, test.shape = {}'.format(train.shape, test.shape))

    # Build the model
    cnt = 0
    p_buf = []
    n_splits = folds
    n_repeats = 1
    kf = StratifiedKFold(
        n_splits=n_splits, 
        )
    err_buf = []   

    n_features = X.shape[1]

    for train_index, valid_index in kf.split(X, y):
        print('Fold {}/{}*{}'.format(cnt + 1, n_splits, n_repeats))
        params = lgb_params.copy() 

        lgb_train = lgb.Dataset(
            pd.concat([ X.iloc[train_index], X_add ], axis=0),
            np.concatenate([ y[train_index], add.target.values ]),
            categorical_feature=cat_features,
            )
        lgb_train.raw_data = None

        lgb_valid = lgb.Dataset(
            X.iloc[valid_index], 
            y[valid_index],
            categorical_feature=cat_features,
            )
        lgb_valid.raw_data = None

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100000,
            valid_sets=[lgb_train, lgb_valid],
            early_stopping_rounds=early_stop, 
            verbose_eval=verbose_eval,
        )

        if cnt == 0:
            importance = model.feature_importance()
            model_fnames = model.feature_name()
            tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
            tuples = [x for x in tuples if x[1] > 0]
            print('Important features:')
            for i in range(60):
                if i < len(tuples):
                    print(tuples[i])
                else:
                    break

            del importance, model_fnames, tuples

        p = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)
        err = roc_auc_score(y[valid_index], p)

        print('{} auc: {}'.format(cnt + 1, err))

        p = model.predict(X_test, num_iteration=model.best_iteration)
        if len(p_buf) == 0:
            p_buf = np.array(p, dtype=np.float16)
        else:
            p_buf += np.array(p, dtype=np.float16)
        err_buf.append(err)


        cnt += 1

        del model, lgb_train, lgb_valid, p
        gc.collect

    err_mean = np.mean(err_buf)
    err_std = np.std(err_buf)
    print('auc = {:.6f} +/- {:.6f}'.format(err_mean, err_std))

    preds = p_buf/cnt
    return preds
    
    
