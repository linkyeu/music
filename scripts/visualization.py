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
    
    
def show_feature(train_df, test_df, feature_name):
    '''
    Inputs:
    =======
    feat_train, feat_test : vectors with shape (len, );
    feature_name : string
    '''
    
    print(np.corrcoef(train_df[f"{feature_name}"].values, 
                      train_df['target'].values))
    
    fig, ax = plt.subplots(1,2, figsize=(14, 6))
    
    # Здесь мы смотрим распределение переменной в обуч. и тестовой выборках
    sns.distplot(train_df[f"{feature_name}"], kde=False, label='Train', ax=ax[0]);
    sns.distplot(test_df[f"{feature_name}"], kde=False, label='Test', ax=ax[0]);
    ax[0].set_yscale('log')
    ax[0].set_title(f'{feature_name} Distr Train / Test')
    ax[0].legend();
    ax[0].set_xlabel(f'{feature_name}')
    fig.tight_layout()
    
    # Здесь мы смотрим распределение переменной для целевых значений и остальных
    sns.distplot(train_df[train_df['target'] == 1][f"{feature_name}"], label='Alice', kde=False)
    sns.distplot(train_df[train_df['target'] == 0][f"{feature_name}"], label='Other', kde=False)
    ax[1].set_yscale('log')
    ax[1].set_title(f'{feature_name} Distr Train')
    ax[1].legend();
    ax[1].set_xlabel(f'{feature_name}')
    
    fig.tight_layout()
    
    
    
def plot_categoricals(x, y, data, annotate = True):
    """Plot counts of two categoricals.
    Size is raw count for each grouping.
    Percentages are for a given value of y."""
    
    # Raw counts 
    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))
    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})
    
    # Calculate counts for each group of x and y
    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))
    
    # Rename the column and reset the index
    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()
    counts['percent'] = 100 * counts['normalized_count']
    
    # Add the raw count
    counts['raw_count'] = list(raw_counts['raw_count'])
    
    plt.figure(figsize = (14, 10))
    # Scatter plot sized by percent
    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',
                s = 100 * np.sqrt(counts['raw_count']), marker = 'o',
                alpha = 0.6, linewidth = 1.5)
    
    if annotate:
        # Annotate the plot with text
        for i, row in counts.iterrows():
            # Put text with appropriate offsets
            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()), 
                               row[y] - (0.15 / counts[y].nunique())),
                         color = 'navy',
                         s = f"{round(row['percent'], 1)}%")
        
    # Set tick marks
    plt.yticks(counts[y].unique())
    plt.xticks(counts[x].unique())
    
    # Transform min and max to evenly space in square root domain
    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))
    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))
    
    # 5 sizes for legend
    msizes = list(range(sqr_min, sqr_max,
                        int(( sqr_max - sqr_min) / 5)))
    markers = []
    
    # Markers for legend
    for size in msizes:
        markers.append(plt.scatter([], [], s = 100 * size, 
                                   label = f'{int(round(np.square(size) / 100) * 100)}', 
                                   color = 'lightgreen',
                                   alpha = 0.6, edgecolor = 'k', linewidth = 1.5))
        
    # Legend and formatting
    plt.legend(handles = markers, title = 'Counts',
               labelspacing = 3, handletextpad = 2,
               fontsize = 16,
               loc = (1.10, 0.19))
    
    plt.annotate(f'* Size represents raw count while % is for a given y value.',
                 xy = (0, 1), xycoords = 'figure points', size = 10)
    
    # Adjust axes limits
    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), 
              counts[x].max() + (6 / counts[x].nunique())))
    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), 
              counts[y].max() + (4 / counts[y].nunique())))
    plt.grid(None)
    plt.xlabel(f"{x}"); plt.ylabel(f"{y}"); plt.title(f"{y} vs {x}");