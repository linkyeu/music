from common import *


#---------------------------------------------------------------------------
# Кодировка сайтов Кашинсонского 
#---------------------------------------------------------------------------

def add_time_features(df, X_sparse):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')
    X = hstack([X_sparse, morning.values.reshape(-1, 1), 
                day.values.reshape(-1, 1), evening.values.reshape(-1, 1), 
                night.values.reshape(-1, 1)])
    return X

def add_time_features_single_var(x):
    hour = x.hour
    if ((hour >= 7) & (hour <= 11)):
        part_of_day = 0
    elif ((hour >= 12) & (hour <= 18)):
        part_of_day = 1
    elif ((hour >= 19) & (hour <= 23)):
        part_of_day = 2
    elif ((hour >= 0) & (hour <= 6)):
        part_of_day = 3
    return part_of_day

#----------------------------------------------------------------------------


def start_session_site(train_df, dataframe):
    '''
    Берем топ 3 сайта с которых Элис чаще всего начинает сессию.
    train_df  : для расчета параметров
    dataframe : считается по этому фрейму
    '''
    feature_np = np.nan_to_num(dataframe[sites].values)
    alice_start_session = train_df[train_df['target'] == 1][sites].values[:, 0]
    alice = Counter(alice_start_session)
    alice_most = [o[0] for o in alice.most_common(n=3)]

    sess_start_alice = np.isin(feature_np[:, 0], alice_most).astype('int')
    return sess_start_alice   


def calc_num_sites_session(matrix_ndarray: np.ndarray) -> np.ndarray:
    '''Расчитывает кол-во посещенных сайтов за сеанс'''
    assert isinstance(matrix_ndarray, np.ndarray), 'Input data should be numpy.ndarray.'
    return np.sum(np.square(np.isnan(matrix_ndarray).astype('int')-1), axis=1)


def calc_num_uniq_sites_session(matrix_ndarray: np.ndarray) -> np.ndarray:
    '''Расчитывает кол-во уникальных посещенных сайтов за сеанс'''
    assert isinstance(matrix_ndarray, np.ndarray), 'Input data should be numpy.ndarray.'
    return np.array([len(set(o[o > 0])) for o in matrix_ndarray])



def number_of_duplicate_sites_in_session(matrix_ndarray: np.ndarray) -> np.ndarray:
    '''Расчитываем количество сайтов которые дублируются в сессии. То есть кол-во
    не уникальных сайтов в сессии. Вторая бинарная переменная говорит есть ли дубли.
  
    Возвращает:
    ==========
    Список целочисленных значений: list with len(matrix)
  
  
    Например:
    =========
    кол-во сайтов 10 (может быть меньше из-за NaN), а кол-во уникальных 2, тогда:
    10 - 2 = 8. То есть это кол-во не уникальных сайтов в сессии.
    '''
    assert isinstance(matrix_ndarray, np.ndarray), 'Input data should be numpy.ndarray.'
  
    matrix_ndarray = np.nan_to_num(matrix_ndarray)
    n_sites_in_session = np.count_nonzero(matrix_ndarray, axis=1)
    n_uniq_sites_in_session = np.array([len(set(o[o > 0])) for o in matrix_ndarray])
    n_of_duplicates = n_sites_in_session-n_uniq_sites_in_session
    return n_of_duplicates


def is_duplicat_sites_in_session(array_of_ints) -> np.ndarray:
    '''Если в сессии есть будликаты (х > 0) тогда 1. Если нет тогда 0.'''
    assert isinstance(array_of_ints, (np.ndarray, list)), 'Input data should be an array.'
    return (array_of_ints > 0).astype('int')
  

def is_youtube(matrix_ndarray) -> bool:
    with open('../data/site_dic.pkl', 'rb') as handle:
        site_dict = pickle.load(handle)
    
    youtube_ix = []
    for key in list(site_dict.keys()):
        if 'youtube' in key:
            youtube_ix.append(site_dict[key])
    return (np.sum(np.isin(matrix_ndarray, youtube_ix), axis=1) > 0).astype('int')


def is_facebook(matrix_ndarray) -> bool:
    with open('../data/site_dic.pkl', 'rb') as handle:
        site_dict = pickle.load(handle)
    
    facebook_ix = []
    for key in list(site_dict.keys()):
        if 'facebook' in key:
            facebook_ix.append(site_dict[key])
    return np.sum(np.isin(matrix_ndarray, facebook_ix), axis=1).astype('int')
    
     
    


def duration_and_average_durat_in_session(full_dataframe) -> np.ndarray:
    '''В секундах. Возращает среднее(медиану) время которое пользователь 
    проводит на сайте.
  
  
    Входные данные:
    ===============
    dataframe: исходный датафрейм со всем колонками с NaN'ами.
  
    Возвращает:
    ===========
    Среднее время на каждом сайте в сессии.
    Сумарное время проведенное в сессии.
    '''
    # calculate delta time in seconds
    times = [f'time{o}' for o in range(1, 11)]
    delta_time = pd.DataFrame()
    for i in range(0, 9):
        delta_time = pd.concat([
         delta_time, 
         pd.DataFrame(full_dataframe[times[i+1]] - full_dataframe[times[i]], 
                     columns=[f'delta{i+1}-{i+2}']),
                            ], axis=1)
  
    for col in tqdm(delta_time.columns):
        delta_time[col] =   delta_time[col].apply(lambda x: x.total_seconds())
    return np.nan_to_num(np.nanmean(delta_time.values, 1)), np.nan_to_num(np.nansum(delta_time.values, 1))

def day_part(x):
    # 1-morning, 2-day, 3-evening, 4-night
    hour = x.hour
    if ((hour >= 7) & (hour <= 11)): return 1
    if ((hour >= 12) & (hour <= 18)): return 2
    if ((hour >= 19) & (hour <= 23)): return 3
    if ((hour >= 0) & (hour <= 6)): return 4
    
    
def time_fastai(dataframe)->pd.core.frame.DataFrame:
    new_df = pd.DataFrame()
    fld = dataframe['time1']
    attr = ['Month', 'Week', 'Day', 'Dayofweek',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    for n in attr: new_df[n] = getattr(fld.dt, n.lower())
    return new_df


def n_alice_sites(train_df, dataframe):
    '''Сайты на которые заходит только Элис. 
    
    Возвращает:
    ===========
    Кол-во сайтов на которые заходит только Элис.
    '''
    
    train_np = np.nan_to_num(train_df[sites].values)
    dataframe_np = np.nan_to_num(dataframe[sites].values)
    
    other_sites_np = np.nan_to_num(train_df[train_df['target'] == 0][sites].values)
    alice_sites_np = np.nan_to_num(train_df[train_df['target'] == 1][sites].values)
    
    other_uniq_sites = set(other_sites_np.reshape(-1))
    alice_uniq_sites = set(alice_sites_np.reshape(-1))
    
    intersection = other_uniq_sites.intersection(alice_uniq_sites)
    
    only_alice_sites = np.array(list(alice_uniq_sites - intersection))
    
    return np.isin(dataframe_np, only_alice_sites).astype('int').sum(axis=1)


def is_long_session(x):
    if x < 3:  return 0
    elif x < 5: return 1
    elif x < 10: return 2
    elif x < 30: return 3
    elif x < 40: return 4
    return 5


def is_youtube_start(matrix_ndarray) -> bool:
    with open('../data/site_dic.pkl', 'rb') as handle:
        site_dict = pickle.load(handle)
    
    youtube_ix = []
    for key in list(site_dict.keys()):
        if 'youtube' in key:
            youtube_ix.append(site_dict[key])
    return np.isin(matrix_ndarray[:, 0], youtube_ix).astype('int')


def is_facebook_start(matrix_ndarray) -> bool:
    with open('../data/site_dic.pkl', 'rb') as handle:
        site_dict = pickle.load(handle)
    
    facebook_ix = []
    for key in list(site_dict.keys()):
        if 'facebook' in key:
            facebook_ix.append(site_dict[key])
    return np.isin(matrix_ndarray[:, 0], facebook_ix).astype('int')

def is_mean_long(x):
    if x < 10:  return 0
    elif x < 15: return 1
    elif x < 25: return 2
    return 3


#--------------------------------------------------------------------

def prepe_add_features(dataframe, train_df):
    '''Обертка для расчета всех доп. признаков. '''

    # prepare data
    col_names = ['part_of_day', 'n_sites_in_session', 'n_uniq_sites_in_session', 'n_dupl_sites_in_session',
                    'is_youtube', 'is_facebook', 'mean_sites_duration', 'sess_duration', 'elice_start', 'n_sites_alice']
    sites = [f'site{o}' for o in range(1, 11)]
    times = [f'time{o}' for o in range(1, 11)]
    matrix_ndarray_sites = dataframe[sites].values

    # word with sites columsn
    part_of_day = dataframe['time1'].apply(lambda x: day_part(x))
    part_of_day = part_of_day.values
    n_sites_in_session = calc_num_sites_session(matrix_ndarray_sites)
    n_uniq_sites_in_session = calc_num_uniq_sites_session(matrix_ndarray_sites)
    n_dupl_sites_in_session = number_of_duplicate_sites_in_session(matrix_ndarray_sites)
    is_youtube_var = is_youtube(matrix_ndarray_sites)
    is_facebook_var = is_facebook(matrix_ndarray_sites)
    elice_start = start_session_site(train_df, dataframe)
    n_sites_alice = n_alice_sites(train_df, dataframe)
    
    # word with times columns
    mean_sites_duration, sess_duration = duration_and_average_durat_in_session(dataframe)
    
    # dataframes merging
    common_np = np.concatenate([
          part_of_day.reshape(-1, 1),
          n_sites_in_session.reshape(-1, 1),
          n_uniq_sites_in_session.reshape(-1, 1),
          n_dupl_sites_in_session.reshape(-1, 1),
          is_youtube_var.reshape(-1, 1),
          is_facebook_var.reshape(-1, 1),
          np.nan_to_num(mean_sites_duration.reshape(-1, 1)),
          sess_duration.reshape(-1, 1),
          elice_start.reshape(-1, 1),
          n_sites_alice.reshape(-1, 1),
      ], axis=1)
    fastai_features_df = time_fastai(dataframe)
    column_names = col_names + fastai_features_df.columns.tolist()
    merge = np.concatenate([common_np, fastai_features_df.values], axis=1)
    
    return pd.DataFrame(merge, columns=column_names, index=fastai_features_df.index.values)