from common import *


def load_data():
    # Читаем данные
    path_to_data = Path('../data')
    train_df = pd.read_csv(path_to_data/'train_music.csv')
    train_y = train_df['target']
    del train_df['target']
    test_df  = pd.read_csv(path_to_data/'test_music.csv')

    # Объединяем выборки для обработки переменных, запоминаем индексы чтобы потом разделить назад
    train_idx = train_df.index[-1]
    merged = pd.concat([train_df, test_df], axis=0)
    
    # Отдельно выделяем бинарные в тип bool
    bool_columns = ['tp_flag', 'block_flag', 'is_obl_center', 'is_my_vf']
    print(f'Из целочисленныъ - {len(bool_columns)} бинарных.')

    # Целочисленные переменные
    int_columns = [o for o in merged.columns for crit in ['flag', 'is', 'count'] if crit in o]
    int_columns += ['sim_count','device_type','manufacturer_category','os_category','tp_flag','days_exp', 'paym_last_days']
    _ = [int_columns.remove(o) for o in bool_columns]
    print(f'Целочисленных переменных  : {len(int_columns)}')

    # Переменные с плавающей точкой
    criterion_for_float_columns = ['data_type', 'rr', 'vol', 'cost', 'dur', 'sum', 'part', 'clc', 'lt', 'brnd']
    float_columns = [o for o in merged.columns for crit in criterion_for_float_columns if crit in o]
    print(f'Переменных с плавающей точкой : {len(float_columns)}')

    # Если в значениях переменной есть 0, тогда пропуски заполняет -1. Если нет 0, тогда заполняем нулем.
    merged[int_columns] = merged[int_columns].apply(lambda x: x.fillna(-1) if 0 in x.values else x.fillna(0))
    merged[int_columns] = merged[int_columns].apply(lambda x: x.astype('int'))
    merged[bool_columns] = merged[bool_columns].apply(lambda x: x.astype('bool'))
    merged[float_columns] = merged[float_columns].apply(lambda x: x.fillna(x.median()) if 0 in x.values else x.fillna(0))
    merged[float_columns] = merged[float_columns].apply(lambda x: x.astype('float'))
    print(f'Переменным присвоен соответствующий тип.')

    # Разделяем обработанные обучающую и тестовую выборки
    train = merged.iloc[:train_idx+1, :]
    train['target'] = train_y
    test = merged.iloc[train_idx+1:, :]
    del merged
    del train_y
    
    return train, test, {
        'int_cols'  : int_columns,
        'bool_cols' : bool_columns,
        'float_cols': float_columns,
        }