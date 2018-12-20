from common import *
import torch


class GaussRankScaler():
	def __init__( self ):
		self.epsilon = 0.001
		self.lower = -1 + self.epsilon
		self.upper =  1 - self.epsilon
		self.range = self.upper - self.lower

	def fit_transform( self, X ):
	
		i = np.argsort( X, axis = 0 )
		j = np.argsort( i, axis = 0 )

		assert ( j.min() == 0 ).all()
		assert ( j.max() == len( j ) - 1 ).all()
		
		j_range = len( j ) - 1
		self.divider = j_range / self.range
		
		transformed = j / self.divider
		transformed = transformed - self.upper
		transformed = erfinv( transformed )
		
		return transformed


#-----------------------------------------------------------------
def add_feature_to_sparse(spars_matrxi, feature_array_or_matrix):
    return hstack([spars_matrxi, feature_array_or_matrix])

def check_lr_crossval(sparse_matrix, y, n_splits=2, balanced=True):
    time_split = TimeSeriesSplit(n_splits=n_splits)
    if balanced:
        logit = LogisticRegression(C=1, random_state=17, class_weight='balanced')
    else:
        logit = LogisticRegression(C=1, random_state=17)
    
    cv_scores = cross_val_score(logit, sparse_matrix, y, cv=time_split, scoring='roc_auc', n_jobs=-1) 
    print(cv_scores, cv_scores.mean(), cv_scores.std())
    return cv_scores, cv_scores.mean(), cv_scores.std()


def process_sites():
    train_sites = train_df[sites].fillna(0).astype(np.int32)
    test_sites = test_df[sites].fillna(0).astype(np.int32)
    return train_sites, test_sites


#-------------------------------------------------------------------
def load_dataframes():
    train_df = pd.read_csv('../data/train_sessions.csv', index_col='session_id')
    test_df = pd.read_csv('../data/test_sessions.csv', index_col='session_id')

    # Convert time1, ..., time10 columns to datetime type
    times = ['time%s' % i for i in range(1, 11)]
    train_df[times] = train_df[times].apply(pd.to_datetime)
    test_df[times] = test_df[times].apply(pd.to_datetime)

    # Sort the data by time
    train_df = train_df.sort_values(by='time1')

    # Look at the first rows of the training set
    return train_df, test_df


#-------------------------------------------------------------------
# A helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
    
    
#---------------------------------------------------------------------



def make_submition(test_df, model, file_name):
    preds = model.get_preds(ds_type=DatasetType.Test)
    p = preds[0][:, 1].tolist()
    submit_columns = ['id', 'prediction']
    df = pd.DataFrame(list(zip(test_df.id.values, p)), columns=['id', 'prediction'])
    df.set_index('id', inplace=True)
    # fillna with medians
    df.fillna(df.prediction.median()).to_csv(f'submissions/{file_name}.csv')
    print('File saved.') 
    

# Прикручиваем ROC-AUC в качестве метрики
class RocAuc(Callback):
    def __init__(self, func):
        self.func, self.name = func, func.__name__

    def on_epoch_begin(self, **kwargs):
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        self.count += last_target.size(0)
        self.val += last_target.size(0) * self.func(last_target, last_output[:, 1])

    def on_epoch_end(self, **kwargs):
        self.metric = self.val/self.count
        
 
def create_df_for_files(path):
    '''
    Read images for all subfolders in path folder.
    
    Inputs:
    - path: folder which contain images in subfolders,
    name of classes will be taken from subfolder name;
    
    Outputs:
    - dataframe: contain three columns {file - path to read a image,
    category_id, category}.
    '''
    CATEGORIES = os.listdir(path)
    CATEGORIES = [i for i in CATEGORIES if not i.startswith('.')]

    dataframe = []
    for category_id, category in enumerate(CATEGORIES):
        for file in os.listdir(os.path.join(path, category)):
            dataframe.append(['{}/{}/{}/{}'.format(os.getcwd(), path, category, file), 
                          category_id, category])
    dataframe = pd.DataFrame(dataframe, columns=['file', 'category_id', 'category'])
    return dataframe
    
    
def prepare_dataframe(PATH):
    data = create_df_for_files(PATH)
    data['file'] = data['file'].apply(lambda x: x.replace('../', ''))
    data['file'] = data['file'].apply(lambda x: x.replace('notebooks/', '') )
    data['source'] = data['file'].apply(lambda x: re.split(r'[/, _]', x)[-3])
    return data

# Example of all these classes is^
# link-to-git-hub
#-------------------------------------------------------------
class MyScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr, mom in zip(self.optimizer.param_groups, self.get_lr(), self.get_moms()):
            param_group['lr'] = lr
            param_group['betas'] = (mom, 0.99)
            
            
class OneCycle(MyScheduler):
    def __init__(self, epochs, optimizer, div_factor, pct_start, dl_len, last_epoch=-1):
        lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.max_lr = lrs[0]
        self.eta_min = self.max_lr / div_factor
        self.num_iterations = dl_len * epochs
        self.upward_steps = int(self.num_iterations * pct_start)
        self.max_moms=0.95
        self.min_moms=0.85
        super(OneCycle, self).__init__(optimizer, last_epoch)
        
    def calculate_schedule(self, lr):
        '''Here we are calculating curve.'''
        upward_lr = np.linspace(start=self.eta_min, stop=lr, num=self.upward_steps)
        downward_lr = [(self.eta_min + (lr - self.eta_min) * (1 + math.cos((math.pi*o)/self.num_iterations)) / 2) 
                for o in np.linspace(start=0, stop=self.num_iterations, num=self.num_iterations-self.upward_steps)]
        
        upward_moms = np.linspace(start=self.max_moms, stop=self.min_moms, num=self.upward_steps)
        downward_moms = [(self.min_moms + (self.max_moms - self.min_moms) * (1 + math.cos((math.pi*o)/self.num_iterations)) / 2) 
                for o in np.linspace(start=self.num_iterations, stop=0, num=self.num_iterations-self.upward_steps)]
        return [np.concatenate([upward_lr, downward_lr]), np.concatenate([upward_moms, downward_moms])]
        
    def get_lr(self):
        lr = [self.calculate_schedule(base_lr)[0][self.last_epoch] for base_lr in self.base_lrs]
        return lr
    
    def get_moms(self):
        moms = [self.calculate_schedule(base_lr)[1][self.last_epoch] for base_lr in self.base_lrs]
        return moms
    
    
#----------------------------------------------
#--------------------- FIND LR ----------------
#----------------------------------------------
# TODO: Add progress bar; do the same with valid loss?
def find_lr(net, dataloader, optimizer, criterion, init_value = 1e-8, final_value=10., beta = 0.98):
    '''
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html#how-do-you-find-a-good-learning-rate.
    '''
    num = len(dataloader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in tqdm_notebook(dataloader):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        _, _, _, outputs = net(inputs) # many outputs from deeply supervized learning
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


def plot_lr_find(log_lrs, losses):
    _, ax = plt.subplots(1,1)
    ax.plot(np.exp(log_lrs)[10:-5],losses[10:-5]);
    ax.set_xscale('log');
    ax.set_xlabel('learning rate');
    ax.set_ylabel('loss');
    
    
    
def trainer(model, optimizer, scheduler, loss_function, epochs, training_data, validation_data):
    '''
    Description.
    '''
    history = {
        'train' : {
            'lr'     : [],
            'betas'  : [],
            'loss'   : [],
            'acc'    : [],
            },
        'valid' : {
            'loss' : [],
            'lr'   : [],
            'acc'  : [],
            },
    }
    for epoch in range(epochs):
        training_loss, validation_loss  = 0.0, 0.0
        # loop over training and validation for each epoch
        for dataset, training in [(training_data, True),
                               (validation_data, False)]:
            correct = total = 0
            torch.set_grad_enabled(training)
            model.train(training)
            t = tqdm.tqdm_notebook(dataset)
            # loop over dataset
            for batch_idx, (images, labels) in enumerate(t):               
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                scores = model(images)
                loss = loss_function(scores, labels)
                # calculate metrics
                predictions = torch.argmax(scores, dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels) 
                accuracy = round(correct / total, 3)
                # do all stuff for train and validation
                if training:
                    loss.backward()
                    training_loss = loss.item()
                    t.set_postfix(epoch=epoch, training_loss=training_loss,
                            accuracy=accuracy, refresh=False)
                    history['train']['lr'].append(optimizer.param_groups[0]['lr'])
                    history['train']['betas'].append(optimizer.param_groups[0]['betas'])
                    history['train']['loss'].append(training_loss)
                    history['train']['acc'].append(accuracy)
                    optimizer.step()
                    scheduler.step()
                else:
                    validation_loss = loss.item()       
                    t.set_postfix(epoch=epoch, validation_loss=validation_loss,
                            accuracy=accuracy, refresh=False)
                    history['valid']['loss'].append(validation_loss)
                    history['valid']['acc'].append(accuracy)
    return history

#-------------------------------------------------------------
def show_lr_and_moms(history):
    '''Отображение шага обучения и энерции.
    Для того чтобы посмотреть схему обучения.
    Как изменяются шаг обучения и энерция.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train']['lr']);
    axes[0].set_title('Learning rate changes over training');
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Learning rate')
    
    axes[1].plot(history['train']['betas']);
    axes[1].set_title('Momentum changes over training');
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Momentum')


def moving_average(sequence, alpha=0.999):
    '''Берет список и осредняет.'''
    avg_loss = sequence[0]
    average = []
    for n, o in enumerate(sequence):
        avg_loss = (alpha*avg_loss) + ((1-alpha)*o)
        average.append(avg_loss)
    return average


def show_train_results(history : dict, step : int=7):
    '''Визуализация результатов обучения.
    
    Аргументы:
    ==========
    history : словать словарей  с результатами по обучающей
              выборке и валидационной, лосс, акураси и шаг
              обучения
    factor  : из-за того что длинна обучающей и валидационной
              выборки разная, обучающую выборку отображаем не 
              полностью а с определенным шагом. То есть каждый
              step значение. Валидационная отобр. полностью.    
    '''
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(history['train']['loss'][::step], label='train')
    axes[0].plot(history['valid']['loss'], label='valid')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train']['acc'][::step], label='train')
    axes[1].plot(history['valid']['acc'], label='valid')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()