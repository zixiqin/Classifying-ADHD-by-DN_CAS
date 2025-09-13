import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from torch.optim.lr_scheduler import ReduceLROnPlateau

import random

from sklearn.model_selection import KFold

# set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# Define the custom MLP class
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates):
        super(CustomMLP, self).__init__()
        
        layers = []
        previous_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
            previous_size = hidden_size
        
        layers.append(nn.Linear(previous_size, output_size))
        
        self.network = nn.Sequential(*layers)

        self._initialize_weights()  # initialize the weights of the model

    
    def forward(self, x):
        return self.network(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Using Kaiming Normal initialization for weights
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Define the custom MLP class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



def train_binary_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience=5, device=None):
    device = device or next(model.parameters()).device
    model.to(device)

    train_losses, val_losses, val_accs = [], [], []
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        # ------- train -------
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device).view(-1)

            optimizer.zero_grad()
            logits = model(inputs).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # ------- validate -------
        model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device).view(-1)

                logits = model(inputs).view(-1)
                loss = criterion(logits, labels)
                running_loss += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).int()
                correct += (preds == labels.int()).sum().item()
                total += labels.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        val_acc = correct / total
        val_accs.append(val_acc)

        # scheduler 
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # early stopping based on val_loss
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return val_losses[-1], train_losses, val_losses, val_accs



def cv_model_adding_layer_binary(
    init_model,
    weight_decay,
    X, y, seed,
    epochs=2000,
    patience=5,
    lr=1e-3,
    full_batch=True,        
    use_pca=False,
    pca_keep=0.99,
    device=None
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    y = np.asarray(y).astype(np.float32).reshape(-1)
    X = np.asarray(X, dtype=np.float32)
    input_size = X.shape[1]

    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    best_val_losses, folds_train_losses, folds_val_losses, folds_val_accs = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # standardize for each fold
        scaler_X = StandardScaler()
        X_tr = scaler_X.fit_transform(X_tr)
        X_va = scaler_X.transform(X_va)


        # using pca or not
        if use_pca:
            if isinstance(pca_keep, float):
                pca = PCA(n_components=pca_keep, svd_solver="full")
            else:
                pca = PCA(n_components=min(int(pca_keep), input_size))
            X_tr = pca.fit_transform(X_tr)
            X_va = pca.transform(X_va)


        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)


        if full_batch:
            train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                                      batch_size=len(X_tr_t), shuffle=False, drop_last=False)
            val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t),
                                      batch_size=len(X_va_t), shuffle=False, drop_last=False)
        else:
            train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                                      batch_size=64, shuffle=True, drop_last=False)
            val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t),
                                      batch_size=64, shuffle=False, drop_last=False)


        model = init_model
        model.to(device)



        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)


        # training for binary classification
        last_val_loss, tr_losses, va_losses, va_accs = train_binary_classifier(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs=epochs, patience=patience, device=device
        )

        fold_best = float(np.nanmin(va_losses)) if len(va_losses) else float('inf')
        best_val_losses.append(fold_best)
        folds_train_losses.append(tr_losses)
        folds_val_losses.append(va_losses)
        folds_val_accs.append(va_accs)

        #print(f"[Fold {fold}] best_val_loss={fold_best:.6f}, last_val_loss={last_val_loss:.6f}, last_val_acc={va_accs[-1]:.4f}")

    #print("best_val_losses (per fold): ", best_val_losses)
    return float(np.mean(best_val_losses))


def single_model_run_binary(
    X, y, min_loss_info, previous_best_model, previous_model_info, seed,
    lr=1e-3, patience=5, num_epochs=2000, use_pca=True, device=None
):
    """
    X: (N, D) numpy array / pandas.values
    y: (N,) 0/1 labels
    previous_model_info: for example {'layers': [h1, h2, ...], 'drops': [d1, d2, ...]}
                          CustomMLP(input_size, hidden_layers, output_size, drops)
    Attention: output size should be 1 for binary classification (single logit)
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # ---- split----
    X_train, X_val, Y_train, Y_val = train_test_split(
        np.asarray(X, dtype=np.float32), np.asarray(y).reshape(-1).astype(np.float32),
        test_size=0.2, random_state=seed, stratify=np.asarray(y).reshape(-1)
    )

    # ---- standardize ----
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)

    # ---- optional PCA----
    if use_pca:
        pca = PCA(n_components=X_train_scaled.shape[1])  
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca   = pca.transform(X_val_scaled)
    else:
        X_train_pca, X_val_pca = X_train_scaled, X_val_scaled

    # ---- y do not standardize! keep 0/1 float ----
    y_train_binary = Y_train.astype(np.float32)
    y_val_binary   = Y_val.astype(np.float32)

    # ---- Tensor & DataLoader ----
    X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_binary, dtype=torch.float32)  # [N]
    X_val_tensor   = torch.tensor(X_val_pca,   dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val_binary, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=len(X_train_tensor), shuffle=False)
    val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                              batch_size=len(X_val_tensor), shuffle=False)

    # ---- build current model: hidden layer structure follows previous_model_info, output=1 ----
    input_size = X_train_pca.shape[1]
    output_size = 1
    current_weight_decay_value = min_loss_info['weight_decay_value']
    current_single_model = CustomMLP(input_size, previous_model_info['layers'], output_size, previous_model_info['drops']).to(device)

    # ---- Copy the alignable prefix layer weights (linear layers only) from the previous model (with one fewer layer). ----
    if previous_best_model is not None:
        with torch.no_grad():
            # Assume that `.network` has the form `[Linear, Act, Drop, Linear, Act, Drop, ..., Linear_out]`.
            # Only copy the prefix Linear layers whose input/output shapes match on both sides; the final new output layer is not copied.
            prev_linears = [m for m in previous_best_model.network if isinstance(m, nn.Linear)]
            curr_linears = [m for m in current_single_model.network if isinstance(m, nn.Linear)]

            copy_L = min(len(prev_linears), len(curr_linears) - 1)  # Reserve the last layer (the output layer) and do not copy it.
            for i in range(copy_L):
                if prev_linears[i].weight.shape == curr_linears[i].weight.shape:
                    curr_linears[i].weight.copy_(prev_linears[i].weight)
                if prev_linears[i].bias is not None and curr_linears[i].bias is not None and \
                   prev_linears[i].bias.shape == curr_linears[i].bias.shape:
                    curr_linears[i].bias.copy_(prev_linears[i].bias)


    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(current_single_model.parameters(), lr=lr, weight_decay=current_weight_decay_value)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-7)

    best_model = None
    best_loss = float('inf')
    early_stop_counter = 0
    val_acc_history = []

    for epoch in range(num_epochs):
        current_single_model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device).view(-1)  # [N]
            optimizer.zero_grad()
            logits = current_single_model(xb).view(-1)       #  logit
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        current_single_model.eval()
        val_loss_sum, n_val = 0.0, 0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device).view(-1)
                logits = current_single_model(xb).view(-1)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                n_val += xb.size(0)
                # 计算 accuracy
                preds = (torch.sigmoid(logits) >= 0.5).int()
                correct += (preds == yb.int()).sum().item()

        val_loss = val_loss_sum / max(n_val, 1)
        val_acc = correct / max(n_val, 1)
        val_acc_history.append(val_acc)

        scheduler_plateau.step(val_loss)
        # print(f'Epoch {epoch+1}/{num_epochs}  val_loss={val_loss:.6f}  val_acc={val_acc:.4f}')

        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            best_model = deepcopy(current_single_model).to("cpu")  
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            # print('Early stopping triggered')
            break

    return best_model, val_acc_history[-1]



data = pd.read_csv(r'data.csv')
col = [
    "Number Matching",
    "Planned Codes",
    "Planned Connections",
    "Nonverbal Matrices",
    "Verbal Spatial Relations",
    "Figure Memory",
    "Expressive Attention",
    "Number Detection",
    "Receptive Attention",
    "Word Series",
    "Sentence Repetition",
    "Speech Rate / Sentence Questions",
    "class"
]
data = data[col]
data["class"] = data["class"].map({"ADHD": 1, "TD": 0}).astype(int)

y = data['class']
X = data.drop(columns=['class'])


# define the input size and output size
# define the input data and target data
input_size = len(X.columns)
output_size = 1
X_all = data.loc[:,X.columns]
y_all = y


X_all = X_all.values
y_all = y_all.values

seed_ls = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
seed_result = {}

for seed in seed_ls:
    set_seed(seed)  
    print("===================================")
    print("Random Seed:", seed)
    print("===================================")
    # split the data into training and testing sets
    global_X_train, global_X_test, global_Y_train, global_Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state = seed)


    X = global_X_train
    y = global_Y_train
    
    # Define the hyperparameters to be tuned

    # the searching range of neurons in the hidden layer
    opt_hidden_sizes = [4, 8, 16,32] 

    #### The model would either have a dropout layer or weight decay, but not both！！  ####
    # the searching range of dropout rates
    opt_dropout_rates = [0.3,0.1,0] 
    # the searching range of weight decay values

    opt_weight_decay = [0.01,0.001, 0.0001]




    # the maximun number of layers
    Target_layer = 8

    # save the model paramter of the previous model with 1 layer less, which will be used as the initial value of the model with 1 more layer
    previous_all_layer_info = None

    # save the hyperparameters of all the model that has been assessed
    # the best_val_loss_single would be used to definie which exact model would be used in the final model and do the testing with the global test set
    previous_model_info = {'layers':[], 'drops':[],'best_val_loss_single':[]}

        
    # let's start ！！
    for layer in range(1,Target_layer+1):
        print("###########################")
        print("Layer:", layer)
        print("###########################")
        # start from the model with only 1 layer
        if layer == 1:
            # model_performance_dict only save the information of the model with 1 layer
            model_performance_dict = {'hidden_layer': [], 'dropout_rate':[], 'weight_decay_value':[],'avg_val_loss':[]}
            for hiden_layer in opt_hidden_sizes:
                # search the best weight decay value
                # drop = 0
                for wd in opt_weight_decay:
                    print("--------------------------------")
                    print("hiden_layer:", hiden_layer)
                    print("wd:", wd)
                    print("--------------------------------")
                    drop = 0
                    new_model = CustomMLP(input_size, [int(hiden_layer)], output_size, [drop])

                    # using the cv to find the hyperparameters with model with one hidden layer
                    model_performance = cv_model_adding_layer_binary(new_model, wd, X, y, seed)

                    # save the hyperparameters and avg validation loss of model with different combination of hyperparameters
                    model_performance_dict['hidden_layer'].append(hiden_layer)
                    model_performance_dict['dropout_rate'].append(drop)
                    model_performance_dict['weight_decay_value'].append(wd)
                    model_performance_dict['avg_val_loss'].append(model_performance)

            for hiden_layer in opt_hidden_sizes:
                for drop in opt_dropout_rates:
                    # search the best weight decay value
                    # wd = 0
                    print("--------------------------------")
                    print("hiden_layer:", hiden_layer)
                    print("drop:", drop)
                    print("--------------------------------")
                    wd = 0
                    new_model = CustomMLP(input_size, [int(hiden_layer)], output_size, [drop])
                    # using the cv to find the hyperparameters with model with one hidden layer
                    model_performance = cv_model_adding_layer_binary(new_model, wd, X, y, seed)

                    # save the hyperparameters and avg validation loss of model with different combination of hyperparameters
                    model_performance_dict['hidden_layer'].append(hiden_layer)
                    model_performance_dict['dropout_rate'].append(drop)
                    model_performance_dict['weight_decay_value'].append(wd)
                    model_performance_dict['avg_val_loss'].append(model_performance)

            # find the hyperparameters with the minimum avg validation loss from the dictionart: model_performance_dict
            layer_result_df = pd.DataFrame(model_performance_dict)
            layer_hl_min_loss_info = layer_result_df.loc[layer_result_df['avg_val_loss'].idxmin()]    
            previous_model_info['layers'].append(int(layer_hl_min_loss_info['hidden_layer']))
            previous_model_info['drops'].append(layer_hl_min_loss_info['dropout_rate'])

            # based on the selected best hyperparameters, train the model with the global training set and save the model parameters
            layer_best_model, layer_best_loss = single_model_run_binary(X, y,layer_hl_min_loss_info, previous_all_layer_info, previous_model_info, seed)

            # save the best val loss of the model with 1 layer
            previous_model_info['best_val_loss_single'].append(layer_best_loss)

            # save the best model parameters of the model with 1 layer
            # which will be used as the initial value of the model with 2 layers
            filename = f'{seed}_{layer}hl_model.pth'
            torch.save(layer_best_model.state_dict(), filename)
            
            previous_all_layer_info = layer_best_model

        else:

            # model with more than 1 layer

            # model_performance_dict only save the information of the model with current layer
            model_performance_dict = {'hidden_layer': [], 'dropout_rate':[], 'weight_decay_value':[],'avg_val_loss':[]}
            for hiden_layer in opt_hidden_sizes:
                for wd in opt_weight_decay:
                    # search the best wd value
                    # drop = 0
                    print("--------------------------------")
                    print("hiden_layer:", hiden_layer)
                    print("wd:", wd)
                    print("--------------------------------")
                    drop = 0
                    new_model = CustomMLP(input_size, previous_model_info['layers']+[hiden_layer], output_size, previous_model_info['drops']+[drop])

                    # load the best model parameters from the previous model with 1 layer less
                    with torch.no_grad():
                        for i in range(len(previous_model_info['layers'])-1):
                            if isinstance(previous_all_layer_info.network[i*3], nn.Linear):
                                new_model.network[i*3].weight.copy_(previous_all_layer_info.network[i*3].weight)
                                new_model.network[i*3].bias.copy_(previous_all_layer_info.network[i*3].bias)
                    
                    # using the cv to find the hyperparameters with model with current layer
                    model_performance = cv_model_adding_layer_binary(new_model, wd, X, y, seed)

                    # save the hyperparameters and avg validation loss of model with different combination of hyperparameters
                    model_performance_dict['hidden_layer'].append(hiden_layer)
                    model_performance_dict['dropout_rate'].append(drop)
                    model_performance_dict['weight_decay_value'].append(wd)
                    model_performance_dict['avg_val_loss'].append(model_performance)

            for hiden_layer in opt_hidden_sizes:
                for drop in opt_dropout_rates:
                    # search the best drop rate
                    # wd = 0
                    print("--------------------------------")
                    print("hiden_layer:", hiden_layer)
                    print("drop:", drop)
                    print("--------------------------------")
                    wd = 0
                    new_model = CustomMLP(input_size, previous_model_info['layers']+[hiden_layer], output_size, previous_model_info['drops']+[drop])

                    # load the best model parameters from the previous model with 1 layer less
                    with torch.no_grad():
                        for i in range(len(previous_model_info['layers'])-1):
                            if isinstance(previous_all_layer_info.network[i*3], nn.Linear):
                                new_model.network[i*3].weight.copy_(previous_all_layer_info.network[i*3].weight)
                                new_model.network[i*3].bias.copy_(previous_all_layer_info.network[i*3].bias)

                    # using the cv to find the hyperparameters with model with current layer
                    model_performance = cv_model_adding_layer_binary(new_model, wd, X, y, seed)

                    # save the hyperparameters and avg validation loss of model with different combination of hyperparameters
                    model_performance_dict['hidden_layer'].append(hiden_layer)
                    model_performance_dict['dropout_rate'].append(drop)
                    model_performance_dict['weight_decay_value'].append(wd)
                    model_performance_dict['avg_val_loss'].append(model_performance)

            # find the hyperparameters with the minimum avg validation loss from the dictionart: model_performance_dict
            layer_result_df = pd.DataFrame(model_performance_dict)
            layer_hl_min_loss_info = layer_result_df.loc[layer_result_df['avg_val_loss'].idxmin()]    
            previous_model_info['layers'].append(int(layer_hl_min_loss_info['hidden_layer']))
            previous_model_info['drops'].append(layer_hl_min_loss_info['dropout_rate'])

            # based on the selected best hyperparameters, train the model with the global training set and save the model parameters
            layer_best_model, layer_best_loss = single_model_run_binary(X, y,layer_hl_min_loss_info, previous_all_layer_info, previous_model_info, seed)
            previous_model_info['best_val_loss_single'].append(layer_best_loss)

            # save the best model parameters of the model with current layer
            filename = f'{seed}_{layer}hl_model.pth'
            torch.save(layer_best_model.state_dict(), filename)
            
            previous_all_layer_info = layer_best_model

    
    # currently we have the best hyperparameters of the model with 1 to k (slef-defined Target_layer) layers
    # we need to figure out which layer number is the best based on there best validation loss
    print(previous_model_info)
    max_index = previous_model_info['best_val_loss_single'].index(max(previous_model_info['best_val_loss_single']))
    print("max_index:", max_index)
    num_hidden_layers = len(previous_model_info['layers'][:max_index+1])
    print("The best number of hidden layers is:", num_hidden_layers)
    best_model_path = f'{seed}_{num_hidden_layers}hl_model.pth'
    print("For seed", seed, ", The best model has ", num_hidden_layers, " hidden layers")



    # Now we have the best model structure and the model parameters
    # Let's do the testing with the global test set
    # 1) Load the global training and testing sets (X, y)
    y_tr = global_Y_train.astype(np.float32).reshape(-1)
    y_te = global_Y_test.astype(np.float32).reshape(-1)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(global_X_train)
    X_test_scaled  = scaler_X.transform(global_X_test)


    use_pca = True
    if use_pca:
        pca = PCA(n_components=X_train_scaled.shape[1])
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca  = pca.transform(X_test_scaled)
    else:
        X_train_pca, X_test_pca = X_train_scaled, X_test_scaled


    X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_tr, dtype=torch.float32)  # 0/1
    X_test_tensor  = torch.tensor(X_test_pca,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_te, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader   = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)


    input_size = X_train_pca.shape[1]
    output_size = 1  
    test_model = CustomMLP(
        input_size,
        previous_model_info['layers'][:max_index+1],
        output_size,
        previous_model_info['drops'][:max_index+1]
    )
    test_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    test_model.eval()


    with torch.no_grad():
        for xb, _ in val_loader:  
            logits = test_model(xb).view(-1)              # [N]
            probs  = torch.sigmoid(logits).cpu().numpy()  # [N]
            preds  = (probs >= 0.5).astype(int)           # [N]


    # calculate metrics
    y_true = y_te.astype(int)
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    sens = recall_score(y_true, preds, zero_division=0)        # sensitivity = recall = TP / (TP+FN)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1   = f1_score(y_true, preds, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        'best_num_hidden_layers': num_hidden_layers
    }
    print(metrics)
    seed_result[seed] = metrics




seed_result_df = pd.DataFrame(seed_result).T
seed_result_df.to_csv("result.csv")