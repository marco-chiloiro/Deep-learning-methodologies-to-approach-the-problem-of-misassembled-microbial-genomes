from utils import create_data_loaders
from utils import train_model, save_model, lr_scheduler
import models
import torch.nn as nn
import torch.optim as optim

####################### PARAMETERS #######################
label = 'BinaryPositiveMean' # 'EuristicLabels'
model_name = 'deep_CNN_simple' # 'CNN_transformer_simple' # 
window_size = 501
batch_size = 64
learning_rate = 1e-4 
epochs = 100
patience = 30
weight_decay = 1e-4
##########################################################

# import model
init_model = getattr(models, model_name)
model = init_model()
model.summary()
# data loaders
train_loader, val_loader = create_data_loaders(label, window_size, batch_size)
# loss, optimizer, scheduler
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler(epochs, len(train_loader), optimizer)
# train and save
model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device=0, num_epochs=epochs, patience=patience, scheduler=scheduler)
model_path_dict = {
    'label':label, 
    'model_name':model_name
}
model_info = {
    'total_samples': batch_size*(len(train_loader)+len(val_loader)),
    'batch_size':batch_size,
    'window_size':window_size,
    'epochs':epochs,
    'lr':learning_rate,
    'patience':patience
}
save_model(model, history, model_path_dict, model_info)
