from utils import create_data_loaders
from utils import train_model, save_model, lr_scheduler
from utils import build_results_paths, model_import
import models
from models import meta_model
import torch.nn as nn
import torch.optim as optim

####################### PARAMETERS #######################
label = 'BinaryPositiveMean' # 'EuristicLabels' # 
window_size = 501
batch_size = 64
learning_rate = 1e-4 
epochs = 10
model_list = ['deep_CNN_simple', 'CNN_transformer_simple']
training_list = ['training_1', 'training_1']
patience = 5
##########################################################

# import models
models = []
for model_name, training_name in zip(model_list, training_list):
    model_path, _, _ = build_results_paths(label, model_name, training_name)
    model = model_import(model_path, summary=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    models.append(model)

model = meta_model(models[0], models[1])
model.summary()
# data loaders
train_loader, val_loader = create_data_loaders(label, window_size, batch_size)
# loss, optimizer, scheduler
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# train and save
model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device=0, num_epochs=epochs, patience=patience)
model_path_dict = {
    'label':label, 
    'model_name':'meta_model'
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
    
