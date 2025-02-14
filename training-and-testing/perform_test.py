from utils import model_import, save_training_history, build_results_paths, test

#### PARAMETERS ####
label_type = 'EuristicLabels'
model_name = 'CNN_transformer_simple'
training_name = 'training_1'
####################

model_path, history_path, info_path = build_results_paths(label_type, model_name, training_name)
save_training_history(history_path, early_stop=30)
model = model_import(model_path)
for i in range(1,4):
    print('Testing level', i)
    test(model, label_type, test_level=i, info_path=info_path)