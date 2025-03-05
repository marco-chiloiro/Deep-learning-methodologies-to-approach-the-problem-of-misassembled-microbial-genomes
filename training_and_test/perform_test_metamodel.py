from utils import model_import, save_training_history, build_results_paths, test, metamodel_import

#### PARAMETERS ####
label_type = 'EuristicLabels'
model_list = ['deep_CNN_simple', 'CNN_transformer_simple']
training_list = ['training_1', 'training_1']
training_name = 'training_1'
####################

metamodel_path, history_path, info_path = build_results_paths(label_type, 'meta_model', training_name)
save_training_history(history_path, early_stop=5)

# import meta-model
models = []
for model_name, training_name in zip(model_list, training_list):
    model_path, _, _ = build_results_paths(label_type, model_name, training_name)
    model = model_import(model_path, summary=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    models.append(model)
model = metamodel_import(metamodel_path, models[0], models[1])
model.summary()

for i in range(1,4):
    print('Testing level', i)
    test(model, label_type, test_level=i, info_path=info_path)