
class DefaultConfig(object):
    train_dataset_path = r'./datasets/context-epitope-train-128.pkl'
    val_dataset_path = r'./datasets/context-epitope-val-128.pkl'
    test_dataset_path = r'./datasets/context-epitope-test-128.pkl'
    save_path = r'./models_saved/'

    epochs = 150
    feature_dim = 128
    learning_rate = 0.001
    weight_decay = 5e-4
    dropout_rate = 0.5
    batch_size = 32
    neg_wt = 0.1

    # NodeAverage
    hidden_dim = [256, 512]

    # BiLSTM
    num_hidden = 32
    num_layer = 1

    # mlp
    mlp_dim = 512

