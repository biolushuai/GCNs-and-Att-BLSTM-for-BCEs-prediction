import os
import csv
import pickle
import torch
from models import *
from metrics import *
from utils import *
from config import DefaultConfig
configs = DefaultConfig()


def test(model, test_graphs):
    model.eval()
    all_trues = []
    all_preds = []

    for test_g in test_graphs:
        if torch.cuda.is_available():
            test_ag_vertex = torch.FloatTensor(test_g['ag_feat']).cuda()
            test_ag_edge = torch.FloatTensor(test_g['ag_edge_feat']).cuda()
            test_ag_nh_indices = torch.LongTensor(test_g['ag_nh_indices']).cuda()
            test_ag_label = torch.LongTensor(test_g['ag_label'])
            test_ag_indices = [index for index, _ in enumerate(test_g['ag_label'])]
            test_ag_indices = torch.LongTensor(test_ag_indices).cuda()

            # test_ag_vertex = torch.FloatTensor(test_g['ag_feat']).cuda()
            # test_ag_edge = torch.FloatTensor(test_g['ag_edge_feat']).cuda()
            # test_ag_nh_indices = torch.LongTensor(test_g['ag_nh_indices']).cuda()
            # test_ag_label = torch.LongTensor(test_g['ag_label'])
            # test_ag_indices = [[index, label] for index, label in enumerate(test_g['ag_label'])]
            # np.random.shuffle(down_sample(np.array(test_ag_indices)))
            # test_ag_indices = [index for index, _ in enumerate(test_g['ag_label'])]
            # test_ag_indices = torch.LongTensor(test_ag_indices).cuda()

        # g_preds = model(test_ag_vertex, test_ag_indices)
        # g_preds = model(test_ag_vertex, test_ag_nh_indices, test_ag_indices)
        g_preds = model(test_ag_vertex, test_ag_edge, test_ag_nh_indices, test_ag_indices)
        g_preds = g_preds.data.cpu().numpy()
        test_ag_label = test_ag_label.numpy()
        all_preds.append(g_preds)
        all_trues.append(test_ag_label)

        g_auc_pr = compute_auc_pr(test_ag_label, g_preds)
        g_auc_roc = compute_auc_roc(test_ag_label, g_preds)
        print(test_g['PDBID'], g_auc_pr, g_auc_roc)

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    auc_roc = compute_auc_roc(all_trues, all_preds)
    auc_pr = compute_auc_pr(all_trues, all_preds)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_trues, all_preds)
    acc = compute_acc(all_trues, predictions_max)
    mcc = compute_mcc(all_trues, predictions_max)
    print(acc, auc_roc, auc_pr, mcc, f_max, p_max, r_max, t_max)
    return [acc, auc_roc, auc_pr, mcc, f_max, p_max, r_max, t_max]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # test_model = NodeAverageModel()
    # test_model = NodeEdgeAverageModel()
    # test_model = BiLSTMNodeAverageModel()
    test_model = BiLSTMNodeEdgeAverageModel()

    if torch.cuda.is_available():
        model = test_model.cuda()

    test_path = configs.test_dataset_path

    with open(test_path, 'rb') as f:
        test_list, test_data = pickle.load(f)

    from utils import propress_data
    propress_data(test_data)

    with open(os.path.join(configs.save_path, 'results.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['acc', 'auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max'])

    for i in range(5):
        print('experiment:', i + 1)
        test_model_path = os.path.join(configs.save_path, 'model{}.tar'.format(i + 1))

        test_model_sd = torch.load(test_model_path)
        test_model.load_state_dict(test_model_sd)
        if torch.cuda.is_available():
            model = test_model.cuda()

        experiment_results = test(test_model, test_data)

        with open(os.path.join(configs.save_path, 'results.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(experiment_results)