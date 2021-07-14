import os
import math
import numpy as np
import torch
import pickle
from utils import *
from models import *
from losses import WeightedCrossEntropy
from config import DefaultConfig
configs = DefaultConfig()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, train_graphs, val_graphs, num=1, re_train_start=0, b_loss=999):
    epochs = configs.epochs - re_train_start
    print('current experiment epochs:', epochs)
    batch_size = configs.batch_size
    lr = configs.learning_rate
    weight_decay = configs.weight_decay
    neg_wt = configs.neg_wt

    model_save_path = configs.save_path
    print(model_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay, nesterov=True)
    loss_fn = WeightedCrossEntropy(neg_wt=neg_wt, device=device)

    model.train()
    train_losses = []
    val_losses = []
    best_loss = b_loss
    count = 0
    for e in range(epochs):
        print("Runing {} epoch".format(e + 1 + re_train_start))
        e_loss = 0.
        for train_g in train_graphs:
            if torch.cuda.is_available():
                train_ag_vertex = torch.FloatTensor(train_g['ag_feat']).cuda()
                train_ag_edge = torch.FloatTensor(train_g['ag_edge_feat']).cuda()
                train_ag_nh_indices = torch.LongTensor(train_g['ag_nh_indices']).cuda()
                train_ag_label = torch.LongTensor(train_g['ag_label']).cuda()
                # train_ag_indices = [[index, label] for index, label in enumerate(train_g['ag_label'])]
                # np.random.shuffle(down_sample(np.array(train_ag_indices)))
                train_ag_indices = [index for index, _ in enumerate(train_g['ag_label'])]
                train_ag_indices = torch.LongTensor(train_ag_indices).cuda()

            label_size = len(train_ag_label)
            iters = math.ceil(label_size / batch_size)
            g_loss = 0.
            for it in range(iters):
                optimizer.zero_grad()
                start = it * batch_size
                end = start + batch_size
                if end > label_size:
                    end = label_size

                # batch_pred = model(train_ag_vertex, train_ag_indices[start:end])
                # batch_pred = model(train_ag_vertex, train_ag_nh_indices, train_ag_indices[start:end])
                batch_pred = model(train_ag_vertex, train_ag_edge, train_ag_nh_indices, train_ag_indices[start:end])
                batch_loss = loss_fn.computer_loss(batch_pred, train_ag_label[start:end])
                b_loss = batch_loss.item()
                g_loss += b_loss
                batch_loss.backward()
                optimizer.step()
            g_loss /= iters
            e_loss += g_loss

        e_loss /= len(train_graphs)
        train_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'epi_train_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        e_loss = 0.
        for val_g in val_graphs:
            if torch.cuda.is_available():
                val_ag_vertex = torch.FloatTensor(val_g['ag_feat']).cuda()
                val_ag_edge = torch.FloatTensor(val_g['ag_edge_feat']).cuda()
                val_ag_nh_indices = torch.LongTensor(val_g['ag_nh_indices']).cuda()
                val_ag_label = torch.LongTensor(val_g['ag_label']).cuda()
                # val_ag_indices = [[index, label] for index, label in enumerate(val_g['ag_label'])]
                # np.random.shuffle(down_sample(np.array(val_ag_indices)))
                val_ag_indices = [index for index, _ in enumerate(val_g['ag_label'])]
                val_ag_indices = torch.LongTensor(val_ag_indices).cuda()

            label_size = len(val_ag_label)
            iters = math.ceil(label_size / batch_size)
            g_loss = 0.
            for it in range(iters):
                optimizer.zero_grad()
                start = it * batch_size
                end = start + batch_size
                if end > label_size:
                    end = label_size

                # batch_pred = model(val_ag_vertex, val_ag_indices[start:end])
                # batch_pred = model(val_ag_vertex, val_ag_nh_indices, val_ag_indices[start:end])
                batch_pred = model(val_ag_vertex, val_ag_edge, val_ag_nh_indices, val_ag_indices[start:end])
                batch_loss = loss_fn.computer_loss(batch_pred, val_ag_label[start:end])
                b_loss = batch_loss.item()
                g_loss += b_loss
            g_loss /= iters
            e_loss += g_loss
        e_loss /= len(val_graphs)
        val_losses.append(e_loss)
        with open(os.path.join(model_save_path, 'epi_val_losses{}.txt'.format(num)), 'a+') as f:
            f.write(str(e_loss) + '\n')

        if best_loss > val_losses[-1]:
            count = 0
            torch.save(model.state_dict(), os.path.join(os.path.join(model_save_path, "model{}.tar".format(num))))
            best_loss = val_losses[-1]
            print("UPDATE\tEpoch {}: train loss {}\tval loss {}".format(e + 1 + re_train_start, train_losses[-1], val_losses[-1]))

        # else:
        #     count += 1
        #     if count >= 5:
        #         with open(os.path.join(model_save_path, 'epi_train_losses{}.txt'.format(num)), 'w') as f:
        #             for i in train_losses:
        #                 f.write(str(i) + '\n')
        #         with open(os.path.join(model_save_path, 'epi_val_losses{}.txt'.format(num)), 'w') as f:
        #             for j in val_losses:
        #                 f.write(str(j) + '\n')
        #         return None

        # if e % 10 == 0:
        #     print("Epoch {}: train loss {}\tval loss {}".format(e + 1, train_losses[-1], val_losses[-1]))


if __name__ == '__main__':
    seeds = [378945, 395408, 252356, 343053, 743746]

    train_path = configs.train_dataset_path
    val_path = configs.val_dataset_path

    # data_context
    with open(train_path, 'rb') as f:
        train_list, train_data = pickle.load(f)

    with open(val_path, 'rb') as f:
        val_list, val_data = pickle.load(f)

    # train_model = NodeAverageModel()
    # train_model = NodeEdgeAverageModel()
    # train_model = BiLSTMNodeAverageModel()
    train_model = BiLSTMNodeEdgeAverageModel()

    # from utils import propress_data
    # propress_data(train_data)
    # propress_data(val_data)

    current_experiment = 1
    trained_epochs = 100
    b_v_loss = 999

    for seed in seeds:
        print('experiment:', current_experiment)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        train_model_dir = os.path.join(configs.save_path, 'model{}.tar'.format(current_experiment))
        if os.path.exists(train_model_dir):
            train_model_sd = torch.load(train_model_dir)
            train_model.load_state_dict(train_model_sd)
            train(train_model, train_data, val_data, current_experiment, trained_epochs, b_v_loss)
            current_experiment += 1
        else:
            train(train_model, train_data, val_data, current_experiment)
            current_experiment += 1