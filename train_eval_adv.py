# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter

from adversarial_attack import FGSMAdvAttack, PGDAdvAttack


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, adv_mode='normal'):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()) + '-adv')

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        adv_attacker = None
        if adv_mode == 'FGSM':
            adv_attacker = FGSMAdvAttack(model)
        elif adv_mode == 'PGD':
            adv_attacker = PGDAdvAttack(model)
        else:
            pass
        for i, (trains, labels, seq_lens) in enumerate(train_iter):
            if adv_mode == 'NORMAL':
                outputs = model(trains)   # (x, seq_len),batch_size = 128
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            elif adv_mode == 'FGSM':
                outputs = model(trains)
                # model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                adv_attacker.attack(epsilon=config.epsilon)
                outputs_adv = model(trains)
                loss_adv = F.cross_entropy(outputs_adv, labels)
                loss_adv.backward()
                adv_attacker.restore()
                optimizer.step()
                model.zero_grad()
            elif adv_mode == 'PGD':
                K = config.K
                outputs = model(trains)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                adv_attacker.backup_grad()
                for t in range(K):
                    adv_attacker.attack(epsilon=config.epsilon, is_first_attack=(t==0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        adv_attacker.restore_grad()
                    output_adv = model(trains)
                    loss_adv = F.cross_entropy(output_adv, labels)
                    loss_adv.backward()
                adv_attacker.restore()
                optimizer.step()
                model.zero_grad()
            elif adv_mode == 'FREE':  #use FreeLA
                K = config.K
                epsilon = config.epsilon
                for t in range(K):
                    outputs = model(trains)
                    loss = F.cross_entropy(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()

                    adv_grad = model.embedding.weight.grad[:trains.size(1)]
                    model.delta = model.delta + epsilon * torch.sign(adv_grad)
                    model.delta.data = torch.max(torch.min(model.delta, epsilon), -epsilon)
                    optimizer.step()
                    adv_grad.zero_()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过1000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    model.is_free_adv = False
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels, seq_lens in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)