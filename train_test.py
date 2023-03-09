import numpy as np
import torch
import pandas as pd
from sklearn import metrics
from torch.nn import functional as F
import torch.nn as nn


def disable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train(False)


def merge_result(id_list, label_list, score_list, method):
    """Merge predicted results of all bags for each patient"""

    assert method in ["max", "mean"]
    merge_method = np.max if method == "max" else np.mean

    df = pd.DataFrame()
    df["id"] = id_list
    df["label"] = label_list
    df["score"] = score_list
    # https://www.jb51.cc/python/438695.html
    df = df.groupby(by=["id", "label"])["score"].apply(list).reset_index()
    df["bag_num"] = df["score"].apply(len)
    df["score"] = df["score"].apply(merge_method, args=(0,))

    return df["id"].tolist(), df["label"].tolist(), df["score"].tolist(), df["bag_num"].tolist()


def compute_confusion_matrix(label_list, predicted_label_list, num_classes=2):
    label_array = np.array(label_list)
    predicted_label_array = np.array(predicted_label_list)
    confusion_matrix = np.bincount(num_classes * label_array + predicted_label_array, minlength=num_classes**2).reshape((num_classes, num_classes))

    return confusion_matrix


def compute_metrics(label_list, predicted_label_list):
    confusion_matrix = compute_confusion_matrix(label_list, predicted_label_list)
    tn, fp, fn, tp = confusion_matrix.flatten()

    acc = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    f1 = (2 * (sens * ppv) / (sens + ppv) + 2 * (spec * npv) / (spec + npv)) / 2
    P = (ppv + npv) / 2
    R = (sens + spec) / 2

    return {"acc": acc, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1, "P": P, "R": R}


def get_best_metrics_and_threshold(label_list, score_list):
    best_metric_dict = {"acc": 0, "sens": 0, "spec": 0, "ppv": 0, "npv": 0, "f1": 0}
    best_threshold = 0

    # search the best metrcis with F1 score (the greater is better)
    for threshold in np.linspace(0, 1, 1000):
        metric_dict, _ = compute_metrics_by_threshold(label_list, score_list, threshold)
        if metric_dict["f1"] > best_metric_dict["f1"]:
            best_metric_dict = metric_dict
            best_threshold = threshold
    best_metric_dict["auc"] = compute_auc(label_list, score_list)

    return best_metric_dict, best_threshold


def compute_metrics_by_threshold(label_list, score_list, threshold):
    # bag will be predicted as the positive (label is 1) when the score is greater than threshold
    predicted_label_list = [1 if score >= threshold else 0 for score in score_list]
    metric_dict = compute_metrics(label_list, predicted_label_list)
    metric_dict["auc"] = compute_auc(label_list, score_list)

    return metric_dict, threshold


def compute_auc(label_list, score_list, multi_class="raise"):
    try:
        # set "multi_class" for computing auc of 2 classes ("raise") and multiple classes ("ovr" or "ovo"), https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        auc = metrics.roc_auc_score(label_list, score_list, multi_class=multi_class)
    except ValueError:
        auc = 0  # handle error when there is only 1 classes in "label_list"
    return auc


def save_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
    # print(f"save {save_path}")

def train_val_test_binary_class_clinical(task_type, epoch, model, data_loader, optimizer):
    total_loss = 0
    label_list = []
    id_list = []
    score_list = []

    if task_type == "train":
        model.train()
        for index, item in enumerate(data_loader, start=1):
            print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
            label = item["label"].cuda()
            clinical_data = item["clinical_data"].cuda() if "clinical_data" in item else None

            optimizer.zero_grad()
            output= model(clinical_data)
            # import pdb;pdb.set_trace()
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            id_list.append(item["patient_id"][0])
            label_list.append(label.item())
            score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
            score_list.append(score)
    else:
        disable_dropout(model)
        with torch.no_grad():
            for index, item in enumerate(data_loader, start=1):
                print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
                label = item["label"].cuda()
                clinical_data = item["clinical_data"].cuda() if "clinical_data" in item else None

                output= model(clinical_data)

                loss = F.cross_entropy(output, label)
                total_loss += loss.item()

                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
                score_list.append(score)

    average_loss = total_loss / len(data_loader)
    metrics_dict, threshold = compute_metrics_by_threshold(label_list, score_list, 0.5)

    print(
        f"\repoch: {epoch}, {task_type}, loss: {average_loss:.3f}, threshold: {threshold}, auc: {metrics_dict['auc']:.3f}, acc: {metrics_dict['acc']:.3f}, sens: {metrics_dict['sens']:.3f}, spec: {metrics_dict['spec']:.3f}, ppv: {metrics_dict['ppv']:.3f}, npv: {metrics_dict['npv']:.3f}, f1: {metrics_dict['f1']:.3f}, P: {metrics_dict['P']:.3f}, R: {metrics_dict['R']:.3f}"
    )

    return metrics_dict["f1"]

def train_val_test_binary_class(task_type, epoch, model, model_t, data_loader, optimizer, optimizer_t, recoder, writer, merge_method, kd):
    total_loss = 0
    label_list = []
    score_list = []  # [score_bag_0, score_bag_1, ..., score_bag_n]
    id_list = []
    patch_path_list = []
    WSI_feas_list = []
    # attention_value_list = []  # [attention_00, attention_01, ..., attention_10, attention_11, ..., attention_n0, attention_n1, ...]

    if task_type == "train":
        model.train()
        model_t.train()
        # disable_dropout(model_t)
        for index, item in enumerate(data_loader, start=1):
            print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
            bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
            clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

            if kd:
                optimizer_t.zero_grad()
                output_t, prompt_t, img_feas_t = model_t(bag_tensor, clinical_data)

                optimizer.zero_grad()
                output, prompt, img_feas = model(bag_tensor, clinical_data)

                loss_t_c = F.cross_entropy(output_t, label)
                loss_kd_imgfeas = F.mse_loss(img_feas_t, img_feas.detach(), reduction='mean', size_average=True)
                loss_t =  0.2 * loss_kd_imgfeas + loss_t_c

                loss = F.cross_entropy(output, label)
                loss_kd_output = nn.KLDivLoss(log_target=True)(F.log_softmax(output, dim=-1)/1.2, F.log_softmax(output_t.detach()/1.2, dim=-1))  # SoftCrossEntropy(output, output_t.detach())#F.mse_loss(output, output_t.detach(), reduction='mean', size_average=True)#
                loss_kd_fused_feas = F.mse_loss(torch.cat([img_feas.detach(), prompt], dim=-1), torch.cat([img_feas_t, prompt_t], dim=-1).detach(), reduction='mean', size_average=True)
                loss = loss + 0.5 * loss_kd_fused_feas + 0.5 * loss_kd_output

                loss_t.backward()
                optimizer_t.step()

                loss.backward()
                optimizer.step()

            else:
                optimizer.zero_grad()
                output, fused_feas, _ = model(bag_tensor, clinical_data)
                loss = F.cross_entropy(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            id_list.append(item["patient_id"][0])
            label_list.append(label.item())
            score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
            score_list.append(score)
            patch_path_list.extend([p[0] for p in item["patch_paths"]])
            # attention_value_list.extend(attention_value[0].cpu().tolist())
    else:
        # model.eavl()
        disable_dropout(model)
        disable_dropout(model_t)
        with torch.no_grad():
            for index, item in enumerate(data_loader, start=1):
                print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
                bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
                clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

                if kd:
                    if item["miss"]:
                        output, fused_feas, _ = model(bag_tensor, clinical_data)
                    else:
                        output, fused_feas, _ = model_t(bag_tensor, clinical_data)
                else:
                    output, _, fused_feas = model(bag_tensor, clinical_data)
                    # if item["miss"]:
                    #     output_c = model_t(clinical_data)

                # output, attention_value = model(bag_tensor, clinical_data)
                loss = F.cross_entropy(output, label)
                total_loss += loss.item()

                WSI_feas_list.append(fused_feas.data.cpu().numpy())
                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
                # if item["miss"]:
                #     score = 0.7*score + 0.3*F.softmax(output_c, dim=-1).squeeze(dim=0)[1].cpu().item()
                score_list.append(score)
                patch_path_list.extend([p[0] for p in item["patch_paths"]])
                # attention_value_list.extend(attention_value[0].cpu().tolist())
    np.save('WSI_feas_img_only.npy',np.array(WSI_feas_list))
    np.save('score_list_img_only.npy',np.array(score_list))
    np.save('id_list.npy',np.array(id_list))
    np.save('label_list.npy',np.array(label_list))
    # recoder.record_attention_value(patch_path_list, attention_value_list, epoch)
    if merge_method != "not_use":
        id_list, label_list, score_list, bag_num_list = merge_result(id_list, label_list, score_list, merge_method)
        recoder.record_score_value(id_list, label_list, bag_num_list, score_list, epoch)

    average_loss = total_loss / len(data_loader)
    metrics_dict, threshold = compute_metrics_by_threshold(label_list, score_list, 0.47)

    print(
        f"\repoch: {epoch}, {task_type}, loss: {average_loss:.3f}, threshold: {threshold}, auc: {metrics_dict['auc']:.3f}, acc: {metrics_dict['acc']:.3f}, sens: {metrics_dict['sens']:.3f}, spec: {metrics_dict['spec']:.3f}, ppv: {metrics_dict['ppv']:.3f}, npv: {metrics_dict['npv']:.3f}, f1: {metrics_dict['f1']:.3f}, P: {metrics_dict['P']:.3f}, R: {metrics_dict['R']:.3f}"
    )

    writer.add_scalars("comparison/loss", {f"{task_type}_loss": average_loss}, epoch)
    writer.add_scalars("comparison/auc", {f"{task_type}_auc": metrics_dict["auc"]}, epoch)
    writer.add_scalars(f"metrics/{task_type}", metrics_dict, epoch)

    return metrics_dict["f1"]
