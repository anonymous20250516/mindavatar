import numpy as np 
import torch
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import torch.nn.functional as F

def compute_topk_accuracy_multi_trials(
    outputs_norm,         # shape: (N, D)
    targets_norm,         # shape: (N, D)
    pca_candidate_norm,   # shape: (M, D)
    targets,              # shape: (N, D), 
    pca_candidate,        # shape: (M, D), 
    num_trials=1000,
    num_classes_list=[2, 10, 20],
    topk_config={2: [1], 10: [1, 3], 20: [1, 3]}
):
    N = outputs_norm.shape[0]
    results = {f'top{k}_acc_{n_cls}': [] for n_cls in num_classes_list for k in topk_config[n_cls]}

    for trial in tqdm(range(num_trials), desc="Running trials"):
        for i in range(N):
            for n_cls in num_classes_list:
                candidate_indices = [j for j in range(len(pca_candidate)) if not torch.equal(pca_candidate[j], targets[i])]
                if len(candidate_indices) < n_cls - 1:
                    continue  

                sampled = np.random.choice(candidate_indices, size=n_cls - 1, replace=False)
                candidates_pool = torch.cat([targets_norm[i:i+1], pca_candidate_norm[sampled]], dim=0)  # gt at 0
                sims = F.cosine_similarity(outputs_norm[i].unsqueeze(0), candidates_pool, dim=1)
                sorted_indices = torch.argsort(sims, descending=True)

                for k in topk_config[n_cls]:
                    hit = 0 in sorted_indices[:k]
                    results[f'top{k}_acc_{n_cls}'].append(float(hit))

    summary = {}
    for key, acc_list in results.items():
        acc_array = np.array(acc_list).reshape(num_trials, -1).mean(axis=1)
        summary[key + "_mean"] = acc_array.mean()
        summary[key + "_std"] = acc_array.std()

    return summary

def constrained_topk(logits, valid_indices, k=1):
    logits = logits.clone()
    mask = torch.ones_like(logits) * float('-inf')
    mask[valid_indices] = logits[valid_indices]
    topk = torch.topk(mask, k=k).indices.tolist()
    return topk

def compute_constrained_topk_predictions(outputs, gender_constraints, k=1):
    topk_preds = []
    for i, logits in enumerate(outputs):
        valid_indices = gender_constraints[i]
        topk = constrained_topk(logits, valid_indices, k)
        topk_preds.append(topk)
    return topk_preds

def process_constrained_topk_task(pred_path, gt_path, gender_constraints, k=1):
    pred = torch.tensor(np.load(pred_path))  # (N, num_classes)
    gt = torch.tensor(np.load(gt_path))      # (N,)
    topk_preds = compute_constrained_topk_predictions(pred, gender_constraints, k)
    
    correct = torch.tensor([
        float(gt[i].item() in topk_preds[i]) for i in range(len(gt))
    ])
    acc = correct.sum().item() / gt.size(0)
    
    return acc, {
        'topk_preds': topk_preds,
        'gt': gt.tolist(),
        'correct': correct.tolist()
    }

if __name__ == '__main__':
    # 路径
    category_csv_path = 'category_mapping_with_gender.csv'
    gender_df = pd.read_csv(category_csv_path)
    gender_map = gender_df.set_index('category_num')['gender'].to_dict()

    task_paths = {
        'cloth': {
            'pred': './results/cloth/sub01/sub_1_test_pred.npy',
            'gt':   './results/cloth/sub01/sub_1_test_gt.npy',
        },
        'texture': {
            'pred': './results/gender/sub01/sub_1_test_pred.npy',
            'gt':   './results/gender/sub01/sub_1_test_gt.npy',
        },
        'hair': {
            'pred': './results/hair/sub01/sub_1_test_pred.npy',
            'gt':   './results/hair/sub01/sub_1_test_gt.npy',
        }
    }

    # texture 不需要修改预测逻辑（不带约束）
    def process_task(pred_path, gt_path):
        pred = torch.tensor(np.load(pred_path))  # (N, num_classes)
        gt = torch.tensor(np.load(gt_path))      # (N,)
        _, preds = torch.max(pred, dim=1)
        correct = (preds == gt).float()
        acc = correct.sum().item() / gt.size(0)
        return acc, {
            'pred': preds.tolist(),
            'gt': gt.tolist(),
            'correct': correct.tolist()
        }

    texture_acc, texture_result = process_task(task_paths['texture']['pred'], task_paths['texture']['gt'])

    texture_gender_list = ['female' if x == 0 else 'male' for x in texture_result['pred']]

    # 构建 cloth 的性别约束类别列表
    cloth_gender_constraints = []
    for gender in texture_gender_list:
        valid_class_ids = [cid for cid, g in gender_map.items() if g == gender]
        cloth_gender_constraints.append(valid_class_ids)

    # 构建 hair 的性别约束类别列表
    hair_gender_constraints = []
    for gender in texture_gender_list:
        if gender == 'male':
            hair_gender_constraints.append([0, 1])
        else:
            hair_gender_constraints.append([2, 3])

    # cloth
    cloth_top3_acc, cloth_top3_result = process_constrained_topk_task(
        task_paths['cloth']['pred'], task_paths['cloth']['gt'], cloth_gender_constraints, k=3
    )

    # hair
    hair_acc, hair_result = process_constrained_topk_task(
        task_paths['hair']['pred'], task_paths['hair']['gt'], hair_gender_constraints
    )

    # 输出准确率
    print("\nSummary of Accuracies with Gender Constraints:")
    print(f"cloth:   {cloth_top3_acc:.4f}")
    print(f"texture: {texture_acc:.4f}")
    print(f"hair:    {hair_acc:.4f}")


    # ===================== Load Data =====================
    pca_candidate_path = './dataset/annotation/id_candidate/pca_candidate.pt'
    pca_candidate = torch.load(pca_candidate_path)

    pca_gt = np.load('./results/id/sub01/sub_1_test_gt.npy') 
    pca_pred = np.load('./results/id/sub01/sub_1_test_pred.npy')
    df = pd.DataFrame(index=range(len(pca_gt)))

    assert pca_gt.shape == pca_pred.shape, "Shape mismatch between pca_gt and pca_pred"
    assert pca_gt.shape[0] == len(df), "Mismatch between number of samples and PCA shape"

    # ===================== CSIM + MSE =====================
    mse_list = []
    csim_list = []

    for gt_vec, pred_vec in zip(pca_gt, pca_pred):
        mse = mean_squared_error(gt_vec, pred_vec)
        csim = cosine_similarity(gt_vec.reshape(1, -1), pred_vec.reshape(1, -1))[0][0]
        mse_list.append(mse)
        csim_list.append(csim)

    df['pca_mse'] = mse_list
    df['pca_csim'] = csim_list
    df['pca_pred_vector'] = [vec.tolist() for vec in pca_pred]
    df['pca_gt_vector'] = [vec.tolist() for vec in pca_gt]

    print('==================ID=====================')
    print('============================================')
    print("Avg MSE:", sum(mse_list) / len(mse_list))
    print("Avg CSIM:", sum(csim_list) / len(csim_list))

    # ===================== Classification Accuracy =====================
    from tqdm import tqdm

    # Normalize for cosine similarity
    pca_candidate_norm = torch.nn.functional.normalize(pca_candidate, dim=1)

    outputs = torch.tensor(pca_pred).float()
    targets = torch.tensor(pca_gt).float()

    outputs_norm = torch.nn.functional.normalize(outputs, dim=1)
    targets_norm = torch.nn.functional.normalize(targets, dim=1)

    N = outputs.shape[0]


    summary = compute_topk_accuracy_multi_trials(
        outputs_norm=outputs_norm,              # predicted vectors
        targets_norm=targets_norm,              # ground-truth vectors
        pca_candidate_norm=pca_candidate_norm,  # all candidates (normalized)
        targets=targets,                        # original GT vectors (not normalized)
        pca_candidate=pca_candidate,            # all candidate vectors (not normalized)
        num_trials=100                         # number of random sampling trials
    )

    # 打印结果
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    # def compute_topk_accuracy(num_classes, topk_list):
    #     topk_correct = {k: 0 for k in topk_list}
    #     for i in range(N):
    #         # 排除自己，采样
    #         candidate_indices = [j for j in range(len(pca_candidate)) if not torch.equal(pca_candidate[j], targets[i])]
    #         if len(candidate_indices) < num_classes - 1:
    #             continue  # 跳过不足的
    #         sampled = np.random.choice(candidate_indices, size=num_classes - 1, replace=False)
    #         candidates_pool = torch.cat([targets_norm[i:i+1], pca_candidate_norm[sampled]], dim=0)
    #         sims = torch.nn.functional.cosine_similarity(outputs_norm[i].unsqueeze(0), candidates_pool, dim=1)
    #         sorted_indices = torch.argsort(sims, descending=True)
    #         for k in topk_list:
    #             if 0 in sorted_indices[:k]:  # gt always at position 0
    #                 topk_correct[k] += 1
    #     accs = {f'top{k}_acc_{num_classes}': topk_correct[k] / N for k in topk_list}
    #     return accs

    # # Run all categories
    # results = {}
    # for num_cls in [2, 10, 20, 50, 100]:
    #     if num_cls == 2:
    #         accs = compute_topk_accuracy(num_cls, [1])
    #     else:
    #         accs = compute_topk_accuracy(num_cls, [1, 3, 5])
    #     results.update(accs)

    # # Print results
    # print("\n=== Classification Accuracy ===")
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")


    # # 保存更新后的表格
    # df.to_csv(save_path, index=False)
    # print(f"\nMerged results with PCA metrics saved to: {save_path}")

    # # 分别计算 male 和 female 的准确率
    # for gender in ['male', 'female']:
    #     gender_mask = df['gender'] == gender
    #     num_samples_gender = gender_mask.sum()

    #     cloth_acc_gender = df.loc[gender_mask, 'cloth_correct'].sum() / num_samples_gender
    #     texture_acc_gender = df.loc[gender_mask, 'texture_correct'].sum() / num_samples_gender
    #     hair_acc_gender = df.loc[gender_mask, 'hair_correct'].sum() / num_samples_gender

    #     print(f"\nAccuracy for {gender.upper()} samples:")
    #     print(f"  cloth:   {cloth_acc_gender:.4f}")
    #     print(f"  texture: {texture_acc_gender:.4f}")
    #     print(f"  hair:    {hair_acc_gender:.4f}")
