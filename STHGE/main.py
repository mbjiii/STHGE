import torch
from sklearn.metrics import f1_score
from utils import EarlyStopping, load_data
from f1_kmeans_nmi_tools import evaluate_results_nc
import numpy as np
import time



def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        embeddings, logits, L_t_HSIC, L_g_HSIC = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1, embeddings


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data(args["dataset"])

    if hasattr(torch, "BoolTensor"):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args["device"])
    labels = labels.to(args["device"])
    train_mask = train_mask.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])

    #if args["hetero"]:
    if args["dataset"] == "ACMRaw":
        from model_hetero import HAN

        model = HAN(
            
            meta_paths=[["pa", "ap"], ["pf", "fp"]],        # “PAP”, "PSP"
            in_size=features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        g = g.to(args["device"])
    else:
        from model import HAN

        model = HAN(
            num_meta_paths=len(g),
            in_size=features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        g = [graph.to(args["device"]) for graph in g]

    stopper = EarlyStopping(patience=args["patience"])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )






    lambda_L_t_HSIC = 5
    lambda_L_g_HSIC = 6

    dur1 = []
    dur2 = []
    dur3 = []
    
    for epoch in range(args["num_epochs"]):
        # train start
        t0 = time.time()
        model.train()
        embeddings, logits, L_t_HSIC, L_g_HSIC= model(g, features)
        # define loss function
        loss_HSIC = lambda_L_t_HSIC * L_t_HSIC + lambda_L_g_HSIC * L_g_HSIC
        loss = loss_fcn(logits[train_mask], labels[train_mask]) + lambda_L_t_HSIC * L_t_HSIC         # logits[train_mask]:选择train_mask中为True的行

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t1 = time.time()
        dur1.append(t1 - t0)

        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[train_mask], labels[train_mask]
        )
        
        # val start
        t0 = time.time()
        val_loss, val_acc, val_micro_f1, val_macro_f1, embeddings = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        t1 = time.time()
        dur2.append(t1 - t0)
        
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print( 
            "Epoch {:d} | Train Loss {:.4f} | Train_time(s) {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} \n "
            "          Val Loss {:.4f} | Val_time(s) {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1,      loss.item(),        np.mean(dur1), train_micro_f1,  train_macro_f1,
                                val_loss.item(),    np.mean(dur2), val_micro_f1,    val_macro_f1,
            )
        )
        

        if early_stop:
            break

    # test start
    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1, embeddings = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    print(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_micro_f1, test_macro_f1
        )
    )

    print("seed = ", args["seed"])
    svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=num_classes)

    # 重复n次训练过程启用：
    '''svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []

    svm_macro_f1_lists.append(svm_macro_f1_list)
    svm_micro_f1_lists.append(svm_micro_f1_list)
    nmi_mean_list.append(nmi_mean)
    nmi_std_list.append(nmi_std)
    ari_mean_list.append(ari_mean)
    ari_std_list.append(ari_std)

    # print out a summary of the evaluations
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists ), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists ), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)
    print('----------------------------------------------------------------')
    print('SVM tests summary')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means tests summary')
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))'''


if __name__ == "__main__":
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument("--patience", type=int, default=10, help="Patience")
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="Use metapath coalescing with DGL's own dataset",
    )
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
