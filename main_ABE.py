'''
Author: Xin ZHou
Time: July 19, 2021
Function: Bert for DNA pretraining
'''

import os

# import torchsnooper

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from transformers import BertForPreTraining, BertConfig
import sys

sys.path.append('.\\model')
from model.bert import BERT
import torch
import argparse
from torch.utils.data.dataloader import DataLoader
from datasets import MaskedLanguageModelingDataset, MaskedLanguageModelingTestDataset
import time
from log import Logger
import numpy as np
import math
import datetime
# from apex import amp
# from apex.parallel import DistributedDataParallel
import torch.distributed as dist
import random
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.multiprocessing as mp


# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device('cuda:3')



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def params_parser():
    parser = argparse.ArgumentParser(description="Hyper-parameters for bert")
    parser.add_argument('--warmup_steps', default=1000, type=int,
                        help='warmup learning steps')
    # parser.add_argument('--vocab_size', default=4106, type=int,
    #                     help='vocab size')
    parser.add_argument('--vocab_size', default=10, type=int,
                        help='vocab size')
    parser.add_argument('--seq_len', default=512, type=int,
                        help='seq length')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    # parser.add_argument(
    #     "--vocab_path", default="./vocab.txt", type=str, help="vocab path ")
    parser.add_argument(
        "--vocab_path", default="iupac", type=str, help="vocab path ")
    parser.add_argument(
        "--model_name", default="bert-base-cased", type=str, help="bert type loaded")
    parser.add_argument(
        "--epochs", default=10, type=int, help="training epoches")
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument(
        "--test_batch_size", default=32, type=int, help="batch size for test")
    # parser.add_argument(
    #     "--train_data", default="data/ABE/labeled/HEK293T_12kChar_ABE-CP1040.csv", type=str, help="directory of train "
    #                                                                                               "data")
    parser.add_argument(
        "--train_data", default="./test_data.csv", type=str, help="directory of train "
                                                                  "data")

    # "--train_data", default="data/ABE/labeled/mES_AtoG_ABE.csv", type=str, help="directory of train data")
    parser.add_argument(
        "--val_data", default="./valid_data.csv", type=str, help="directory of val data")
    parser.add_argument(
        "--test_data", default="./test_data.csv", type=str, help="directory of test data")
    parser.add_argument(
        "--log_dir", default="result/log", type=str, help="directory of the log")
    parser.add_argument(
        "--val_step", default=2000, type=int, help="the interval steps tp get the mask accuracy with validate dataset")
    parser.add_argument(
        "--evaluate", default=True, type=bool, help="whether to evaluate with test dataset")
    parser.add_argument(
        "--save_model", default=True, type=bool, help="whether to save the model")
    parser.add_argument(
        "--save_model_dir", default="result/pre_model", type=str, help="file directory to save the best model")
    parser.add_argument(
        "--load_model", default=True, type=bool,
        help="whether load the self-pretrained model or not, if load, then not train")
    parser.add_argument(
        "--load_model_dir",
        default="result/pre_model/2021-09-05-06-45-35",
        type=str, help="if load the self-model, the directory of the self-model")
    parser.add_argument(
        "--mut_dir", default="result/mut_result", type=str, help="file directory to save the mutated results")
    parser.add_argument(
        "--lr", default=2e-04, type=float, help="learning rate for training the model")
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="beta1 for training the model")
    parser.add_argument(
        "--beta2", default=0.999, type=float, help="beta2 for training the model")
    parser.add_argument(
        "--eps", default=1e-08, type=float, help="eps for training the model")
    parser.add_argument(
        "--wdecay", default=0.01, type=float, help="weight decay for training the model")
    parser.add_argument(
        "--amsgrad", default=False, type=bool, help="amsgrad for training the model")
    parser.add_argument(
        "--train_printstep", default=100, type=int, help="print the training result  every some steps")
    parser.add_argument(
        "--hidden_size", default=384, type=int, help="hidden size in bert model")
    parser.add_argument(
        "--num_attention_heads", default=6, type=int, help="number of attention heads in bert model")
    parser.add_argument(
        "--num_hidden_layers", default=6, type=int, help="number if hidden layers in bert model")
    return parser.parse_args()


def loadModel(saveName, model, optimizer):
    # checkpoint = torch.load(saveName, map_location='cpu')
    checkpoint = torch.load(saveName)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def saveModel(model, optimizer, scheduler, save_model_dir, epoch, train_step):
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    saveName = f"{save_model_dir}/{now_time}"
    state = {'model': model.module.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'epoch': epoch,
             'train_step': train_step}
    torch.save(state, saveName)


def eqformat(seq, findn, type):
    tmp = str(np.where(seq == findn)).replace("(array([", "").replace("]),)", "").replace("\n", "").strip().replace(
        "      ", "").replace(",  ", ", ").replace(", ", ",").split(",")
    return [type(x) for x in tmp]


def neqformat(seq, findn, type):
    a = seq[np.where(seq != findn)]
    tmp = str(a).replace("(array([", "").replace("]),)", "").replace("\n", "").strip().replace("      ", "").replace(
        ",  ", ", ").replace("[", "").replace("]", "").split(" ")
    return [type(x) for x in tmp]


def neqformat1(seq, findn):
    return seq[np.where(seq != findn)]


def sort3vec(seq1, seq2, seq3):
    seq1 = list(seq1)
    seq2 = list(seq2)
    seq3 = list(seq3)
    a, b, c = zip(*sorted(zip(seq1, seq2, seq3), reverse=True))
    return a, b, c


def mutDec(ori_ids, pre_ids, pre_pro):
    ori_ids = np.array(ori_ids)
    pre_ids = np.array(pre_ids)
    pre_pro = np.array(pre_pro)
    # infer whether two mtx equal
    posss = (ori_ids != pre_ids).astype(int)
    # sort
    new_pos, new_bases, new_pros = [], [], []
    for index, poss in enumerate(posss):
        pos = eqformat(poss, 1, int)
        # record the mutated base
        base = neqformat(poss * pre_ids[index], 0, int)
        # record the mutated pro
        pro = neqformat1(poss * pre_pro[index], 0.0)
        npros, nbase, npos = sort3vec(pro, base, pos)
        new_pos.append(npos)
        new_bases.append(nbase)
        new_pros.append(npros)
    return new_pos, new_bases, new_pros


def reIdx(mut_pos):
    reix_mut_pos = []


def compute_acc(targets, predicts):
    total_acc = 0.0
    mask_len = len(targets)
    for idx, target in enumerate(targets):
        target = np.array(target)
        mask_num = np.sum(target != -1)
        if mask_num == 0:
            mask_len -= 1
            continue
        predict = predicts[idx]
        mask_idx = np.where(target != -1)
        predict_mask_right = np.sum(np.equal(predict[mask_idx], target[mask_idx]) == True)
        acc = predict_mask_right / mask_num
        total_acc += acc
    return total_acc / mask_len


def fileFormat(file, content, path):
    if path == "ori_tokens":
        lines = str(content).replace("(['", "").replace("[", "").replace("]", "").replace("')", "").split("<eos>")
        for line in lines:
            line = line.replace("', '", "").replace("<cls>", "")
            line += "\n"
            file.write(line)
    else:
        for tup in content:
            line = str(tup).replace("(", "").replace(")", "")
            file.write(line)
            file.write("\n")
    file.close()


def wCont(mut_dir, path, content, test_path):
    file_dir = test_path.split(".csv")[0].split("/")[-1]
    dir = f"{mut_dir}/{file_dir}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    file = open(f"{dir}{path}", 'a')
    fileFormat(file, content, path)


def recordRes(mut_dir, ori_tokens, mut_pos, mut_bases, mut_pros, test_path):
    wCont(mut_dir, "ori_tokens", ori_tokens, test_path)
    wCont(mut_dir, "mut_pos", mut_pos, test_path)
    wCont(mut_dir, "mut_bases", mut_bases, test_path)
    wCont(mut_dir, "mut_pros", mut_pros, test_path)


def process_mask(masked_label, idx, pro):
    l_mask_pos, l_mask_idx, l_mask_pro = [], [], []
    for index, label in enumerate(masked_label):
        mask_pos = np.where(label == 1)
        mask_idx = idx[index][mask_pos]
        mask_pro = pro[index][mask_pos]
        mask_pos = np.array(mask_pos)[0]
        #
        mask_pro, mask_idx, mask_pos = sort3vec(mask_pro, mask_idx, mask_pos)
        #
        l_mask_pos.append(mask_pos)
        l_mask_idx.append(mask_idx)
        l_mask_pro.append(mask_pro)
    return l_mask_pos, l_mask_idx, l_mask_pro


# def main_worker(proc, nprocs, args):
if __name__ == '__main__':
    # load args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = params_parser()
    logger = Logger(args.log_dir).logger
    logger.info(args)
    # device = torch.device("cuda", args.local_rank)
    #@@@@@@
    # dist.init_process_group(backend='nccl')
    # dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=args.local_rank)
    #@@@@@
    # torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)
    #@@@@@@
    # dist.init_process_group(backend='nccl')
    setup_seed(20)
    # dist.init_process_group(backend='nccl')
    # device = torch.device("cuda", args.local_rank)

    # print(args.local_rank)
    # load data
    #@@@@@@
    train_dataset = MaskedLanguageModelingDataset(args.train_data, args.vocab_path)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #sampler=train_sampler,
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False,
                              drop_last=True,
                              collate_fn=train_dataset.collate_fn)
    # print("len(train_loader):", len(train_loader))
    #@@@@@@@
    val_dataset = MaskedLanguageModelingDataset(args.test_data, args.vocab_path)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False,
                            collate_fn=val_dataset.collate_fn)
    # train_dataset1 = MaskedLanguageModelingDataset(args.train_data1, 'train')
    # train_loader1 = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    #                           collate_fn=train_dataset.collate_fn)
    # train_dataset2 = MaskedLanguageModelingDataset(args.train_data2, 'train')
    # train_loader2 = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    #                           collate_fn=train_dataset.collate_fn)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)

    # load BertConfig
    # config_kwargs = {
    #     "hidden_size": args.hidden_size,
    #     "num_attention_heads": args.num_attention_heads,
    #     "num_hidden_layers": args.num_hidden_layers,
    #     "output_hidden_states": True,
    #     "output_attentions": True,
    #     "vocab_size": 10
    # }
    config_kwargs = {
        "hidden": args.hidden_size,
        "attn_heads": args.num_attention_heads,
        "n_layers": args.num_hidden_layers,
        "vocab_size": args.vocab_size,
        "dropout": 0.1,
        "seq_len": args.seq_len,
    }
    # config = BertConfig.from_pretrained(args.model_name, **config_kwargs)
    # load model
    # model1 = BertForPreTraining(config)
    # vocab_size = 10, hidden = 256, n_layers = 6, attn_heads = 8, dropout = 0.1
    model = BERT(**config_kwargs)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
    #                            weight_decay=args.wdecay, amsgrad=False)

    # load_beg = time.time()
    # model, optimizer = loadModel(args.load_model_dir, model, optimizer)
    # load_end = time.time()
    # logger.info("Load the pretrained model. Using time: {:.2f}s.".format(load_end - load_beg))
    # model1.train()
    # model = model.to(device)
    model = model.to(device)
    # define train
    criteria = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # model = amp.initialize(model)
    # model.cuda()
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    #@@@@@
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank,
    #                                                   find_unused_parameters=True)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
    #                                                   )
    # model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank)
    # optimizer = amp.initialize(optimizer)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
    #                             weight_decay=args.wdecay, amsgrad=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps,
                                  betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_loader)
    )
    # optimizer = amp.initialize(optimizer)
    # model, optimizer = loadModel(args.load_model_dir, model, optimizer)
    # dist.barrier()
    model = model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank,
    #                                                   find_unused_parameters=True)

    best_ed = 1e10
    model.train()
    # load self-mode
    if args.load_model:
        # load_beg = time.time()
        # model, optimizer = loadModel(args.load_model_dir, model, optimizer)
        # load_end = time.time()
        # logger.info("Load the pretrained model. Using time: {:.2f}s.".format(load_end - load_beg))
        # else:
        # train
        for epoch in range(args.epochs):
            # TRAIN
            if epoch > 0:
                train_dataset = MaskedLanguageModelingDataset(args.train_data, args.vocab_path)
                # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False,
                                           drop_last=True,
                                          collate_fn=train_dataset.collate_fn)

            train_beg = time.time()
            train_loss = 0
            train_ed = 0
            train_step = 0
            total_tra_acc_withspecial = 0.0
            total_tra_acc_nospecial = 0.0
            # for i, batch in enumerate(zip(train_loader,train_loader1,train_loader2)):
            for tra_i, batch in enumerate(train_loader):
                # print(batch["masked_ids"])
                # print(batch["targets"].shape)
                # print(batch["ori_ids"].shape)
                # sys.exit(0)
                masked_tokens = batch["masked_ids"].cuda()
                # masked_tokens = batch["masked_ids"].cuda()
                masked_label = batch["targets"]
                # print("label: ", masked_label)
                # print("masked_tokens", masked_tokens.shape)
                #
                result = model(masked_tokens)
                # print(result.shape)
                # print("打印result",result[0])
                pred = result.view(-1, args.vocab_size)
                # print("prediction_logits", result.prediction_logits)
                # targ = masked_label.view(-1).to(device)
                targ = masked_label.view(-1).cuda()
                # print("targ", targ)
                #
                MLM_loss = criteria(pred, targ)

                train_loss += MLM_loss.item()
                step_train_ed = math.pow(2, MLM_loss.item())
                train_ed += step_train_ed
                train_step += 1
                #
                optimizer.zero_grad()
                # with amp.scale_loss(MLM_loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                MLM_loss.backward()
                optimizer.step()
                scheduler.step()
                # get masked accuracy
                # @@@
                logits = result.cpu().detach().numpy()
                idx_withspecial = np.argmax(logits, axis=2)
                # print("idx_withspecial", idx_withspecial, idx_withspecial.shape)
                ####tmp = np.argmax(logits[:, :, 1:5], axis=2)
                # print("tmp", tmp, tmp.shape)
                ####idx_nospecial = [x + 1 for x in tmp]

                acc_withspecial = compute_acc(masked_label, idx_withspecial)
                # acc_withspecial = compute_acc(batch["masked_ids"], idx_withspecial)
                ###acc_nospecial = compute_acc(masked_label, idx_nospecial)
                # acc_nospecial = compute_acc(batch["masked_ids"], idx_nospecial)
                total_tra_acc_withspecial += acc_withspecial
                ####total_tra_acc_nospecial += acc_nospecial
                # print(train_step)
                # @@@
                # current_train_ed = train_ed/train_step
                # if current_train_ed < best_ed:
                #     logger.info(
                #         "    Best (Loss)*(Loss) for validate dataset is: {:.2f}. In training epoch: {}. In Training Step: {}. Save model and parameters.".format(
                #             current_train_ed, epoch, train_step))
                #     # save model and parameters
                #     saveModel(model, optimizer, args.save_model_dir, epoch)
                #     best_ed = current_train_ed
                #
                if train_step > 0 and train_step % args.train_printstep == 0:
                    # if math.isnan(total_tra_acc_nospecial/train_step):
                    #     print(total_tra_acc_nospecial)
                    # print(result.prediction_logits.view(-1, config.vocab_size).shape)
                    logger.info(
                        "Epoch: {}.  Train Step: {} / {}. Train Loss: {:.2f}. (Loss)*(Loss): {:.2f}. Mask Acc (special): {:.2f}. ".
                            format(epoch, train_step, len(train_loader), train_loss / train_step,
                                   (train_ed / train_step),
                                   total_tra_acc_withspecial / train_step))
                # validate

                # dist.barrier()
                if (train_step > 0 and train_step % args.val_step == 0):
                    model.eval()
                    val_beg = time.time()
                    val_loss = 0
                    val_step = 0
                    val_ed = 0  # Euclidean distance
                    total_val_acc_withspecial = 0.0
                    total_val_acc_nospecial = 0.0
                    for val_i, val_batch in enumerate(val_loader):
                        val_masked_tokens = val_batch["masked_ids"].cuda()
                        # val_masked_tokens = val_batch["masked_ids"].cuda()
                        # val_masked_label = val_batch["targets"].cuda(non_blocking=True)
                        val_masked_label = val_batch["targets"]
                        with torch.no_grad():
                            # get loss
                            val_result = model(val_masked_tokens)
                            # val_result = model(val_batch["masked_ids"].cuda())
                            val_pred = val_result.view(-1, args.vocab_size)
                            # val_targ = val_masked_label.view(-1).to(device)
                            val_targ = val_masked_label.view(-1).cuda()
                            val_MLM_loss = criteria(val_pred, val_targ)
                        step_val_ed = math.pow(2, val_MLM_loss.item())
                        val_loss += MLM_loss.item()
                        val_ed += step_val_ed
                        val_step += 1
                        # get masked accuracy
                        val_logits = val_result.cpu().detach().numpy()
                        val_idx_withspecial = np.argmax(val_logits, axis=2)
                        ####val_tmp = np.argmax(val_logits[:, :, 1:5], axis=2)
                        ####val_idx_nospecial = [x + 1 for x in val_tmp]
                        val_acc_withspecial = compute_acc(val_masked_label, val_idx_withspecial)
                        #                         val_acc_withspecial = compute_acc(val_batch["masked_ids"], val_idx_withspecial)
                        ####val_acc_nospecial = compute_acc(val_masked_label, val_idx_nospecial)
                        #                         val_acc_nospecial = compute_acc(val_batch["masked_ids"], val_idx_nospecial)
                        total_val_acc_withspecial += val_acc_withspecial
                        #####total_val_acc_nospecial += val_acc_nospecial
                    val_acc_withspecial = total_val_acc_withspecial / val_step
                    ####val_acc_nospecial = total_val_acc_nospecial / val_step
                    val_end = time.time()
                    # if math.isnan(val_acc_nospecial):
                    #     print(total_val_acc_nospecial)
                    # print(val_step)
                    logger.info(
                        "                Epoch: {}. Train Step: {}. Validate Loss: {:.2f}. (Loss)*(Loss): {:.2f}. Validate Time for val data: {:.2f}s. Mask Acc (special): {:.2f}. ".
                            format(epoch, train_step, (val_loss / val_step), (val_ed / val_step), (val_end - val_beg),
                                   val_acc_withspecial))
                    val_loss = val_loss / val_step
                    val_ed = val_ed / val_step

                    # save model and parameters
                    if args.save_model:
                        if val_ed < best_ed:
                            best_ed = val_ed
                            logger.info("                在测试集上loss足够小")
                            save_beg = time.time()
                            saveModel(model, optimizer, scheduler, args.save_model_dir, epoch, train_step)

                            save_end = time.time()
                            logger.info(
                               "                Best (Loss)*(Loss) for validate dataset is: {:.2f}. In training epoch: {}. In Training Step: {}. Save model and parameters. Save time: {:.2f}s".format(
                                   val_ed, epoch, train_step, (save_end - save_beg)))
                # dist.barrier()
            train_end = time.time()
            logger.info(
                "Finish Training Epoch: {}. Train Loss: {:.2f}. (Loss)*(Loss): {:.2f}. Mask Acc (special): {:.2f}.. Train Time: {:.2f}s.".
                    format(epoch, train_loss / train_step, train_ed / train_step,
                           total_tra_acc_withspecial / train_step,
                           train_end - train_beg))

    # evaluate
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    if args.evaluate:
#        test_dataset = MaskedLanguageModelingTestDataset(args.test_data, 'test')
# test_sample= torch.utils.data.distributed.DistributedSampler(test_dataset)
#        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
#                                 collate_fn=test_dataset.collate_fn)
#        for i, batch in enumerate(test_loader):
#            ori_tokenids = batch['ori_ids']
#            masked_tokens = batch['masked_ids'].to(device)
# masked_tokens = batch['masked_ids']
#            masked_label = batch['targets']
#            ori_tokens = batch['ori_tokens']
#            result = model(masked_tokens)
#            logits = result.cpu().detach().numpy()
# the base for each position would mutate to
#            idx = np.argmax(logits, axis=2)
# mutation possibility
#            pro = np.amax(logits, axis=2)
#
#            mut_pos, mut_bases, mut_pros = process_mask(masked_label, idx, pro)
# mut_pos, mut_bases, mut_pros = process_mask(batch['targets'], idx, pro)
# reIdx(mut_pos)
# record the result
#            recordRes(args.mut_dir, ori_tokens, mut_pos, mut_bases, mut_pros, args.test_data)

# args = params_parser()
# mp.spawn(main_worker, nprocs=4, args=(4, args)