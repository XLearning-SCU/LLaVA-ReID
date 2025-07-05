import datetime
import json
import os.path
import re
import time
from easydict import EasyDict
from copy import deepcopy
from typing import Iterable
import torch

from data.prompt import prompt_question_generator_v3
from data.prompt import wrap_question_prompt
from engine_eval import interactive_evaluate, evaluate
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.llava_reid import delete_prefix
from utils import misc
import torch.distributed as dist

from utils.iotools import LoggerX
from utils.misc import SmoothedValue, concat_all_gather, data_all_gather


def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable,
                    val_img_loader: Iterable, val_text_loader: Iterable,
                    optimizer: torch.optim.Optimizer, scheduler, logger,
                    device: torch.device,
                    epoch: int,
                    args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    if args.stage == "train_retriever":
        metric_logger.add_meter('sdm_loss', misc.SmoothedValue(window_size=1, fmt='{value:.2f} ({global_avg:.2f})'))
        metric_logger.add_meter('id_loss', misc.SmoothedValue(window_size=1, fmt='{value:.2f} ({global_avg:.2f})'))
        metric_logger.add_meter('mlm_loss', misc.SmoothedValue(window_size=1, fmt='{value:.2f} ({global_avg:.2f})'))

    print_freq = 100
    header = 'Epoch: [{}]'.format(epoch)
    data_loader = enumerate(metric_logger.log_every(train_loader, print_freq, header))
    model.train(True)
    optimizer.zero_grad()
    for data_iter_step, batch in data_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        ret = model(batch)
        total_loss = sum([v for k, v in ret.items() if "loss" in k])

        metric_logger.update(**ret)
        metric_logger.update(lr=scheduler.get_lr()[0])
        metric_logger.update(loss=total_loss.item())
        if '1_R1' in ret:
            metric_logger.update(_1_R1=ret['1_R1'])
            metric_logger.update(_2_R1=ret['2_R1'])
            metric_logger.update(_3_R1=ret['3_R1'])

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

    scheduler.step()

    result = {}
    model.eval()
    if epoch % args.eval_period == 0 or epoch == args.num_epoch - 1:
        subset_test = -1 if epoch == args.num_epoch - 1 else 50

        if args.stage == 'eval':
            result = interactive_evaluate(model, val_img_loader, val_text_loader, subset_test, args)
        elif args.stage == 'train_retriever':
            result = evaluate(model, val_img_loader, val_text_loader, args)
        else:
            result['R1'] = 0
        result['epoch'] = epoch
    torch.cuda.empty_cache()
    if args.distributed:
        dist.barrier()

    return result


@torch.no_grad()
def prepare_data(model: torch.nn.Module,
                 train_loader: Iterable, logger: LoggerX,
                 device: torch.device,
                 args=None, ):
    metric_logger = misc.MetricLogger(delimiter="  ")
    n_samples = len(train_loader.dataset)
    image_feats = None
    image_pid = torch.zeros(n_samples, dtype=torch.long, device=device)
    idx_cnt = torch.zeros(n_samples, dtype=torch.long, device=device)
    header = 'Extracting image feature'
    start_time = time.time()
    model.eval()
    for data_iter_step, batch in enumerate(metric_logger.log_every(train_loader, 50, header)):
        batch_input = {'images': batch['images'].to(device)}
        ret = model(batch_input)
        batch_feats = ret.image_feats
        batch_ids = batch['image_ids'].to(device)
        batch_pids = batch['pids'].to(device)
        if image_feats is None:
            image_feats = torch.zeros([n_samples, batch_feats.shape[1]], device=batch_feats.device)

        image_feats[batch_ids, :] = batch_feats
        image_pid[batch_ids] = batch_pids
        idx_cnt[batch_ids] += 1

    dist.all_reduce(image_pid, op=dist.ReduceOp.SUM)
    dist.all_reduce(image_feats, op=dist.ReduceOp.SUM)
    dist.all_reduce(idx_cnt, op=dist.ReduceOp.SUM)
    image_feats = image_feats / idx_cnt.view(-1, 1)
    image_pid = image_pid // idx_cnt

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug('>>> Extracting time {}\n'.format(total_time_str))

    model.module.set_gallery(gallery_cls=image_feats,
                             gallery_image_path=None,
                             gallery_pid=image_pid)

    dist.barrier()
    header = 'Preparing Dialog Data'

    idx_cnt = idx_cnt.fill_(0)
    QA_indices = torch.zeros([n_samples, args.interact_round], dtype=torch.int, device=device)
    for data_iter_step, batch in enumerate(metric_logger.log_every(train_loader, 1, header)):
        batch_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'images'}
        ret = model(batch_input)
        idx_cnt[ret.image_ids] += 1
        QA_indices[ret.image_ids, :] = ret.QA_indices

    dist.barrier()
    dist.all_reduce(idx_cnt, op=dist.ReduceOp.SUM)
    dist.all_reduce(QA_indices, op=dist.ReduceOp.SUM)

    QA_indices = QA_indices // idx_cnt.view(-1, 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug('>>> Dialog time {}\n'.format(total_time_str))

    if misc.is_main_process():
        image_paths = [x['image_path'] for x in train_loader.dataset]
        save_name = f"training_conversations-c{args.num_candidates}.json"
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.debug('>>> Paths time {}\n'.format(total_time_str))

        data = prepare_training_data(image_paths, QA_indices, None, train_loader, args)
        # training_conversations =

        pt_path = os.path.join(args.output_dir, "preprocessed_data.pt")
        torch.save({"image_feats": image_feats.cpu(),
                    # "indices": data.candidate_indices.cpu(),
                    "image_paths": image_paths},
                   pt_path)
        json_path = os.path.join(args.output_dir, "description.json")
        json.dump(data.description, open(json_path, 'w'), indent=4)
        logger.debug(f"Saved gallery features to {pt_path} and image paths to {json_path}")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.debug('>>> Total time {}\n'.format(total_time_str))

        json.dump(data.training_conversations,
                  open(os.path.join(args.output_dir, save_name),
                       'w'),
                  indent=4)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.debug('>>> Save time {}\n'.format(total_time_str))

        logger.debug(">>> Prepare dialog dataset for retrieval model")
        dataset_new = []
        for idx in range(n_samples):
            row = QA_indices[idx]
            if torch.min(row).item() < 0:
                # print('Skipping duplicate elements')
                continue
            annos_this = train_loader.dataset[idx]
            description = annos_this['initial_query']
            answers = [annos_this['answers'][i] for i in QA_indices[idx]]
            for r in range(args.interact_round):
                description += ' ' + delete_prefix(answers[r])
            annos_this.update({'fine-grained_caption': description,
                               'questions': [],
                               'answers': []})
            annos_this.pop("images")
            if idx == 0:
                print(annos_this)
            dataset_new.append(annos_this)

        json.dump(dataset_new, open(os.path.join(args.output_dir, 'finetune_captions.json'), 'w'), indent=4)
    dist.barrier()


def prepare_training_data(image_paths, QA_indices, candidates_indices, train_loader, args):
    # n_samples, n_rounds, num_candidates = candidates_indices.shape
    n_samples, n_rounds = QA_indices.shape
    num_candidates = args.num_candidates
    prompt_q = prompt_question_generator_v3
    training_conversations = []
    for idx in range(n_samples):
        qa_ids = QA_indices[idx]
        annos_this = train_loader.dataset[idx]
        pid = annos_this['pids']
        initial_query = annos_this['initial_query']
        questions, answers = [], []

        for r in range(n_rounds):
            if qa_ids[r] == -100:
                # print("no other questions")
                break
            q_this, a_this = annos_this['questions'][qa_ids[r]], annos_this['answers'][qa_ids[r]]
            # candidate_images = [image_paths[j] for j in candidates_indices[idx, r]]
            candidate_images = [image_paths[idx]]
            conv_this = wrap_question_prompt(prompt_q, initial_query, questions[:r], answers[:r], num_candidates, False)
            description = ' '.join([initial_query] + [delete_prefix(a) for a in answers[:r]])
            data = {
                "id": "{}_{}".format(pid, r),
                "image": candidate_images,
                "conversations": [
                    {
                        "from": "human",
                        "value": conv_this
                    },
                    {
                        "from": "gpt",
                        "value": q_this,
                    }
                ],
                "round": r,
                "description": description
            }
            training_conversations.append(data)
            questions.append(q_this)
            answers.append(a_this)
            if idx == 0:
                print(conv_this)
        if idx % 1000 == 0:
            print(idx, '/', n_samples)

    description_list = [{"description": x["description"], "round": x["round"]} for x in training_conversations]
    len_train = len(training_conversations)

    # add some I don't know cases
    for idx in range(n_samples):
        if torch.randn(1).item() < 1.5:
            continue
        qa_ids = QA_indices[idx]
        annos_this = train_loader.dataset[idx]
        pid = annos_this['pids']
        initial_query = annos_this['initial_query']
        questions, answers = [], []

        for r in range(n_rounds):
            if qa_ids[r] == -100:
                # print("no other questions")
                break
            q_this, a_this = annos_this['questions'][qa_ids[r]], annos_this['answers'][qa_ids[r]]
            candidate_images = [image_paths[idx]]
            conv_this = wrap_question_prompt(prompt_q, initial_query, questions[:r], answers[:r], num_candidates, False)
            description = ' '.join([initial_query] + [delete_prefix(a) for a in answers[:r]])
            data = {
                "id": "{}_{}".format(pid, r),
                "image": candidate_images,
                "conversations": [
                    {
                        "from": "human",
                        "value": conv_this
                    },
                    {
                        "from": "gpt",
                        "value": q_this,
                    }
                ],
                "round": r,
                "description": description
            }
            training_conversations.append(data)
            questions.append(q_this)
            answers.append(a_this if torch.randn(1).item() < 1.25 else "I don't know")
            if idx == 0:
                print(conv_this)
        if idx % 1000 == 0:
            print(idx, '/', n_samples)

    len_idk = len(training_conversations) - len_train
    print(f"len_train: {len_train}  len_idk: {len_idk}")
    return EasyDict(dict(training_conversations=training_conversations,
                         description=description_list))
