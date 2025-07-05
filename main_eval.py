import argparse
import os
import datetime
import time
import torch
import transformers
import yaml
import utils.misc as misc
from engine_eval import interactive_evaluate
from model.plugir_reid import PlugIRForPersonReID
from reid_datasets.interactive_build import build_interactive_dataloader
from engine_interactive_train import train_one_epoch, prepare_data
from model.llava_reid import LlavaForPersonReID
from solver import build_lr_scheduler
from solver.build import build_optimizer
from train_llava_reid import ModelConfig
from utils.iotools import LoggerX, save_model


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Config')

    # config file path
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--interact_round', type=int, default=5)
    parser.add_argument('--num_candidates', type=int, default=10)
    parser.add_argument('--max_answer_length', type=int, default=256)
    parser.add_argument('--max_question_length', type=int, default=384)

    parser.add_argument('--clip_grad', default=None, type=float)

    # backbone settings
    parser.add_argument('--pretrain_choice', default='ViT-B/16')  # whether use pretrained model
    parser.add_argument('--temperature', type=float, default=0.02,
                        help='''initial temperature value, if 0, don't use temperature''')
    parser.add_argument('--img_aug', default=False, action='store_true')

    ##vison trainsformer settings
    parser.add_argument('--img_size', type=tuple, default=(384, 128))
    parser.add_argument('--stride_size', type=int, default=16)

    # text transformer settings
    parser.add_argument('--max_retrieve_length', type=int, default=77)
    parser.add_argument('--vocab_size', type=int, default=49408)

    # dataset
    parser.add_argument('--dataset_name', default='CUHK-PEDES', help='[CUHK-PEDES, ICFG-PEDES, RSTPReid]')
    parser.add_argument('--sampler', default='random', help='choose sampler from [idtentity, random]')
    parser.add_argument('--num_instance', type=int, default=4)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--test', dest='training', default=True, action='store_false')

    # evaluation setting
    parser.add_argument('--eval_period', default=1)
    parser.add_argument('--val_dataset', default='test')  # use val set when evaluate, if test use test set
    parser.add_argument('--print_freq', default=50)

    parser.add_argument('--output_dir', type=str, default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--run_name', type=str, default='OBJ_005Base')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    return parser


def main(args):
    # misc.init_distributed_mode(args)
    # device = torch.device(args.device)
    args.distributed = False
    args.output_dir = os.path.join(args.output_dir, args.dataset_name + '_' + args.run_name)
    logger = LoggerX(args, training=False)

    parser = transformers.HfArgumentParser(ModelConfig)
    model_cfg = parser.parse_dict(vars(args), allow_extra_keys=True)[0]

    if args.seed is not None:
        misc.fix_random_seed(args.seed)
        logger.info('enable cudnn.deterministic, seed fixed: {}'.format(args.seed))

    logger.debug('Using {} gpus'.format(misc.get_world_size()))

    train_loader, val_img_loader, val_txt_loader, num_classes = build_interactive_dataloader(args, transforms=False,
                                                                                             logger=logger)
    model_cfg.num_classes = num_classes
    if os.path.exists(model_cfg.question_model_path):
        model = LlavaForPersonReID(model_cfg, {"model_path": model_cfg.question_model_path}, logger)
    else:
        model = PlugIRForPersonReID(model_cfg, logger)
    start_time = time.time()

    subset_test = -1

    result = interactive_evaluate(model, val_img_loader, val_txt_loader, subset_test, args)
    logger.wandb_log({"Questioner": model_cfg.question_model_path})
    logger.wandb_log({"Selector": model_cfg.selector_model_path})
    logger.wandb_log({"checkpoint_name": args.checkpoint_name})
    for r in range(model_cfg.interact_round + 1):
        logger.wandb_log(result[r])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Evaluation time {}'.format(total_time_str))

    logger.wandb_finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, 'FullLoader'):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())

        # override with mode specified config
        configs.update(configs['stage_config']["eval"])
        configs.update({"stage": "eval"})
        print(f"Update stage config {eval}: {configs['stage_config']['eval']}")
        del configs['stage_config']

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)
    main(args)
