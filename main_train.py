import argparse
from dataclasses import dataclass
import os
import datetime
import time
import torch
import transformers
import yaml
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

import utils.misc as misc
from reid_datasets.interactive_build import build_interactive_dataloader
from engine_interactive_train import train_one_epoch, prepare_data
from solver import build_lr_scheduler
from solver.build import build_optimizer
from utils.iotools import LoggerX, save_model
from train_llava_reid import ModelConfig, LlavaForPersonReID
from utils.args_parser import get_args_parser


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    args.output_dir = os.path.join(args.output_dir, args.dataset_name + '_' + args.run_name)
    logger = LoggerX(args)

    parser = transformers.HfArgumentParser(ModelConfig)
    model_cfg = parser.parse_dict(vars(args), allow_extra_keys=True)[0]

    if args.seed is not None:
        misc.fix_random_seed(args.seed)
        logger.info('enable cudnn.deterministic, seed fixed: {}'.format(args.seed))

    logger.debug('Using {} gpus'.format(misc.get_world_size()))
    train_loader, val_img_loader, val_txt_loader, num_classes = build_interactive_dataloader(args,
                                                                                             transforms=False,
                                                                                             logger=logger)

    print("num_classes", num_classes)
    model_cfg.num_classes = num_classes
    model = LlavaForPersonReID(config=model_cfg,
                               llava_config=None,
                               logger=logger)
    if args.stage in ["warmup_selector"]:
        logger.debug('Loading gallery features')
        pt = torch.load(os.path.join(model_cfg.output_dir, "preprocessed_data.pt"), map_location='cpu')
        model.set_gallery(gallery_cls=pt["image_feats"], gallery_image_path=pt["image_paths"])

    print(model)

    model.to(device)
    model_wo_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=False)
        model_wo_ddp = model.module

    if args.stage == 'prepare_data':
        prepare_data(model, train_loader, logger, device, args)
        return

    optimizer = build_optimizer(args, model_wo_ddp, logger)
    scheduler = build_lr_scheduler(args, optimizer)
    best_result = {'epoch': -1, 'R1': 0}

    start_time = time.time()

    for epoch in range(args.start_epoch, args.num_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_state = train_one_epoch(
            model, train_loader, val_img_loader, val_txt_loader,
            optimizer, scheduler, logger,
            device, epoch,
            args
        )
        logger.wandb_log(train_state)
        if "f-R1" in train_state and train_state['f-R1'] >= best_result['R1']:
            best_result = {'epoch': epoch, 'R1': train_state['f-R1']}
            save_model(name='checkpoint',
                       model=model_wo_ddp.retrieval_model,
                       save_path=os.path.join(args.output_dir, 'retrieval_model_mix_IRRA'),
                       logger=logger, args=args)
        elif model_cfg.stage == "warmup_selector" and epoch + 1 == args.num_epoch:
            save_model(name="selector",
                       model=model_wo_ddp.selector,
                       save_path=os.path.join(args.output_dir, 'selector_model-top-k'),
                       logger=logger, args=args)

    logger.info('Best results: {}'.format(best_result))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

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
        configs.update(configs['stage_config'][configs['stage']])
        del configs['stage_config']

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)
    main(args)
