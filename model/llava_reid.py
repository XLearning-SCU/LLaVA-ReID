import datetime
import os
import sys
import time
import warnings
from copy import deepcopy
from typing import List
import math
import torch
import torch.nn.functional as F
from PIL import Image
from easydict import EasyDict
from torch import nn
from tqdm import tqdm

from data.prompt import prompt_question_generator_v3
from data.prompt import wrap_question_prompt
from reid_datasets.bases import tokenize_simple
from utils.iotools import LoggerX
from utils.metrics import per_sample_ranks, max_discriminative
from model.IRRA import IRRA
from utils import misc
from .clip_model import convert_weights
from utils.simple_tokenizer import SimpleTokenizer

from .llava.conversation import conv_templates
from .llava.mm_utils import tokenizer_image_token, process_images
from .llava.train.train_utils import pad_sequence
from .llava.utils import rank0_print
from .llava_reid_utils import delete_prefix, load_llava_reid_model, AnswerGeneratorSGLang, \
    process_image_train, AnswerVisualGeneratorSGLang
from .selector import Selector, SelectorConfig

warnings.filterwarnings("ignore", category=UserWarning)


class LlavaForPersonReID(nn.Module):
    def __init__(self, config, llava_config=None, logger: LoggerX = None):
        super(LlavaForPersonReID, self).__init__()
        self.retriever_device = "cuda" if misc.is_dist_avail_and_initialized() else "cuda:0"
        self.interact_round = config.interact_round
        self.num_candidates = config.num_candidates
        self.q_length = config.max_question_length
        self.a_length = config.max_answer_length
        self.r_length = config.max_retrieve_length
        self.num_potential_candidates = 200
        self.stage = config.stage

        self.total_time = 0
        self.inference_count = 0

        if llava_config is not None and "data_args" in llava_config:
            self.image_folder = llava_config["data_args"].image_folder

        match self.stage:
            case "train_retriever":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, logger=logger)
                self.forward = self._train_retriever
            case "prepare_data":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, logger=logger)
                self.forward = self._prepare_data
            case "warmup_selector":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                self.forward = self._warmup_selector
            case "train_questioner":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                self.question_model = LlavaForPersonReIDQuestionModel(config, llava_config)
                if config.selector_model_path is not None:
                    self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                    self.load_selector(config, logger)
                self.forward = self._train_questioner_selector
            case "train_selector":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                self.question_model = LlavaForPersonReIDQuestionModel(config, llava_config)
                self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                self.load_selector(config, logger)
                self.forward = self._train_questioner_selector
            case "eval":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                self.question_model = LlavaForPersonReIDQuestionModel(config, llava_config, logger)
                if config.selector_model_path is not None:
                    self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                    self.load_selector(config, logger)

                self.answer_model = AnswerGeneratorSGLang("http://192.168.49.58:10500/v1",
                                                          "API_KEY")

                self.forward = self._inference
            case _:
                raise ValueError("Unknown stage:", self.stage)

    # def forward(self, batch):
    #     raise NotImplementedError

    def _train_retriever(self, batch):
        if not self.training:
            batch_images = batch['images'].to(self.retriever_device)
            image_feats = self.retrieval_model.encode_image(batch_images)
            image_feats = image_feats[:, 0, :]
            return EasyDict({'image_feats': image_feats, 'image_id': batch['image_id']})

        f_cap_ids_list, mlm_ids_batch, mlm_labels_batch = [], [], []
        for i in range(len(batch['fine-grained_caption'])):
            f_cap = batch['fine-grained_caption'][i]
            f_cap_ids = tokenize_simple(f_cap, self.clip_tokenizer, self.r_length)
            mlm_ids, mlm_labels = self.retrieval_model.build_random_masked_tokens_and_labels(f_cap_ids.cpu().numpy(),
                                                                                             self.clip_tokenizer)

            f_cap_ids_list.append(f_cap_ids)
            mlm_ids_batch.append(mlm_ids)
            mlm_labels_batch.append(mlm_labels)

        f_cap_ids_list = torch.stack(f_cap_ids_list, dim=0).cuda()
        mlm_ids_batch = torch.stack(mlm_ids_batch, dim=0).cuda()
        mlm_labels_batch = torch.stack(mlm_labels_batch, dim=0).cuda()
        # m_cap_ids = torch.stack(m_cap_ids, dim=0).cuda()

        batch = {
            'pids': batch['pids'],
            'image_ids': batch['image_ids'],
            'images': batch['images'],
            'caption_ids': f_cap_ids_list,
            'mlm_ids': mlm_ids_batch,
            'mlm_labels': mlm_labels_batch
        }
        ret = self.retrieval_model(batch)
        return ret

    @torch.no_grad()
    def _inference(self, batch):
        '''
        Batch size = N
        image encode:
        batch = {
            'image': Tensor
        }
        text interact:
        batch = {
            'coarse_grained_caption': [List] coarse-grained_caption,
            'fine_grained_caption': [List]fine-grained_caption,
        }
        '''
        if "images" in batch:
            batch_images = batch["images"].to(self.retriever_device)
            image_feats = self.retrieval_model.encode_image(batch_images)
            image_feats = image_feats[:, 0, :]
            return EasyDict({"image_feats": image_feats, "image_id": batch["image_id"]})

        B = len(batch["initial_description"])
        initial_description = batch["initial_description"]
        collection = deepcopy(batch["initial_description"])
        questions_log = [[] for _ in range(B)]
        answers_log = [[] for _ in range(B)]
        ret_dict = EasyDict({"text_id": batch["text_id"],
                             "text_feats": [],
                             "conversations": []})
        for i in range(B):
            conversation = {"id": batch['text_id'][i].item(),
                            "initial_query": batch['initial_description'][i],
                            "interaction": []}
            ret_dict.conversations.append(conversation)

        for r in tqdm(range(self.interact_round), desc="Interactive retrieving", file=sys.stdout, ncols=75):
            text_feats = self.encode_text(collection)
            k = torch.full([B], fill_value=math.ceil(self.num_potential_candidates / (r + 1)),
                           device=self.retriever_device)
            image_indices, image_feats, padding_mask = self.get_topK_images(text_feats, k)

            # start_time = time.time()
            if hasattr(self, "selector"):
                text_feats = text_feats.to(torch.bfloat16)
                image_feats = image_feats[:, 1:].to(torch.bfloat16)
                padding_mask = padding_mask[:, 1:]
                khot = self.selector(text_feats, image_feats, padding_mask=padding_mask, temperature=None)
                khot = torch.cat([torch.ones([B, 1], device=khot.device), khot], dim=1)
            else:
                khot = self.get_topk_images_mask(text_feats, image_feats, padding_mask)

            image_indices = image_indices[khot.bool()].view(B, -1)

            image_paths = [[self.gallery_path[i] for i in image_indices[idx]] for idx in range(B)]

            questions = self.question_model(initial_description, image_paths, questions_log, answers_log)

            answers = self.answer_model(batch['fine-grained_description'], questions)

            collection = [(collection[i] + ' ' + delete_prefix(answers[i])).strip() for i in range(B)]

            for i in range(B):
                questions_log[i].append(questions[i])
                answers_log[i].append(answers[i])
                ret_dict.conversations[i]["interaction"].append({
                    "round": r,
                    "question": questions[i],
                    "answer": answers[i],
                    "candidate": image_paths[i]
                })

            ret_dict.text_feats.append(text_feats.unsqueeze(0))

        text_feats = self.encode_text(collection)
        ret_dict.text_feats.append(text_feats.unsqueeze(0))
        ret_dict.text_feats = torch.cat(ret_dict.text_feats, dim=0)

        for i in range(B):
            if i == 0:
                print('------- Case --------')
                print('Fine-grained description', batch['fine-grained_description'][i])
                print('Initial query:', initial_description[i])
                for j in range(self.interact_round):
                    print("Q#{}:".format(j + 1), questions_log[i][j])
                    print("A#{}:".format(j + 1), answers_log[i][j])
                print('Final Query:', collection[i])
                print('')

        return ret_dict

    def _warmup_selector(self, batch):
        descriptions = batch["descriptions"]
        rounds = batch["rounds"]
        text_feats = self.encode_text(descriptions)
        k = torch.ceil(self.num_potential_candidates / (rounds + 1))
        image_indices, image_feats, padding_mask = self.get_topK_images(text_feats, k)
        img_kmeans_mask = self.get_topk_images_mask(text_feats, image_feats, padding_mask)

        text_feats = text_feats.to(torch.bfloat16)
        image_feats = image_feats[:, 1:].to(torch.bfloat16)
        padding_mask = padding_mask[:, 1:]
        khot = self.selector(text_feats, image_feats, padding_mask=padding_mask, temperature=1.0)
        khot = torch.cat([torch.ones([khot.shape[0], 1], device=khot.device), khot], dim=1)

        acc = (khot[:, 1:] == img_kmeans_mask[:, 1:].bool()).float()[img_kmeans_mask[:, 1:].bool()].mean()

        khot = khot[img_kmeans_mask.bool()]
        img_kmeans_mask = img_kmeans_mask[img_kmeans_mask.bool()]

        selector_loss = nn.BCELoss()(khot, img_kmeans_mask.to(torch.float))
        return dict(loss=selector_loss, acc=acc)

    def _train_questioner_selector(self,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   labels: torch.Tensor,
                                   n_rounds: torch.Tensor,
                                   descriptions: List[str],
                                   temperature: float = None):
        B = len(descriptions)
        text_feats = self.encode_text(descriptions)
        k = torch.ceil(self.num_potential_candidates / (n_rounds + 1))
        # k = torch.full([B], fill_value=self.num_potential_candidates, device=self.retriever_device)
        image_indices, image_feats, padding_mask = self.get_topK_images(text_feats, k)
        if temperature is None:
            khot = self.get_topk_images_mask(text_feats, image_feats, padding_mask)
        else:
            text_feats = text_feats.to(torch.bfloat16)
            image_feats = image_feats[:, 1:].to(torch.bfloat16)
            padding_mask = padding_mask[:, 1:]
            khot = self.selector(text_feats, image_feats, padding_mask=padding_mask, temperature=temperature)
            khot = torch.cat([torch.ones([B, 1], device=khot.device), khot], dim=1)

        image_indices = image_indices[khot.bool()].view(B, self.num_candidates)
        khot_grad_c = khot[torch.nonzero(khot, as_tuple=True)]

        image_paths = [[self.gallery_path[i] for i in image_indices[idx]] for idx in range(B)]
        processor = self.question_model.image_processor
        images = [[process_image_train(im, processor, self.image_folder) for im in sl] for sl in image_paths]
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": [im[0].to(torch.bfloat16) for im_list in images for im in im_list],
            "image_sizes": [im[1] for im_list in images for im in im_list],
            "modalities": [im[2] for im_list in images for im in im_list],
            "selector_grad_c": khot_grad_c.to(torch.bfloat16)
        }
        ret = self.question_model(**batch)
        return ret

    @torch.no_grad()
    def get_topK_images(self, text_feats: torch.Tensor, k: torch.Tensor):
        N, max_K = text_feats.shape[0], self.num_potential_candidates
        if text_feats.ndim == 1:
            text_feats = text_feats.unsqueeze(0)
        similarity = text_feats @ self.gallery_cls.t()
        top_k_img_ids = torch.topk(similarity, max_K, dim=1, largest=True).indices
        top_k_img_feats = self.gallery_cls[top_k_img_ids]

        padding_mask = torch.arange(max_K).to(text_feats.device).unsqueeze(0).expand(N, max_K)
        padding_mask = torch.less(padding_mask, k.unsqueeze(1))
        top_k_img_ids[torch.logical_not(padding_mask)] = -1

        return top_k_img_ids, top_k_img_feats, padding_mask

    @torch.no_grad()
    def get_topk_images_mask(self, text_feats: torch.Tensor, image_feats: torch.Tensor, padding_mask: torch.BoolTensor):
        if text_feats.ndim == 1:
            text_feats = text_feats.unsqueeze(0)
        candidate_mask = torch.zeros([image_feats.shape[0], self.num_potential_candidates], dtype=torch.bool,
                                     device=self.retriever_device)
        for idx in range(image_feats.shape[0]):
            top_k_similarity = text_feats[idx].unsqueeze(0) @ image_feats[idx].t()
            candidate_ids = torch.topk(top_k_similarity, self.num_candidates, dim=1).indices
            candidate_mask[idx, candidate_ids] = True
        return candidate_mask

    @torch.no_grad()
    def _prepare_data(self, batch):
        if 'images' in batch:
            batch_images = batch['images'].to(self.retriever_device)
            image_feats = self.retrieval_model.encode_image(batch_images)
            image_feats = image_feats[:, 0, :]
            return EasyDict({'image_feats': image_feats})

        B = len(batch['initial_query'])
        image_ids, pids, answers = batch['image_ids'], batch["pids"], batch['answers']
        dialog_history = batch['initial_query']

        QA_indices = torch.zeros([B, self.interact_round], dtype=torch.int, device='cuda')
        rank_all = {'R1': [], 'R5': [], 'R10': []}
        for r in range(self.interact_round):
            rank_this_round = []
            for i in range(B):
                if r > 0 and QA_indices[i, r - 1] == -100:
                    continue
                dialog = dialog_history[i]
                ans_set = answers[i]
                candidates, cand_index = [], []
                for idx, ans in enumerate(ans_set):
                    ans = delete_prefix(ans)
                    if ans not in dialog:
                        candidates.append(dialog_history[i] + ' ' + ans)
                        cand_index.append(idx)
                if len(candidates) < 2:
                    QA_indices[i, r] = -100
                    continue
                for j in range(len(candidates)):
                    candidates[j] = tokenize_simple(candidates[j], self.clip_tokenizer,
                                                    self.r_length).unsqueeze(0)

                candidates = torch.cat(candidates, dim=0).to(self.retriever_device)
                with torch.no_grad():
                    text_cand_cls = F.normalize(self.retrieval_model.encode_text(candidates), dim=-1, p=2)

                similarity = text_cand_cls @ self.gallery_cls.t()
                query_labels = torch.full((similarity.shape[0],), pids[i], dtype=torch.long,
                                          device=self.retriever_device)

                ranks = per_sample_ranks(similarity, query_labels, self.gallery_pid)
                select_idx = torch.argmin(ranks)

                rank_this_round.append(ranks[select_idx])

                select_idx = cand_index[select_idx]
                QA_indices[i, r] = select_idx
                dialog_history[i] += ' ' + delete_prefix(answers[i][select_idx])

            rank_this_round = torch.tensor(rank_this_round)
            rank_all['R1'].append(round((rank_this_round == 1).float().mean().item() * 100, 1))
            rank_all['R5'].append(round((rank_this_round <= 5).float().mean().item() * 100, 1))
            rank_all['R10'].append(round((rank_this_round <= 10).float().mean().item() * 100, 1))
        print(rank_all)
        ret = EasyDict({
            'image_ids': image_ids,
            'initial_query': batch['initial_query'],
            'QA_indices': QA_indices,
        })
        return ret

    @torch.no_grad()
    def encode_text(self, text):
        with torch.amp.autocast('cuda', enabled=True):
            text_tokens = [tokenize_simple(t, self.clip_tokenizer, self.r_length) for t in text]
            text_tokens = torch.stack(text_tokens, dim=0).to(self.retriever_device)
            text_feats = F.normalize(self.retrieval_model.encode_text(text_tokens), dim=-1)
        return text_feats

    @torch.no_grad()
    def set_gallery(self, gallery_cls, gallery_image_path, **kwargs):
        self.gallery_cls = F.normalize(gallery_cls.to(self.retriever_device), dim=-1)
        self.gallery_path = gallery_image_path
        if "gallery_pid" in kwargs.keys():
            self.gallery_pid = kwargs.pop("gallery_pid").to(self.retriever_device)
        rank0_print(">>> set_gallery unused data: ", kwargs.keys())

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        return self.question_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _build_retrieval_model(self, cfg, device, training=True, logger=None):
        rank0_print("Building retrieval on device: {}".format(device))
        retrieval_model_config = EasyDict({
            'pretrain_choice': cfg.clip_pretrain_model,
            'img_size': cfg.img_size,
            'stride_size': cfg.stride_size,
            'text_length': cfg.max_retrieve_length,
            'temperature': cfg.temperature,
            'vocab_size': cfg.vocab_size,
            'training': cfg.stage == "train_retriever",
            'num_classes': cfg.num_classes
        })
        retrieval_model = IRRA(retrieval_model_config, device)
        ckpt_path = os.path.join(cfg.output_dir, 'retrieval_model_mix_IRRA')
        if self.stage != 'train_retriever':
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError('No checkpoint of retrieval model in {}'.format(ckpt_path))
            else:
                ckpt_path = os.path.join(ckpt_path, 'checkpoint.pth')
                ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
                response = retrieval_model.load_state_dict(ckpt, strict=False)
                if logger is not None:
                    logger.info(f'Load retrieval model from {ckpt_path}. Missing Parameters: {response.missing_keys}')
                else:
                    rank0_print(f'Load retrieval model from {ckpt_path}. Missing Parameters: {response.missing_keys}')

        convert_weights(retrieval_model)
        if not training:
            for name, param in retrieval_model.named_parameters():
                param.requires_grad = False
        retrieval_model.to(device)
        self.clip_tokenizer = SimpleTokenizer()
        return retrieval_model

    def load_selector(self, cfg, logger):
        ckpt_path = os.path.join(cfg.selector_model_path, "selector.pth")
        if not os.path.exists(ckpt_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError('No checkpoint of selector model in {}'.format(ckpt_path))
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            response = self.selector.load_state_dict(ckpt, strict=False)
            if logger is not None:
                logger.info(f'Load selector model from {ckpt_path}: {response.missing_keys}')
            else:
                rank0_print(f'Load selector model from {ckpt_path}. Missing Parameters: {response.missing_keys}')
            self.selector.to(self.retriever_device)


class LlavaForPersonReIDQuestionModel(nn.Module):

    def __init__(self, questioner_config, llava_args, logger=None):
        super(LlavaForPersonReIDQuestionModel, self).__init__()
        model, tokenizer = load_llava_reid_model(llava_args, logger)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = self.model.get_vision_tower().image_processor
        self.num_candidates = questioner_config.num_candidates
        self.conv_template = conv_templates['qwen_reid']
        self.prompt = prompt_question_generator_v3
        self.mini_batch_size = 4
        match questioner_config.stage:
            case "train_questioner" | "train_selector":
                self.forward = self.model.forward
            case "eval":
                self.forward = self._eval

    def generate_mini_batch(self, initial_query: list[str], candidate_images: list[list[str]],
                            questions: list[list[str]], answers: list[list[str]]):
        """
                initial_query: List[str], len = N
                candidate_images: List[List[str]]
                question: List[str] len = N
                labels: List[str] len = N
        """
        B = len(initial_query)
        image_all = [Image.open(img) for sublist in candidate_images for img in sublist]
        image_tensor = process_images(image_all, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16) for _image in image_tensor]
        image_sizes = [_image.size for _image in image_all]

        conv_batch = wrap_question_prompt(self.prompt, initial_query, questions, answers, self.num_candidates, True)

        input_ids_list, answers = [], []
        for idx in range(B):
            conv = deepcopy(self.conv_template)
            conv.append_message(conv.roles[0], conv_batch[idx])
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_text, self.tokenizer, return_tensors='pt')
            input_ids_list.append(input_ids)

        input_ids_batch = pad_sequence(self.tokenizer, input_ids_list, True, self.tokenizer.pad_token_id)
        attention_mask_batch = input_ids_batch.ne(self.tokenizer.pad_token_id).to(self.model.device)
        input_ids_batch = input_ids_batch.to(self.model.device)

        gen_ids = self.model.generate(input_ids_batch, images=image_tensor, image_sizes=image_sizes,
                                      modalities=["image"] * len(image_all),
                                      attention_mask=attention_mask_batch,
                                      do_sample=True, top_p=0.5,
                                      begin_suppress_tokens=[self.tokenizer.eos_token_id],
                                      max_new_tokens=100)
        questions = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for i in range(len(questions)):
            if questions[i].startswith("assistant\n"):
                questions[i] = questions[i].replace("assistant\n", "")
        return questions

    def _eval(self, initial_query: list[str], candidate_images: list[list[str]],
              question: list[list[str]], answer: list[list[str]]):
        question_all = []
        for idx in range(0, len(initial_query), self.mini_batch_size):
            end_idx = min(idx + self.mini_batch_size, len(initial_query))
            question_mini_batch = self.generate_mini_batch(initial_query[idx:end_idx],
                                                           candidate_images[idx:end_idx],
                                                           question[idx:end_idx],
                                                           answer[idx:end_idx])
            question_all.extend(question_mini_batch)
        return question_all

    def get_model(self):
        return self.model
