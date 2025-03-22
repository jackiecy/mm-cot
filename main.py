import os
import wandb
os.environ["WANDB_PROJECT"] = 'mm-cot'
import numpy as np
import torch
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from model import T5ForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg
from utils_prompt import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate

os.environ['https_proxy'] = 'http://172.20.0.113:12798'
# os.environ['https_proxy'] = 'http://101.43.194.204:9981'
# os.environ["WANDB_MODE"] = "offline"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val')
    parser.add_argument('--test_split', type=str, default='test')
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default="detr", choices=['detr', 'clip', 'resnet','vit', 'region_clip'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/instruct_captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--memo', type=str, default=None, help='the memo of the model which used for generating the model path')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--proj_only', action='store_true', help='only train the proj layer of the model')
    parser.add_argument('--fusion_type', type=str, default="STLA", help='one of the HMF, STLA, RAW')
    parser.add_argument('--dataset', type=str, default="QG", help='one of the QG, QS')

    args = parser.parse_args()
    return args
        
def T5Trainer(
    dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    # if args.evaluate_dir is not None:
    #     args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    test_problems = dataframe['test_problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.model.split('/')[-1]}{'_' + args.memo if args.memo else ''}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, local_files_only=True, input_len=args.input_len, args=args.__dict__)
        name_maps = dataframe['name_maps'] 
        image_features = dataframe['image_features']
        train_set = ScienceQADatasetImg(
            problems,
            train_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
        )
        eval_set = ScienceQADatasetImg(
            test_problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
        )
        test_set = ScienceQADatasetImg(
            test_problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )
        example_set = ScienceQADatasetImg(
            test_problems,
            test_qids[:2],
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model) 
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = ScienceQADatasetStd(
            test_problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )
        
        test_set = ScienceQADatasetStd(
            test_problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())
    if args.proj_only:
        for k, v in model.named_parameters():
            for x in ['image_dense', 'mha_layer', 'gate_dense', 'patch_dense', 'proj']:
                if x not in k:
                    v.requires_grad = False
                else:
                    v.requires_grad = True
                    print(k, v.requires_grad)
                    break
    
    args.num_parameters = model.num_parameters()
    with open(os.path.join(save_dir, "run_args.json"), 'w') as f:
        f.write(json.dumps(vars(args), indent=2, sort_keys=False))
    print(save_dir)
    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)
        
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED" 
        return answer  

    # accuracy for answer inference
    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct +=1
        return {'accuracy': 1.0*correct/len(targets)}
    
    # rougel for rationale generation
    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="wandb" if args.wandb else "none",
            run_name=f"{args.model.split('/')[-1]}{'_' + args.memo if args.memo else ''}_ep{args.epoch}"
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            # load_best_model_at_end=True,
            report_to="wandb" if args.wandb else "none",
            run_name=f"{args.model.split('/')[-1]}{'_' + args.memo if args.memo else ''}_ep{args.epoch}"
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=example_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel
    )

    if args.epoch > 0:
        print("train:")
        trainer.train()
        trainer.save_model(save_dir)

    # print("evaluate:")
    # metrics = trainer.evaluate(eval_dataset = test_set, max_length=args.output_len)
    # trainer.log_metrics("test", metrics)
    # trainer.save_metrics("test", metrics)
    
    print("example:")
    predict_results = trainer.predict(test_dataset=example_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(preds, targets)
        
    print("predict:")
    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results_ans = {}
        results_rationale = {}
        results_reference = {}

        num_fail = 0
        for idx, qid in enumerate(test_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            extract_pred = extract_ans(pred)
            if extract_pred != "FAILED":
                if extract_pred in args.options:
                    extract_pred = args.options.index(extract_pred)
                else:
                    extract_pred = random.choice(range(0,len(args.options)))
            else:
                num_fail += 1
                extract_pred = random.choice(range(len(args.options))) # random choose one option
            results_ans[str(qid)] = extract_pred
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref
        # print(results_rationale)
        preds = [pred.strip() for pred in preds]
        output_prediction_file = os.path.join(save_dir, "predictions_ans_test.json")
        print(f"write to file before compute scores: {output_prediction_file}")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps({
                "results_ans": results_ans,
                "results_rationale": results_rationale,
                "results_reference": results_reference,
                "num_fail": num_fail,
                "preds": preds,
                 "labels": targets}, indent=4))
            
        # with open("../experiments/mm-cot-base-rationale_None_ep1_None_ep9_None_ep0/predictions_ans_test.json") as f:
        #     kk = json.load(f)
        # results_ans, results_rationale, results_reference, preds, targets, num_fail = \
        # kk['results_ans'], kk['results_rationale'], kk['results_reference'], kk['preds'], kk['labels'], kk['num_fail']
        
        print("\tcompute scores:")
        scores = get_scores(results_ans, results_rationale, results_reference, os.path.join(args.data_root, "scienceqa/problems.json"), args)
        output_data = {
                "num_fail": num_fail,
                "scores": scores,
                "preds": preds,
                 "labels": targets}
        print(f"write to file: {output_prediction_file}")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))

    # generate the rationale for the eval set
    # if args.prompt_format == "QCM-LE" or args.prompt_format == "QCM-E":
    #     torch.cuda.empty_cache()
    #     del predict_results, preds, targets
    #     print("generate rationale:")
    #     predict_results = trainer.predict(test_dataset=eval_set, max_length=args.output_len)
    #     if trainer.is_world_process_zero():
    #         if args.use_generate:
    #             preds, targets = predict_results.predictions, predict_results.label_ids
    #         else:
    #             preds = predict_results.predictions[0]
    #             targets = predict_results.label_ids
    #             preds = preds.argmax(axis=2)
    #         preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    #         preds = tokenizer.batch_decode(
    #             preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #         )
    #         targets = tokenizer.batch_decode(
    #             targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #         )
    #         preds = [pred.strip() for pred in preds]
    #         output_data = {
    #             "problems": [problems[qid] for qid in val_qids],
    #             "preds": preds,
    #             "labels": targets}
    #         output_prediction_file = os.path.join(save_dir,"predictions_ans_eval.json")
    #         print(f"write to file: {output_prediction_file}")
    #         with open(output_prediction_file, "w") as writer:
    #             writer.write(json.dumps(output_data, indent=4))
    

if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features, test_problems = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features, 'test_problems': test_problems}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids}

    T5Trainer(
        dataframe=dataframe,
        args = args
    )
