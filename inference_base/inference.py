import os
import sys
import time
import argparse
import datetime as dt

import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

import load_data
import llama_model
from tokenizer import tokenization

if __name__ == "__main__":

    # initiallize multiprocess
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl", world_size=4, rank=local_rank)
    initialize_model_parallel(4)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(1)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    
    parser = argparse.ArgumentParser(
        prog="dataloader",
        description="load hellaswag dataset with preprocessing"
    )
    parser.add_argument("--data-path",
                        type=str,
                        default="/home1/ohs/cechallenge/inference_base/dataset/hellaswag_val.jsonl",
                        help="data path")
    parser.add_argument("--tokenizer-path",
                        type=str,
                        default="/home1/ohs/cechallenge/inference_base/tokenizer/tokenizer.model",
                        help="tokenizer path")
    parser.add_argument("--ckpt-path",
                        type=str,
                        default="/home1/ohs/cechallenge/ckpt/origin/",
                        help="checkpoint path")

    args = parser.parse_args()

    # loaing and processing data
    print("loading data from jsonl")
    ddict = load_data.load_hellaswag_jsonl(args.data_path)
    print("processing data")
    doc = load_data.process_doc(ddict)
    print("concatenating context and continuation")
    concat_lst = load_data.concat_doc_ctx_cont(doc)

    # prepare model
    print("loading model")
    load_args = {}
    load_args["ckpt_dir"] = args.ckpt_path
    load_args["tokenizer_path"] = args.tokenizer_path
    load_args["local_rank"] = local_rank
    load_args["max_seq_len"] = 1024
    load_args["world_size"] = world_size
    load_args["max_batch_size"] = 4
    
    llama_model = llama_model.load(**load_args)
    llama_model.model.to("cuda")

    # get accuracy
    now_dt = dt.datetime.now()
    print("start measure at", now_dt.strftime("%H시 %M분 %S초"))
    start_time = time.time()
    correct = 0
    for i, concat in enumerate(concat_lst):
        new_reqs = tokenization.loglikelihood(llama_model.tokenizer, concat)

        res = []
        
        re_ord = tokenization.reorder(new_reqs, tokenization.collate)

        n_reordered_requests = len(re_ord)

        inps = []
        cont_toks_list = []
        inplens = []
        padding_length = 1024
        
        for j, (_, context_enc, continuation_enc) in enumerate(re_ord):

            max_length = 1024
            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= max_length
    
            # how this all works:
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # gpt2    \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice
    
            # when too long to fit in context, truncate from the left
            inp = torch.tensor(
                (context_enc + continuation_enc)[-(max_length + 1) :][:-1],
                dtype=torch.long,
            ).to("cuda")
            (inplen,) = inp.shape
    
            cont = continuation_enc
    
            # since in _collate we make sure length is descending, the longest is always the first one.
            padding_length = (
                padding_length if padding_length is not None else inplen
            )
    
            # pad length from seq to padding_length
            inp = torch.cat(
                [
                    inp,  # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(
                        inp.device
                    ),  # [padding_length - seq]
                ],
                dim=0,
            )
    
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
        multi_logits = llama_model.model.forward(tokens = batched_inps, start_pos = 0)
        multi_logits = torch.nn.functional.log_softmax(multi_logits, dim=-1)
        
        for k, ((cache_key, _, _), logits, inp, inplen, cont_toks) in enumerate(zip(
            re_ord, multi_logits, inps, inplens, cont_toks_list)):
            # Slice to original seq length
            contlen = len(cont_toks)
            # if "virtual tokens" (from prompt tuning) are added, inplen is larger
            inplen = inplen + (logits.shape[0] - padding_length)  
            logits = logits[inplen - contlen : inplen].unsqueeze(
                0
            )  # [1, seq, vocab]

            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                0
            ).to("cuda")  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()

            # Obtain log-probs at the corresponding continuation token indices
            # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                -1
            )  # [1, seq]

            # Answer: (log prob, is-exact-match)
            answer = (float(logits.sum()), bool(max_equal))
            res.append(answer)

        res = tokenization.get_original(new_reqs, res, tokenization.collate)
        res = [x[0] for x, req in zip(res, concat)]
        if res.index(max(res)) == doc[i]["gold"]:
            correct += 1
            
        if i %100 == 0:
            now_dt = dt.datetime.now()
            print("at", local_rank, "processed", i," examples", now_dt.strftime("%H시 %M분 %S초"))
            print("already guess right", correct, "examples!")
            
    
    end_time = time.time()
    total_time = end_time - start_time
    acc = correct / len(concat_lst)
    acc_percent = acc*100
    print(f"measure complete: {total_time:>.2f}")
    print(correct)
    print(f"accuracy, {acc_percent:<.3f} %")
    print