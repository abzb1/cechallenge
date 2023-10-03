import collections

from llama import Tokenizer

def get_llama_tokenizer(path="/home1/ohs/cechallenge/inference_base/tokenizer/tokenizer.model"):
    tokenizer = Tokenizer(model_path=path)
    return tokenizer

def encode_pair(tokenizer, context, continuation):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tokenizer.encode(context + continuation, True, True)
    context_enc = tokenizer.encode(context, True, False)
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc

def loglikelihood(tokenizer, query_choices_lst):
    new_reqs = []
    for context, continuation in query_choices_lst:
        context_enc, continuation_enc = encode_pair(tokenizer, context, continuation)
        new_reqs.append(((context, continuation), context_enc, continuation_enc))

    return new_reqs

def collate(x):
    toks = x[1] + x[2]
    return -len(toks), tuple(toks)

def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())

def reorder(arr, fn):
    
    ordered_arr = list(enumerate(arr))
    ordered_arr = group(ordered_arr, lambda x: fn(x[1]))
    ordered_arr = [([y[0] for y in x], x[0][1]) for x in ordered_arr]
    ordered_arr.sort(key=lambda x: fn(x[1]))
    ordered_arr = [x[1] for x in ordered_arr]
    
    return ordered_arr

def get_original(arr, new_arr, fn):
    size = len(arr)
    reord_arr = list(enumerate(arr))
    reord_arr = group(reord_arr, lambda x: fn(x[1]))
    reord_arr = [([y[0] for y in x], x[0][1]) for x in reord_arr]
    reord_arr.sort(key=lambda x: fn(x[1]))

    res = [None] * size
    cov = [False] * size

    for (inds, _), v in zip(reord_arr, new_arr):
        for ind in inds:
            res[ind] = v
            cov[ind] = True

    assert all(cov)

    return res