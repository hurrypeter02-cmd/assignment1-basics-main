import regex as re
from collections import Counter



def get_stats(unique_token_list):
    stats = {}
    for token,count in unique_token_list.items():
        for i in range(len(token)-1):
            pair = (token[i],token[i+1])
            stats[pair] = stats.get(pair,0) + count
    return stats

def merge(old_ids,unique_token_list,new_idx):
    new_token_list = Counter()
    old_idx1, old_idx2 = old_ids
    for token,count in unique_token_list.items():
        new_token = []
        i = 0
        found = False
        for i in range(len(token)-1):
            if(token[i] == old_idx1 and token[i+1] == old_idx2):
                found = True
                break;
        if(not found):
            new_token_list[token] += count
            continue
        # Only build a new token tuple when this token contains the target pair.
        i = 0
        while(i<len(token)):
            if(i<len(token)-1 and token[i] == old_idx1 and token[i+1] == old_idx2):
                new_token.append(new_idx)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_token_list[tuple(new_token)] += count
    return new_token_list

def run_train_bpe(input_path,vocab_size,special_tokens):
    with open(input_path,'r',encoding='utf-8') as f:
        text = f.read()
    new_st = []
    for st in special_tokens:
        st = re.escape(st)
        new_st.append(st)
    split_parten = "|".join(new_st)
    text_list = re.split(split_parten,text)
    new_text_list =  []
    gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    for text in text_list:
        new_text_list.extend(re.findall(gpt2pat, text))
    # Pre-tokenize by special_tokens and the GPT-2 pattern; merges must not cross pre-token boundaries.

    unique_token_list = Counter(tuple(text.encode("utf-8")) for text in new_text_list)
    # Convert pre-tokens to byte-id tuples and count duplicate tuples.

    vocab = {i:bytes([i]) for i in range(256)}
    for i,st in enumerate(special_tokens):
        vocab[i+256] = st.encode("utf-8")
    # Initialize vocab with raw bytes first, then append special tokens.

    epoch = vocab_size - 256 - len(special_tokens)
    merge_dic = []
    for i in range(epoch):
        stats = get_stats(unique_token_list)
        if not stats:
            break;
        merge_pair = max(stats,key=lambda p:(stats[p],vocab[p[0]],vocab[p[1]]))
        new_idx = 256 + len(special_tokens) + i
        vocab[new_idx] = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        merge_dic.append((vocab[merge_pair[0]],vocab[merge_pair[1]]))
        unique_token_list = merge(merge_pair,unique_token_list,new_idx)
    # Maintain vocab and merges during training so tie-breaks can compare token bytes.
    return vocab,merge_dic

