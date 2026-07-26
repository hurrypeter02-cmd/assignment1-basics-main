import regex as re
from collections import Counter,defaultdict


def better(value1,value2):
    return value1[:3]>value2[:3]

def push(heap,ct_vcabs_pairs):
    heap.append(ct_vcabs_pairs)
    son = len(heap)-1
    father = (son-1)//2
    while son>0:
        if better(heap[son],heap[father]):
            tmp = heap[father]
            heap[father] = heap[son]
            heap[son] = tmp
            son = father
            father = (son-1)//2
        else:
            break
def pop(heap):
    if not heap:
        return None
    max_value = heap[0]
    last = heap.pop()
    if not heap:
        return max_value
    heap[0] = last
    father = 0
    son1 = 2*father + 1
    son2 = 2*father + 2
    son = son2 if son1<len(heap) and son2<len(heap) and not better(heap[son1],heap[son2]) else son1
    # son 取 son1 的情况下可能越界,此时无需 pop ,直接返回即可,下面的 while 循环条件恰好帮助进行了条件筛选
    while son<len(heap):
        if better(heap[son],heap[father]):
            tmp = heap[father]
            heap[father] = heap[son]
            heap[son] = tmp
            father = son
            son1 = 2*father + 1
            son2 = 2*father + 2
            son = son2 if son1<len(heap) and son2<len(heap) and not better(heap[son1],heap[son2]) else son1
        else:
            break
    return max_value
def init_heap(stats,vocab):
    heap = []
    for pair,count in stats.items():
        push(heap,(count,vocab[pair[0]],vocab[pair[1]],pair))
    return heap
# 优化代码:merge_pair = max(stats,key=lambda p:(stats[p],vocab[p[0]],vocab[p[1]]))

def get_stats(unique_token_list):
    stats = {}
    pair_to_token = defaultdict(set)
    for token,count in unique_token_list.items():
        for pair in zip(token,token[1:]):
            stats[pair] = stats.get(pair,0) + count
            pair_to_token[pair].add(token)
    return stats,pair_to_token # 方便后续 merge 或者 get_stats 不用遍历全局 token ,只需要遍历出现 pair 的 token 

def merge(old_ids,unique_token_list,new_idx,stats,pair_to_token):
    old_idx1, old_idx2 = old_ids
    changed_pairs = set()
    for token in list(pair_to_token[old_ids]):
        if not token:
            continue
        new_token = []
        count = unique_token_list[token]
        i = 0
        while(i<len(token)):
            if(i<len(token)-1 and token[i] == old_idx1 and token[i+1] == old_idx2):
                new_token.append(new_idx)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        # 更新 affected_token
        for pair in zip(token,token[1:]):
            stats[pair] -= count
            changed_pairs.add(pair)
            if stats[pair] <= 0:
                del stats[pair]
            pair_to_token[pair].discard(token)
            if not pair_to_token[pair]:
                del pair_to_token[pair]
        # 删除旧 token 痕迹
        for pair in zip(new_token,new_token[1:]):
            stats[pair] = stats.get(pair,0) + count
            changed_pairs.add(pair)
            pair_to_token[pair].add(tuple(new_token))
        del unique_token_list[token]
        unique_token_list[tuple(new_token)] += count
        # 更新 相关变量
    return unique_token_list,stats,pair_to_token,changed_pairs

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
    # 根据 special_tokens 和 GPT-2 正则进行预分词, 不同 pre-token 之间不能合并

    unique_token_list = Counter(tuple(text.encode("utf-8")) for text in new_text_list)
    # 将去重后的 text 转换成 token tuple, 并用 Counter 统计重复次数

    vocab = {i:bytes([i]) for i in range(256)}
    for i,st in enumerate(special_tokens):
        vocab[i+256] = st.encode("utf-8")
    # 初始化 vocab, 基础字节 token 是 0-255, special_tokens 接在后面

    epoch = vocab_size - 256 - len(special_tokens)
    merge_dic = []
    stats = {}
    for i in range(epoch):
        if not unique_token_list:
            break;
        if not stats:
            stats,pair_to_token = get_stats(unique_token_list)
            heap = init_heap(stats,vocab)

        merge_pair = ()
        while(heap):
            ct_vcabs_pairs = pop(heap)
            pair = ct_vcabs_pairs[3]
            if ct_vcabs_pairs and pair in stats and ct_vcabs_pairs[0] == stats[pair]:
                merge_pair = ct_vcabs_pairs[3]
                break
        # 懒更新 heap 最大堆,O(1) 时间返回 最大 count 的 pair

        new_idx = 256 + len(special_tokens) + i
        vocab[new_idx] = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        merge_dic.append((vocab[merge_pair[0]],vocab[merge_pair[1]]))

        unique_token_list,stats,pair_to_token,changed_pairs = merge(merge_pair,unique_token_list,new_idx,stats,pair_to_token)
        # 增量更新 stats ,同时维护 pair_to_token

        for pair in changed_pairs:
            if pair in stats:
                push(heap,(stats[pair],vocab[pair[0]],vocab[pair[1]],pair))
        # 维护更新 heap 最大堆

    # vocab 和 merges 在训练过程中同步维护, 保证 tie-break 能按 bytes 比较, 并直接得到测试要求的 bytes merges
    return vocab,merge_dic
