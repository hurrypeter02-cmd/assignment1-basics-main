from tokenizer import init_text,get_stats,merge,init_heap,push,pop
from pretokenization_example import find_chunk_boundaries

import regex as re
from collections import Counter
import multiprocessing
import argparse



def task(text,special_tokens):
    text_list = init_text(text,special_tokens)
    unique_token_list = Counter(tuple(text.encode("utf-8")) for text in text_list)
    return unique_token_list




def run_train_bpe_parallel(input_path,vocab_size,special_tokens):
    with open(input_path,'rb') as f:
        cpu_num = multiprocessing.cpu_count()
        num_processes = cpu_num
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        text_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            text_list.append(f.read(end - start).decode("utf-8", errors="ignore"))
        # 拆分成多 chunk ,方便后续分配
    with multiprocessing.Pool(cpu_num) as pool:
        results = [] 
        # 循环提交所有异步任务
        for i in range(len(text_list)):
            async_res = pool.apply_async(task, args=(text_list[i],special_tokens))
            results.append(async_res)
        unique_token_list = Counter()
        for counts in results:
            unique_token_list += counts.get()
        # 聚集 每个子进程 得到的结果

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

def main():
    parser = argparse.ArgumentParser(description = "并行 BPE 分词器训练脚本")
    parser.add_argument("--input_path",type=str,required=True,help="数据地址")
    parser.add_argument("--vocab_size",type=int,required=True,help="词汇表大小")
    parser.add_argument("--special_tokens",action="append",equired=True,help="特殊的字符串")
    args = parser.parse_args()
    run_train_bpe_parallel(args.input_path,args.vocab_size,args.special_tokens)

if __name__ == "_main_":
    main()