from jiayan import load_lm
from jiayan import CharHMMTokenizer
import jsonlines
import copy


train_dir = "CCPM_data/train.jsonl"
valid_dir = "CCPM_data/valid.jsonl"
test_public = "CCPM_data/test_public.jsonl"

# text = '是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。'
train_data_list = []
with open(train_dir, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        train_data_list.append(item)
print(len(train_data_list))

########## initialize jiayan tokenizer ##########
lm = load_lm('jiayan.klm')
tokenizer = CharHMMTokenizer(lm)


def mutate_choices(old_choices, ans_id, pair_ans):
    sorted_ans = sort(pair_ans, lambda x:x[0]) # from the longest to the shortest
    for s in sorted_ans:
        new_choices = copy.deepcopy(old_choices)
        mutate_token = s[0]
        mutate_pos = s[1]
        mutate_len = len(s[0])
        flag = True
        for i in range(4):
            if i == ans_id:
                continue # don't mutate the right answer
            else:
                new_choices[i][mutate_pos:mutate_pos + mutate_len] = mutate_token
                if new_choices[i] == old_choices[ans_id]:
                    flag = False
                    break
        if not flag:  # flag = false, find the next
            continue

    if flag: # cannot find a mutation token that won't cause misprediction
        return new_choices
    else:
        return old_choices


for data in train_data_list:
    trans = data['translation']
    choices = data['choices']
    ans_id = data['answer']
    ans = choices[ans_id]
    print(trans, ans)
    tokenized_ans = list(tokenizer.tokenize(ans))
    tokenized_pair_ans = []
    cnt = 0
    for token in tokenized_ans:
        tokenized_pair_ans.append(token, cnt)  # save the token's start position
        cnt += len(token)
    mutate_choices(choices, ans_id, tokenized_pair_ans)
    # res = max(tokenized_ans, key=len, default='')
    # print(tokenized_ans)
    # print(res)
    # exit()

# print(list(tokenizer.tokenize(text)))

