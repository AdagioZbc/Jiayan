import thulac
import jsonlines
import copy

########### Initialize tokenizer ################
tokenizer = thulac.thulac(seg_only=True)

train_dir = "CCPM_data/train.jsonl"
valid_dir = "CCPM_data/valid.jsonl"
test_public = "CCPM_data/test_public.jsonl"

# text = '是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。'
train_data_list = []
with open(train_dir, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        train_data_list.append(item)


def mutate_choices(old_choices, ans_id, pair_ans):
    # print("OLD:", old_choices)
    sorted_ans = sorted(pair_ans, key=lambda x:len(x[0]), reverse=True) # from the longest to the shortest
    # print(sorted_ans)
    for s in sorted_ans:
        # print(s)
        new_choices = copy.deepcopy(old_choices)
        mutate_token = s[0]
        mutate_pos = s[1]
        mutate_len = len(s[0])
        flag = True
        for i in range(4):
            if i == ans_id:
                continue # don't mutate the right answer
            else:
                new_sentence = new_choices[i][:mutate_pos] + mutate_token + new_choices[i][mutate_pos + mutate_len:]
                new_choices[i] = new_sentence
                if new_choices[i] == old_choices[ans_id]:
                    flag = False
                    break
        if not flag:  # flag = false, find the next
            continue
        if new_choices == old_choices:
            continue
        break

    if flag: # cannot find a mutation token that won't cause misprediction
        return new_choices
    else:
        return old_choices


new_train_data_list = []
for data in train_data_list:
    trans = data['translation']
    choices = data['choices']
    ans_id = data['answer']
    ans = choices[ans_id]
    tokenized_ans = tokenizer.cut(ans, text=True).split()
    if "，" in tokenized_ans:
        tokenized_ans.remove("，")
    if "。" in tokenized_ans:
        tokenized_ans.remove("。")
    tokenized_pair_ans = []
    cnt = 0
    for token in tokenized_ans:
        tokenized_pair_ans.append([token, cnt])  # save the token's start position
        cnt += len(token)
    new_choices = mutate_choices(choices, ans_id, tokenized_pair_ans)
    # print(new_choices)
    # exit()
    new_train_data_list.append({"translation": trans, "choices": new_choices, "answer": ans_id})

############## Write to jsonl file ################
with jsonlines.open("./new_train_thulac.jsonl", "w") as w:
    for line in new_train_data_list:
        w.write(line)
