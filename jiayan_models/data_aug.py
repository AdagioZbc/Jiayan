from jiayan import load_lm
from jiayan import CharHMMTokenizer
import jsonlines
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

for data in train_data_list:
    trans = data['translation']
    choices = data['choices']
    ans_id = data['answer']
    ans = choices[ans_id]
    print(trans, ans)
    tokenized_ans = tokenizer.tokenize(ans)
    res = max(tokenized_ans, key=len, default='')
    print(res)
    exit()

# print(list(tokenizer.tokenize(text)))

