from jiayan import load_lm
from jiayan import CharHMMTokenizer

text = '是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。'

lm = load_lm('jiayan.klm')
tokenizer = CharHMMTokenizer(lm)
print(list(tokenizer.tokenize(text)))