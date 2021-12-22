from jiayan import load_lm
from jiayan import CRFSentencizer

text = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'

lm = load_lm('jiayan.klm')
sentencizer = CRFSentencizer(lm)
sentencizer.load('cut_model')
print(sentencizer.sentencize(text))