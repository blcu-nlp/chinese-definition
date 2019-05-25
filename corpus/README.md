# 数据处理过程

## 1. count_words.sh

读取CCD数据，输出CCD词表

输入：
- ./xls
> CCD数据文件夹

输出：
- ./processed/ccd_word_list.txt
> CCD词表

## 2. extract_sememes.sh

从知网中抽取词-义原对，便于后续处理

输入：
- utils/HowNet.txt
> HowNet文件

输出：
- processed/word_sememes.txt
> HowNet中的词-义原对

## 3. extract_same_words.sh

抽取知网和CCD中共有的词

输入：
- ./processed/ccd_word_list.txt
> CCD词表
- ./processed/hownet_word_list.txt
> HowNet词表，使用脚本中的命令获得

输出：
- ./processed/same_words.txt
> CCD和HowNet中共有的词

## 4. sort_by_giga_freq.sh

在same_words文件中，删除包含特定义原的词，并对所得词表按照gigaword语料库中的词频进行排序。

义原列表在`preprocess/数据预处理过程（基于义原的概念释义自动生成）.md`中。

排序时，在gigaword语料库中的词按词频排在前，不在gigaword语料库中的OOV词排在后。

输入：
- ./utils/gigawords_freq_sun.txt
> gigaword语料库词频表

- ./processed/same_words.txt
> CCD和HowNet中共有的词

- ./processed/word_sememes.txt
> 从HowNet中抽取的词-义原对

输出：
- ./processed/sorted_words.txt
> 筛选、排序之后的词表

## 5. extract_definitions.sh

从CCD中，抽取`sorted_words.txt`中的词对应的释义

输入：
- ./xls
> CCD数据保存文件夹

- ./processed/sorted_words.txt
> 筛选、排序之后的词表

输出：
- ./processed/word_definitions.txt
> 从CCD中抽取的词语-释义对

## 6. wsd.sh

将词语、义原、释义进行对应

输入：
- ./processed/word_definitions.txt
> 词语-释义对

- ./processed/word_sememes.txt
> 词语-义原对

输出：
- ./processed/wsd_on_hownet.txt
> 每行都是词-义原-释义

## 7. make_final_dataset.sh

将`word_definitions.txt`（即所有的词语-释义对），与`wsd_on_hownet.txt`（即词语-义原-释义）合并，没有义原的词以自身作为义原

输入：
- ./processed/word_definitions.txt
> 词语-释义对

- ./processed/wsd_on_hownet.txt
> 词语-义原-释义三元组

输出：
- ./processed/dataset.txt
> 合并之后输出的数据及文件

## 8. del_oov.sh

将被解释词不在gigaword词向量词表（jieba分词）中的条目删除

输入：
- ./utils/vocab_jieba.txt
> jieba分词的gigaword语料库词向量词表

- ./processed/dataset.txt
> 待删OOV的数据集文件

输出：
- ./processed/dataset.txt.new
> 删除OOV之后的数据集文件

## 9. cut_and_split.py

将数据集按照18:1:1进行切分，并对释义使用jieba工具进行分词。将所得train/valid/test数据写入文件，并获得用于测试的`shortlist_test.txt`（只包含词语和义原，不包含释义）

输入：
- ./processed/dataset.txt.new
> 待切分的数据集

- ./processed/results/
> 分词、切分后的数据集保存位置
