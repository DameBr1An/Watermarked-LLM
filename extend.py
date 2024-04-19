import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练的 GPT-2 模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("D:\\DDA4210\\gpt")
model = GPT2LMHeadModel.from_pretrained("D:\\DDA4210\\gpt")

# 定义函数来计算困惑度并替换单词
def calculate_perplexity_and_replace(text):
    # 分词
    tokenized_text = tokenizer.tokenize(text)
    # 在句子开头添加起始符号
    tokenized_text = ['<|endoftext|>'] + tokenized_text
    # 将单词转换为索引
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    # 转换为张量
    input_ids = torch.tensor([input_ids])

    # 使用模型预测下一个单词的概率分布
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    # 计算困惑度
    perplexities = []
    for i, word in enumerate(tokenized_text):
        word_logits = logits[0, i]  # 获取当前单词的概率分布
        word_index = tokenizer.convert_tokens_to_ids(word)
        word_probability = torch.softmax(word_logits, dim=-1)[word_index].item()  # 获取当前单词的概率
        perplexity = 1 / word_probability  # 计算困惑度
        perplexities.append(perplexity)

    # 找出困惑度过高的单词
    high_perplexity_words = [tokenized_text[i] for i, perplexity in enumerate(perplexities) if perplexity > 1]

    # 对困惑度过高的单词进行替换
    for word in high_perplexity_words:
        # 替换成其他单词或者通过某种方法重新生成
        pass

    return tokenized_text

# 示例文本
text = "The quick brown fox jumps over the lazy dog."
# 计算困惑度并替换单词
processed_text = calculate_perplexity_and_replace(text)
print(processed_text)


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

# 查找单词的同义词
synonyms = []
for syn in wordnet.synsets("happy"):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print("Synonyms:", synonyms)

# 查找单词的反义词
antonyms = []
for syn in wordnet.synsets("happy"):
    for lemma in syn.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
print("Antonyms:", antonyms)
