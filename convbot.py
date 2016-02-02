import pandas as pd
from io import StringIO
import nltk

min_conv_len = 7

with open('data/dialogs/movie_conversations.txt', 'r', encoding='ISO-8859-1') as file:
    conversation_data = file.read()

with open('data/dialogs/movie_lines.txt', 'r', encoding='ISO-8859-1') as file:
    line_data = file.read()

conversations = pd.read_table(StringIO(conversation_data), delimiter='\+\+\+\$\+\+\+', engine='python')
lines = pd.read_table(StringIO(line_data), delimiter='\+\+\+\$\+\+\+', engine='python')

line_nums = lines.ix[:, 0].tolist()
line_nums = [num.replace(' ', '') for num in line_nums]
lines = lines.ix[:, 4].tolist()
lines = [str(line)[1:] for line in lines]
conversations = conversations.ix[:, 3].tolist()
conversations = [conv.replace(' ', '') for conv in conversations]
conversations = [conv.replace("'", '') for conv in conversations]
conversations = [conv[1:-1].split(',') for conv in conversations]

line_dict = {}
for line_num, line in zip(line_nums, lines):
    line_dict[line_num] = line

full_conversations = [None] * len(conversations)
for i, conv in enumerate(conversations):
    conv_lines = []
    for num in conv:
        conv_lines.append(' '.join(line_dict[num].split()))
    full_conversations[i] = conv_lines

long_conversations = []
for conv in full_conversations:
    if len(conv) >= min_conv_len:
        long_conversations.append(conv)

print(long_conversations[0])
print(len(long_conversations))

token_conversations = []
for conv in long_conversations:
    token_conv = []
    for line in conv:
        token_conv.append(nltk.word_tokenize(line))
    token_conversations.append(token_conv)

print(token_conversations[0])

std_conversations = []
for conv in token_conversations:
    std_conv = []
    for line in conv:
        line.insert(0, '<START>')
        line.append('<END>')
        std_conv.append(line)
    std_conversations.append(std_conv)

max_len = max([len(line) for conv in std_conversations for line in conv])
print(max_len)

padded_conversations = []
for conv in std_conversations:
    padded_conv = []
    for line in conv:
        for i in range(max_len - len(line)):
            line.append('<PAD>')
        padded_conv.append(line)
    padded_conversations.append(padded_conv)

print(padded_conversations[0])
print(len(padded_conversations[0][0]))
pad_line = []
for i in range(max_len):
    pad_line.append('<PAD>')

examples = []
for conv in padded_conversations:
    for i in range(len(conv) - min_conv_len):
        pass
