import pandas as pd
import csv

with open('msr_paraphrase_train.txt') as f:
    reader = csv.reader(f, delimiter="\t")
    data = [row for row in reader]

label, s1, s2 = [], [], []
for i,row in enumerate(data[1:]):
    if len(row) != 5:
        continue
    label.append(int(row[0]))
    s1.append(row[3])
    s2.append(row[4])

df=pd.DataFrame({'label':label, 'sentence1':s1, 'sentence2':s2})
df.to_csv("mrpc_train.csv", index=False)



with open('msr_paraphrase_test.txt') as f:
    reader = csv.reader(f, delimiter="\t")
    data = [row for row in reader]

label, s1, s2 = [], [], []
for i,row in enumerate(data[1:]):
    if len(row) != 5:
        continue
    label.append(int(row[0]))
    s1.append(row[3])
    s2.append(row[4])

df=pd.DataFrame({'label':label, 'sentence1':s1, 'sentence2':s2})
df.to_csv("mrpc_test.csv", index=False)
