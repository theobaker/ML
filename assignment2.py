def check(i):
    stringversion = str(i)
    halflen = len(stringversion)/2
    str1 = stringversion[0:halflen]
    str2 = stringversion[halflen:]
    for s in range(halflen):
        if ((int)(str1[s]) != (int)(str2[s])):
            return false
    return true


answers = []
sum = 0

for i in range(100000, 0, -1):
    if (check(i)):
        binarynum = bin(i).replace("0b","")
        if (check(binarynum)):
            answers.append(i)

for m in range(answers):
    sum += answers[m]

print(sum)
