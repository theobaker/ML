"""
3-way quicksort alogorithm in python
Theo Baker
20210915 CSC630
"""
import random
def partition3(a, l, r):
    lessthan = l
    i = l
    greaterthan = r
    pivot = a[l]

    while i <= greaterthan:
        if a[i] < pivot:
            a[lessthan],a[i] = a[i], a[lessthan]
            lessthan += 1
            i += 1
        elif a[i] > pivot:
            a[i], a[greaterthan] = a[greaterthan], a[i]
            greaterthan -= 1
        else:
            i += 1

    return lessthan, greaterthan

def quick_sort(a, l, r):
    if l >= r:
        return
    k = random.randint(l, r)
    a[k], a[l] = a[l], a[k]

    lessthan, greaterthan = partition3(a, l, r)
    quick_sort(a, l, lessthan - 1)
    quick_sort(a, greaterthan + 1, r)


def printarr(a, n):
    for i in range(n):
        print(a[i], end=" ")
    print()

a = [4, 9, 4, 4, 1, 9, 4, 4, 9, 4, 4, 1, 4]
size = len(a)
printarr(a, size)
quick_sort(a, 0, size - 1)
printarr(a, size)
