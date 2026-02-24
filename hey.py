if __name__ == '__main__':
    lis = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        record = [name, score]
        lis.append(record)

    scores = []
    for i in range(len(lis)):
        scores.append(lis[i][1])

    scores.sort()
    minimum= scores.index(min(scores))

    print(scores)

    for i in scores:
        if i > scores[minimum]:
            print(i)
            slg= i
            break

    name = []
    for i in range(len(lis)):
        if lis[i][1] == slg:
            name.append(lis[i][0])

    name.sort()
    for n in name:
        print(n)