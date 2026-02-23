#!/bin/python3


n = 24

if __name__ == '__main__':
    n = int(input().strip())
    if n > 0 & n < 100:
        if n // 2 == 0:
            if n in range(2, 6):
                print("Not Weird")
            elif n in range(6, 21):
                print("Weird")
            elif n > 20:
                print("Weird")

        else:
            print('Weird')
    else:
        print("Invalid Number")

