import random
from itertools import count


def main():
    n = 1000000
    rnd_items = [random.uniform(-999, 999) for _ in range(n)]
    print(
        sum(1 for i in rnd_items if i < 0)
    )
    print(
        sum(1 for i in rnd_items if i == 0)
    )
    print(
        sum(1 for i in rnd_items if i > 0)
    )


if __name__ == '__main__':
    main()
