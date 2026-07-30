import numpy as np
import random

def main():

    maps = range(10, 100)
    auswahl = sorted(random.sample(maps, 50))

    print(auswahl)
    print(auswahl[0])

if __name__ == '__main__':
    main()