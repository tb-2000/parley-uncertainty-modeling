import numpy as np
import random

def main():

    selected_maps = [16, 20, 21, 22, 23, 29, 30, 31, 32, 33, 36, 37,
                     40, 43, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55,
                     56, 57, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73,
                     75, 76, 79, 81, 82, 83, 85, 86, 87, 89, 90, 97]
    auswahl = sorted(random.sample(selected_maps, 29))

    print(auswahl)
    print(auswahl[0])

if __name__ == '__main__':
    main()