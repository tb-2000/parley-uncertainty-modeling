import urc_synthesis

def main():
    i = 0
    infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
    urc_synthesis.manipulate_prism_model(infile, outfile, baseline=False)


if __name__ == '__main__':
    main()