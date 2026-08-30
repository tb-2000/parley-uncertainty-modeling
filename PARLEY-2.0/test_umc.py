import urc_synthesis_hmm
import prism_model_generator_hmm

def main():
    maps = [10]
    for i in maps:
        infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
        outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
        prism_model_generator_hmm.generate_model(i)
        urc_synthesis_hmm.manipulate_prism_model(infile, outfile, baseline=False)
        


if __name__ == '__main__':
    main()