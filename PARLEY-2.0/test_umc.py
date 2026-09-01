import urc_synthesis
import prism_model_generator

def main():
    maps = [14,21,23]
    for i in maps:
        infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
        outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
        prism_model_generator.generate_model(i)
        urc_synthesis.manipulate_prism_model(infile, outfile, baseline=False)


if __name__ == '__main__':
    main()