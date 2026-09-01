import urc_synthesis_hmm_behavioral_structured
import prism_model_generator_hmm_behavioral_structured

def main():
    maps = [14, 21, 23] # selected_maps
    for i in maps:
        infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
        outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
        prism_model_generator_hmm_behavioral_structured.generate_model(i)
        urc_synthesis_hmm_behavioral_structured.manipulate_prism_model(infile, outfile, baseline=False)
        


if __name__ == '__main__':
    main()