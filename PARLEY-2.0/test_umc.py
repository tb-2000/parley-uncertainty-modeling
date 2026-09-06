import urc_synthesis_hmm_exact_local
import prism_model_generator_hmm_exact_local

def main():
    maps = [14, 21] # selected_maps
    for i in maps:
        infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
        outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
        prism_model_generator_hmm_exact_local.generate_model(i)
        urc_synthesis_hmm_exact_local.manipulate_prism_model(infile, outfile, baseline=False)
        


if __name__ == '__main__':
    main()