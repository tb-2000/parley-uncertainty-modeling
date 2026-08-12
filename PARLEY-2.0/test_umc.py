import prism_model_generator_interval
import urc_synthesis_interval
import prism_model_generator_interval_per_map
import urc_synthesis_interval_per_map

def main():
   
    # for i in range(10 ,100):
    #     infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    #     outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
    #     prism_model_generator_interval.generate_model(i)
    #     urc_synthesis_interval.manipulate_prism_model(infile, outfile, baseline=False)
    i=10
    infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
    prism_model_generator_interval_per_map.generate_model(i)
    urc_synthesis_interval_per_map.manipulate_prism_model(infile, outfile, baseline=False)


if __name__ == '__main__':
    main()