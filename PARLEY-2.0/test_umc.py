import prism_model_generator_interval_per_map
import urc_synthesis_interval_per_map
from RQ3_interval_scaling import update_input, generate_base_models, calculate_thresholds, synthesize_models, SCALING_MAPS, MODELS_DIR, THRESHOLD_OUTPUT_DIR, THRESHOLD_FILE

def main():
   
    # for i in range(10 ,100):
    #     infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    #     outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
    #     prism_model_generator_interval.generate_model(i)
    #     urc_synthesis_interval.manipulate_prism_model(infile, outfile, baseline=False)
    # maps = [21,23,30]
    # for i in maps:
    #     infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    #     outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
    #     prism_model_generator_interval_per_map.generate_model(i)
    #     urc_synthesis_interval_per_map.manipulate_prism_model(infile, outfile, baseline=False)

    # 1. Generate the base interval models with size-dependent targets.
    generate_base_models()

    # 2. Determine map-specific interval thresholds from steps 1..20.
    calculate_thresholds()

    # 3. Build the URC/UMC models using those thresholds.
    synthesize_models()


if __name__ == '__main__':
    main()