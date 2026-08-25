import urc_synthesis
import prism_model_generator
import prism_model_generator_belief_full
import urc_synthesis_belief_full
import plot_fronts

def main():
    # for i in range(10,100):
    #     infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    #     outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
    #     prism_model_generator.generate_model(i)
    #     print(f"generated: {infile}")
    #     urc_synthesis.manipulate_prism_model(infile, outfile, baseline=False)
    #     print(f"generated: {outfile}")

    maps = [57]
    for i in maps:
        infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
        outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
        prism_model_generator_belief_full.generate_model(i)
        print(f"generated: {infile}")
        urc_synthesis_belief_full.manipulate_prism_model(infile, outfile, baseline=False)
    print(f"generated: {outfile}")

    # max_replications = 10
    # for period in range(max_replications):
    #     plot_fronts.plot_pareto_front(i, period)

if __name__ == '__main__':
    main()