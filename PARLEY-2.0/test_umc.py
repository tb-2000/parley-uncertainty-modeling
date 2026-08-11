import urc_synthesis
import prism_model_generator

def main():
    for i in range(10,100):
        infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
        outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
        prism_model_generator.generate_model(i)
        print(f"generated: {infile}")
        urc_synthesis.manipulate_prism_model(infile, outfile, baseline=False)
        print(f"generated: {outfile}")


if __name__ == '__main__':
    main()