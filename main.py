from Experiments.scarlet import exp_scarlet
from Experiments.guybresler import exp_guybresler

if __name__ == "__main__":
  import argparse

#   parser = argparse.ArgumentParser(description='Process some integers.')
#   parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                       help='an integer for the accumulator')

#   args = parser.parse_args()
  ## Uncomment experiments and run.
  
  num_nodes = [10,20,30]     # Number of nodes.
  num_samples = [200,500,1000]
  zeta = 0.05 # Confidence bound.

  # Scarlet
  # exp_scarlet(args.integers[0],args.integers[1],args.integers[2],args.integers[3])
  # exp_scarlet(num_nodes, num_samples)

  # Guy Bresler
  alpha = 0.1
  beta = 0.5
  exp_guybresler(num_nodes, num_samples, zeta, alpha, beta)
