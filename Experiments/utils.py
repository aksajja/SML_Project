import matplotlib.pyplot as plt

def plot_error_vs_samples(exp_name,error,samples,num_nodes):
  import matplotlib.pyplot as plt
  fig = plt.figure(dpi=100, figsize=(14, 7))
  rows = int(len(samples)**0.5)
  ax = fig.add_subplot(111)
  ax.plot(samples, error)

  ax.set(xlabel='Samples', ylabel='Error',
        title=f'{exp_name} Error vs Samples - {num_nodes} nodes')
  ax.grid()
  plt.show()
  plt.savefig(f'../results/{exp_name}_Error_vs_Samples_{num_nodes}.png')
  print(f'Saved plot for {num_nodes} nodes in ../results/{exp_name}_Error_vs_Samples_{num_nodes}.png')
