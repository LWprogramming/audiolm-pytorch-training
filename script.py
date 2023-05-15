import matplotlib.pyplot as plt

loss = []

with open('output.log', 'r') as f:
    for line in f:
        if 'loss:' in line:
            loss_value = float(line.split(':')[-1])
            loss.append(loss_value)

# remove the first few lines that don't contain loss values
while loss and loss[0] > 10:
    loss.pop(0)

# plot loss against steps
plt.plot(range(len(loss)), loss)
plt.xlabel('steps')
plt.ylabel('loss')
plt.show()
