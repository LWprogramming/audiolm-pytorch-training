import re
import matplotlib.pyplot as plt
import os


filename = os.path.expanduser("~/Downloads/output-264330.log")
loss_pattern = re.compile(r"(\d+): loss: (\d+\.\d+)")
valid_loss_pattern = re.compile(r"(\d+): valid loss (\d+\.\d+)")

def load_data(filename):
    with open(filename, "r") as f:
        content = f.readlines()

    loss_data = {"semantic": [], "coarse": [], "fine": []}
    valid_loss_data = {"semantic": [], "coarse": [], "fine": []}

    for line in content:
        loss_match = loss_pattern.search(line)
        valid_loss_match = valid_loss_pattern.search(line)

        if loss_match:
            step, loss = int(loss_match.group(1)), float(loss_match.group(2))
            if len(loss_data["semantic"]) < 1000001:
                loss_data["semantic"].append((step, loss))
            elif len(loss_data["coarse"]) < 1000001:
                loss_data["coarse"].append((step, loss))
            else:
                loss_data["fine"].append((step, loss))

        elif valid_loss_match:
            step, valid_loss = int(valid_loss_match.group(1)), float(valid_loss_match.group(2))
            if len(valid_loss_data["semantic"]) < 11:
                valid_loss_data["semantic"].append((step, valid_loss))
            elif len(valid_loss_data["coarse"]) < 11:
                valid_loss_data["coarse"].append((step, valid_loss))
            else:
                valid_loss_data["fine"].append((step, valid_loss))
    assert len(loss_data["semantic"]) == 1000001, f"Loss data for `semantic` key should have exactly 1000001 data points but instead got {len(loss_data['semantic'])}"
    assert len(loss_data["coarse"]) == 1000001, f"Loss data for `coarse` key should have exactly 1000001 data points but instead got {len(loss_data['coarse'])}"
    assert len(loss_data["fine"]) == 1000001, f"Loss data for `fine` key should have exactly 1000001 data points but instead got {len(loss_data['fine'])}"
    assert len(valid_loss_data["semantic"]) == 11, f"Valid loss data for `semantic` key should have exactly 11 data points but instead got {len(valid_loss_data['semantic'])}"
    assert len(valid_loss_data["coarse"]) == 11, f"Valid loss data for `coarse` key should have exactly 11 data points but instead got {len(valid_loss_data['coarse'])}"
    assert len(valid_loss_data["fine"]) == 11, f"Valid loss data for `fine` key should have exactly 11 data points but instead got {len(valid_loss_data['fine'])}"
    return (loss_data, valid_loss_data)

def plot_loss_section(loss_data_tuple, phase, start_train_step, end_train_step):
    loss_data, valid_loss_data = loss_data_tuple
    loss_section = [(step, loss) for step, loss in loss_data[phase] if start_train_step <= step <= end_train_step]
    valid_loss_section = [(step, valid_loss) for step, valid_loss in valid_loss_data[phase] if start_train_step <= step <= end_train_step]

    plt.figure()
    plt.plot(*zip(*loss_section), label="Loss")
    plt.plot(*zip(*valid_loss_section), label="Valid Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"{phase.capitalize()} Phase Loss ({start_train_step}-{end_train_step})")
    plt.legend()
    plt.show()

loss_data_tuple = load_data(filename)
# show the first 100k
plot_loss_section(loss_data_tuple, "semantic", 0, 100000)
plot_loss_section(loss_data_tuple, "coarse", 0, 100000)
plot_loss_section(loss_data_tuple, "fine", 0, 100000)
# show the remainder after things have stabilized in theory
plot_loss_section(loss_data_tuple, "semantic", 100000, 1000000)
plot_loss_section(loss_data_tuple, "coarse", 100000, 1000000)
plot_loss_section(loss_data_tuple, "fine", 100000, 1000000)
