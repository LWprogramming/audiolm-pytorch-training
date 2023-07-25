import re
import matplotlib.pyplot as plt
import os

# sample usage: python data_analysis.py output-7789.log semantic

def load_data(filename, loss_pattern, valid_loss_pattern):
    with open(filename, "r") as f:
        content = f.readlines()

    loss_data = []
    valid_loss_data = []
    for line in content:
        loss_match = loss_pattern.search(line)
        valid_loss_match = valid_loss_pattern.search(line)

        if loss_match:
            step, loss = int(loss_match.group(1)), float(loss_match.group(2))
            loss_data.append((step, loss))
        elif valid_loss_match:
            step, valid_loss = int(valid_loss_match.group(1)), float(valid_loss_match.group(2))
            valid_loss_data.append((step, valid_loss))
    print(f"loss data and valid loss data loaded from {filename} and have lengths {len(loss_data)} and {len(valid_loss_data)} respectively")
    return loss_data, valid_loss_data


def plot_loss(loss_data, valid_loss_data, transformer_type, log_filename, skip_first_n=0):
    # given loss_data and valid_loss_data for a particular transformer, plot them on the same graph and title with transformer_type and log filename it came from
    # loss_data and valid_loss_data are lists of tuples (step, loss)
    # skip_first_n is the number of data points to skip at the beginning of the graph because loss is naturally high at the beginning
    loss_data = loss_data[skip_first_n:]
    valid_loss_data = valid_loss_data[skip_first_n:]
    plt.figure()
    plt.plot(*zip(*loss_data), label="train loss")
    plt.plot(*zip(*valid_loss_data), label="valid loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"{transformer_type} loss from {log_filename}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to log file") # something like output-7789.log
    parser.add_argument("transformer_type", help="Type of transformer used", choices=["semantic", "coarse", "fine"])
    parser.add_argument("--skip_first_n", help="Number of data points to skip at the beginning of the graph", type=int, default=0)
    args = parser.parse_args()

    loss_pattern = re.compile(fr".*{args.transformer_type} (\d+): loss: (\d+\.\d+)")
    valid_loss_pattern = re.compile(fr".*{args.transformer_type} (\d+): valid loss (\d+\.\d+)")

    loss_data, valid_loss_data = load_data(args.filename, loss_pattern, valid_loss_pattern)
    plot_loss(loss_data, valid_loss_data, args.transformer_type, args.filename, args.skip_first_n)
