import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import random


def estimate_price(mileage, intercept, slope):
    """Estimate the price of a car based on the provided mileage with the current model."""
    return intercept + mileage * slope


def estimate_price_nav(intercept, slope):
    """Prompt for mileage and estimate price on valid input."""
    line = ""
    print()
    while (42):
        try:
            line = input("Please give a mileage (in km) to estimate the price for: ")
            mileage = float(line)
            print(f"\nPrice estimation for {mileage} km: {estimate_price(mileage, intercept, slope):.2f}€")
            input("Press enter to continue...")
            return
        except ValueError:
            if (line != ""):
                print(f"'{line}' is not a valid float value.")
                line = ""


def train_once(values, intercept, slope, lr):
    """Do one iteration of training."""
    delta_intercept = lr * sum([estimate_price(n[0], intercept, slope) - n[1] for n in values]) / len(values)
    delta_slope = lr * sum([(estimate_price(n[0], intercept, slope) - n[1]) * n[0] for n in values]) / len(values)
    return delta_intercept, delta_slope


def train(csv, intercept, slope, lr, epochs):
    """Normalize data, do the requested iterations of training and return the new theta values."""
    kms = csv.iloc[:, 0]
    xmax = max(kms)

    prices = csv.iloc[:, 1]
    ymax = max(prices)

    intercept /= ymax
    slope *= ymax / xmax
    values = list(zip(kms / xmax, prices / ymax))

    for n in range(epochs):
        delta_intercept, delta_slope = train_once(values, intercept, slope, lr)
        intercept -= delta_intercept
        slope -= delta_slope
    return intercept * ymax, (slope * ymax) / xmax


def load_training_data(path):
    """Load csv data from path."""
    csv = pd.read_csv(path)
    return csv


def train_nav(csv, intercept, slope, lr):
    """Prompt for number of iterations and run training on valid input."""
    line = ""
    print()
    while (42):
        try:
            line = input("Please enter number of iterations: ")
            epochs = int(line)
            assert epochs > 0, f"{epochs} is not a valid positive integer value."
            assert epochs <= 1000000, "Max number of iterations: 1000000."
            test = train(csv, intercept, slope, lr, epochs)
            intercept, slope = test
            print(f"\nNew values after training: th0: {intercept:.4f}, th1: {slope:.4f}.")
            input("Press enter to continue...")
            return intercept, slope
        except ValueError:
            if (line != ""):
                print(f"'{line}' is not a valid positive integer value.")
                line = ""
        except AssertionError as msg:
            print(msg)


def plot_data(csv, intercept, slope):
    """Do scatterplot of original data and lineplot of the current model in one graph."""
    csv.plot.scatter(x='km', y='price')
    xmax = max(csv.iloc[:, 0])
    x_values = [0, xmax]
    y_values = [intercept, (slope*xmax + intercept)]
    plt.plot(x_values, y_values, color="red", )
    plt.xlabel = "price [€]"
    plt.ylabel = "mileage [km]"
    plt.show()


def change_lr(lr):
    """Prompt for new learning rate value."""
    line = ""
    print()
    while (42):
        try:
            line = input(f"Please enter a new value for the learning rate (currently {lr}): ")
            new_lr = float(line)
            assert 0.001 <= new_lr <= 1.0, "Please enter a value between 0.001 and 1.0"
            return new_lr
        except AssertionError as msg:
            print(msg)
        except ValueError:
            if (line != ""):
                print(f"'{line}' is not a valid float value.")
                line = ""


def list_commands():
    """List all valid commands."""
    print("This is the list of available commands:\n")
    print("help\t\t: List all available commands")
    print("change\t\t: Change the learning rate")
    print("estimate\t: Estimate the price of a car")
    print("train\t\t: Train the model")
    print("error\t\t: Calculate the error of the current model")
    print("plot\t\t: Generate plots")
    print("create\t\t: Create a new dataset (note: this will reset training data)")
    print("exit\t\t: Exit the program")
    input("Press enter to continue...")


def calc_error(csv, intercept, slope):
    """Calculate and display the Root Relative Squared Error (RRSE) of the current model."""
    mean = sum([n[1] for n in csv.values]) / len(csv.values)
    print(mean)
    square_error = sum([(n[1] - estimate_price(n[0], intercept, slope))**2 for n in csv.values])
    square_mean_distance = sum([(n[1] - mean)**2 for n in csv.values])
    RRSE = math.sqrt((square_error / square_mean_distance))
    print(f"\nRoot Relative Squared Error: {RRSE*100:.2f}%")
    input("Press enter to continue...")


def get_new_price():
    """Prompts for and returns the price for a car with zero mileage for the new dataset."""
    line = ""
    while line == "":
        try:
            line = input("Enter price of new car in €: ")
            new_price = float(line)
            assert new_price > 0, "Please enter a positive float value."
        except ValueError:
            print(f"'{line}' is not a valid float.")
            line = ""
        except AssertionError as msg:
            print(msg)
            line = ""
    return new_price


def get_zero_mileage():
    """Prompts for and returns the mileage at which the price of a car reaches zero for the new dataset."""
    line = ""
    while line == "":
        try:
            line = input("Enter the mileage (in km) at which the price reaches 0: ")
            zero_mileage = float(line)
            assert zero_mileage > 0, "Please enter a positive float value."
        except ValueError:
            print(f"'{line}' is not a valid float.")
            line = ""
        except AssertionError as msg:
            print(msg)
            line = ""
    return zero_mileage


def get_n_values():
    """Prompts for and returns the desired number of values for the new dataset."""
    line = ""
    while line == "":
        try:
            line = input("Enter number of values: ")
            n_values = int(line)
            assert n_values > 1, "Please enter an integer larger than 1."
        except ValueError:
            print(f"'{line}' is not a valid integer.")
            line = ""
        except AssertionError as msg:
            print(msg)
            line = ""
    return n_values


def get_spread():
    """Prompts for and returns the spread for the new dataset."""
    line = ""
    while line == "":
        try:
            line = input("Enter spread (between 0 and 1): ")
            spread = float(line)
            assert 0 <= spread <= 1, "Please enter a value between 0 and 1."
        except ValueError:
            print(f"'{line}' is not a valid float.")
            line = ""
        except AssertionError as msg:
            print(msg)
            line = ""
    return spread


def create_dataset():
    """Create a new custom dataset"""
    new_price = get_new_price()
    zero_mileage = get_zero_mileage()
    n_values = get_n_values()
    spread = get_spread()
    x_values = [(n * zero_mileage) / n_values for n in range(n_values)]
    y_values = [new_price * ((zero_mileage - n) / zero_mileage) for n in x_values]
    y_values = [n + random.uniform(-spread, spread) * new_price for n in y_values]
    df = pd.DataFrame({"km": x_values, "price": y_values})
    input("Press enter to continue...")
    return df


def main():
    try:
        assert len(sys.argv) < 3, "More than one argument provided"
        path = ""
        if len(sys.argv) == 1:
            while path == "":
                path = input("Enter path to csv file: ")
        else:
            path = sys.argv[1]
        csv = load_training_data(path)
        intercept = 0
        slope = 0
        lr = 0.1
        print()
        print('#'*80)
        print("\nWelcome to ft_linear_regression - an introductory project to Machine Learning.\n")
        print('#'*80)
        while True:
            print(f"\nLearning rate: {lr} th0: {intercept:.4f} th1: {slope:.4f}\n")
            command = input("Enter 'help' to list all available commands.\nPlease enter a command: ")
            if command == "help":
                list_commands()
            elif command == "estimate":
                estimate_price_nav(intercept, slope)
            elif command == "train":
                intercept, slope = train_nav(csv, intercept, slope, lr)
            elif command == "plot":
                plot_data(csv, intercept, slope)
            elif command == "change":
                lr = change_lr(lr)
            elif command == "error":
                calc_error(csv, intercept, slope)
            elif command == "create":
                csv = create_dataset()
                intercept = 0
                slope = 0
                lr = 0.1
            elif command == "exit":
                sys.exit(0)
            else:
                print(f"Unknown command: '{command}'")
    except AssertionError as msg:
        print(f"AssertionError: {msg}")
    except KeyboardInterrupt:
        sys.exit()
    except EOFError:
        print()
        sys.exit()
    except FileNotFoundError:
        print(f"FileNotFoundError: '{path}' is not a valid file path")
    except pd.errors.ParserError:
        print(f"Parser Error: Error while parsing '{path}'")
    except ZeroDivisionError:
        print("ZeroDivisionError: Please check if the data file is correct.")


if __name__ == "__main__":
    main()
