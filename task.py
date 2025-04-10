#5 different Python logic examples to calculate Mean, Median, and Mode
#1. Using Built-in statistics Module
import statistics

data = [1, 2, 2, 3, 4, 5, 5, 5, 6]

mean = statistics.mean(data)
median = statistics.median(data)
mode = statistics.mode(data)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

#output
#Mean: 3.6666666666666665
#Median: 4
#Mode: 5

#2. Manual Logic without Libraries
data = [1, 2, 2, 3, 4, 5, 5, 5, 6]

# Mean
mean = sum(data) / len(data)

# Median
sorted_data = sorted(data)
n = len(sorted_data)
median = (sorted_data[n // 2] if n % 2 != 0 
          else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2)

# Mode
frequency = {}
for number in data:
    frequency[number] = frequency.get(number, 0) + 1

mode = max(frequency, key=frequency.get)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
#output
#Mean: 3.6666666666666665
#Median: 4
#Mode: 5

#3. Using NumPy
import numpy as np
from scipy import stats

data = [1, 2, 2, 3, 4, 5, 5, 5, 6]

mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0][0]  # returns array and count

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
#output
#Mean: 3.6666666666666665
#Median: 4.0
#Mode: 5

#4.Using Pandas
import pandas as pd

data = [1, 2, 2, 3, 4, 5, 5, 5, 6]
series = pd.Series(data)

mean = series.mean()
median = series.median()
mode = series.mode()[0]  # Can return multiple, we pick the first

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
#output
#Mean: 3.6666666666666665
#Median: 4.0
#Mode: 5

#5. Handling Multiple Modes (Custom Logic)
from collections import Counter

data = [1, 2, 2, 3, 4, 5, 5, 5, 6]

# Mean
mean = sum(data) / len(data)

# Median
sorted_data = sorted(data)
n = len(data)
median = sorted_data[n // 2] if n % 2 else (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2

# Mode (all with highest frequency)
counts = Counter(data)
max_freq = max(counts.values())
mode = [key for key, val in counts.items() if val == max_freq]

print("Mean:", mean)
print("Median:", median)
print("Mode(s):", mode)
#output
#Mean: 3.6666666666666665
#Median: 4
#Mode(s): [5]

#5 different Python logic examples to  Standard Deviation
#1. Basic Approach using Python

import math

def standard_deviation_basic(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return math.sqrt(variance)

data = [10, 12, 23, 23, 16, 23, 21, 16]
result = standard_deviation_basic(data)
print("Standard Deviation (Basic):", result)
#output
#Standard Deviation (Basic): 4.898979485566356

#2. Using Numpy (Library-based)
import numpy as np

def standard_deviation_numpy(data):
    return np.std(data, ddof=1)  # ddof=1 for sample standard deviation

data = [10, 12, 23, 23, 16, 23, 21, 16]
result = standard_deviation_numpy(data)
print("Standard Deviation (Numpy):", result)
#output
#Standard Deviation (Numpy): 4.898979485566356

#3. Using a for-loop (Manual Calculation)

def standard_deviation_loop(data):
    n = len(data)
    mean = sum(data) / n
    variance = 0
    for x in data:
        variance += (x - mean) ** 2
    variance /= (n - 1)
    return variance ** 0.5

data = [10, 12, 23, 23, 16, 23, 21, 16]
result = standard_deviation_loop(data)
print("Standard Deviation (Loop):", result)
#output
#Standard Deviation (Loop): 4.898979485566356

#4. Using the Statistics Module
import statistics

def standard_deviation_statistics(data) :
    return statistics.stdev(data)

data = [10, 12, 23, 23, 16, 23, 21, 16]
result = standard_deviation_statistics(data)
print("Standard Deviation (Statistics Module):", result)
#output
#Standard Deviation (Statistics Module): 4.898979485566356

#5. Using Pandas (DataFrames)
import pandas as pd

def standard_deviation_pandas(data):
    return pd.Series(data).std()

data = [10, 12, 23, 23, 16, 23, 21, 16]
result = standard_deviation_pandas(data)
print("Standard Deviation (Pandas):", result)
#output
#Standard Deviation (Pandas): 4.898979485566356

#5 different Python logic examples to Percentiles

def percentile_basic(data, percentile):
    data.sort()
    index = (percentile / 100) * (len(data) - 1)
    lower = int(index)
    upper = lower + 1 if lower + 1 < len(data) else lower
    weight = index - lower
    return data[lower] + weight * (data[upper] - data[lower])

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
percentile_25 = percentile_basic(data, 25)
percentile_50 = percentile_basic(data, 50)
percentile_75 = percentile_basic(data, 75)
print(f"25th percentile: {percentile_25}")
print(f"50th percentile (Median): {percentile_50}")
print(f"75th percentile: {percentile_75}")
#output
#25th percentile: 27.5
#50th percentile (Median): 50.0
#75th percentile: 72.5


#5 different Python logic examples to  Percentiles
#1. Basic Approach without any Libraries
def percentile_basic(data, percentile):
    data.sort()
    index = (percentile / 100) * (len(data) - 1)
    lower = int(index)
    upper = lower + 1 if lower + 1 < len(data) else lower
    weight = index - lower
    return data[lower] + weight * (data[upper] - data[lower])

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
percentile_25 = percentile_basic(data, 25)
percentile_50 = percentile_basic(data, 50)
percentile_75 = percentile_basic(data, 75)
print(f"25th percentile: {percentile_25}")
print(f"50th percentile (Median): {percentile_50}")
print(f"75th percentile: {percentile_75}")
#output
#25th percentile: 27.5
#50th percentile (Median): 50.0
#75th percentile: 72.5

#2. Using Numpy (Library-based)
import numpy as np

def percentile_numpy(data, percentile):
    return np.percentile(data, percentile)

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
percentile_25 = percentile_numpy(data, 25)
percentile_50 = percentile_numpy(data, 50)
percentile_75 = percentile_numpy(data, 75)
print(f"25th percentile: {percentile_25}")
print(f"50th percentile (Median): {percentile_50}")
print(f"75th percentile: {percentile_75}")
#output
#25th percentile: 27.5
#50th percentile (Median): 50.0
#75th percentile: 72.5

#3. Using the Statistics Module
import statistics

def percentile_statistics(data, percentile):
    return statistics.quantiles(data, n=100)[percentile-1]

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
percentile_25 = percentile_statistics(data, 25)
percentile_50 = percentile_statistics(data, 50)
percentile_75 = percentile_statistics(data, 75)
print(f"25th percentile: {percentile_25}")
print(f"50th percentile (Median): {percentile_50}")
print(f"75th percentile: {percentile_75}")
#output
#25th percentile: 27.5
#50th percentile (Median): 50.0
#75th percentile: 72.5

#4. Using Pandas (DataFrames)
import pandas as pd

def percentile_pandas(data, percentile):
    return pd.Series(data).quantile(percentile / 100)

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
percentile_25 = percentile_pandas(data, 25)
percentile_50 = percentile_pandas(data, 50)
percentile_75 = percentile_pandas(data, 75)
print(f"25th percentile: {percentile_25}")
print(f"50th percentile (Median): {percentile_50}")
print(f"75th percentile: {percentile_75}")
#output
#25th percentile: 27.5
#50th percentile (Median): 50.0
#75th percentile: 72.5

#5. Using a For-Loop for Manual Calculation
def percentile_loop(data, percentile):
    data.sort()
    index = (percentile / 100) * (len(data) - 1)
    lower = int(index)
    upper = lower + 1 if lower + 1 < len(data) else lower
    weight = index - lower
    return data[lower] + weight * (data[upper] - data[lower])

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
percentile_25 = percentile_loop(data, 25)
percentile_50 = percentile_loop(data, 50)
percentile_75 = percentile_loop(data, 75)
print(f"25th percentile: {percentile_25}")
print(f"50th percentile (Median): {percentile_50}")
print(f"75th percentile: {percentile_75}")
#output
#25th percentile: 27.5
#50th percentile (Median): 50.0
#75th percentile: 72.5


#2 different Python logic examples to  Data Distribution
#1. Using Numpy for Basic Statistics (Distribution Features)
import numpy as np

def data_distribution_numpy(data):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    variance = np.var(data)
    return mean, median, std_dev, variance

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
mean, median, std_dev, variance = data_distribution_numpy(data)
print(f"Mean: {mean}, Median: {median}, Standard Deviation: {std_dev}, Variance: {variance}")
#output
#Mean: 50.0, Median: 50.0, Standard Deviation: 26.457513110669084, Variance: 700.0

#2.import pandas as pd

def data_distribution_pandas(data):
    series = pd.Series(data)
    description = series.describe()  # Returns count, mean, std, min, 25%, 50%, 75%, max
    return description

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
distribution = data_distribution_pandas(data)
print(distribution)
#output
#count     9.000000
#mean     50.000000
#std      26.457513
#min      10.000000
#25%      30.000000
#50%      50.000000
#75%      70.000000
#max      90.000000
#dtype: float64










