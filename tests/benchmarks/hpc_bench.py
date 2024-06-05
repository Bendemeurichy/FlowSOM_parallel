import csv
import logging
import os
import timeit

import numpy as np
from flowio import FlowData
from memory_profiler import memory_usage
from sklearn.metrics import v_measure_score

import flowsom as fs
from flowsom.pp import aggregate_flowframes


def read_labelled_fcs(path, label_column=-1):
    # read in FCS file
    fcs_data = FlowData(path)
    # convert to numpy array
    npy_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
    # get label column
    y = npy_data[:, label_column]
    # filter out unlabelled data
    mask = ~np.isnan(y)
    X = npy_data[mask, :-1]
    y = npy_data[mask, label_column]
    # if no 0 in y, subtract 1 from all labels
    # this is to make sure that the labels start at 0, as sklearn clustering algorithms usually output
    if 0 not in y:
        y = y - 1
    # cast y to int
    y = y.astype(np.int32)
    return X, y


def bench_file(path: str, flowsom_implementation, dimensions: int, label_col=-1, cols_to_use: np.ndarray = None,
               seed: int = None, variant: str = "umba", batch=False, batch_size=0):
    """
    Benchmark a file with the given implementation of flowsom, this includes time and v-measure score
    @param path: path to the fcs file
    @param flowsom_implementation: implementation of flowsom to use
    @param dimensions: number of dimensions to use
    @param label_col: column to use as the column containing the labels
    @param cols_to_use: columns to use
    @param seed: random seed to use
    @param variant: the version of the SOM algorithm to use, either "numba","original", "lr" or "xpysom"
    @param batch: if True, the batch version of the SOM algorithm will be used
    @param batch_size: the batch size to use
    """
    # read in fcs file
    X, y = read_labelled_fcs(path, label_col)

    # finding the best number of clusters is not part of this test
    # here we use labeled data to find the number of unique labels
    n_clusters = np.unique(y).shape[0]

    # cluster data and predict labels
    fsom = []
    exec_time = timeit.timeit(lambda: fsom.append(
        flowsom_implementation(X, n_clusters=max(n_clusters, dimensions, len(cols_to_use) if cols_to_use is not None else 0),
                               xdim=10, ydim=10, cols_to_use=cols_to_use, seed=seed, variant=variant, batch=batch,
                               batch_size=batch_size)), number=1)
    y_pred = fsom[0].metacluster_labels

    # Measure peak memory usage
    peak_memory = max(memory_usage(proc=(
        lambda: flowsom_implementation(X,
                                       n_clusters=max(n_clusters, dimensions, len(cols_to_use) if cols_to_use is not None else 0),
                                       xdim=10, ydim=10, cols_to_use=cols_to_use, seed=seed, variant=variant,
                                       batch=batch, batch_size=batch_size)), interval=0.1))

    # because the v_measure_score is independent of the absolute values of the labels
    # we don"t need to make sure the predicted label values have the same value as the true labels
    # the v_measure_score will be the same regardless, as it only depends on homogeneity and completeness
    # alternatively, a lookup table from the cluster centers can be used to have a consistent label value mapping
    # https://stackoverflow.com/questions/44888415/how-to-set-k-means-clustering-labels-from-highest-to-lowest-with-python
    v_measure = v_measure_score(y, y_pred)
    print(f"V-measure score: {v_measure}")
    print(f"Execution time: {exec_time}s")
    print(f"Peak memory usage: {peak_memory:.2f} MiB")
    return v_measure, exec_time, peak_memory


def get_bench_params() -> list[tuple]:
    """Get the benchmark parameters from the metadata_accuracy_bench.csv file.
    The file should have the following columns:
    - Filename: The path to the FCS file form the benchmark directory
    - Dimensions: The number of dimensions to use
    - Begin column: The first column to use
    - End column: The last column to use
    - Label column: The column to use as label
    """
    params = []
    with open("./metadata_accuracy_bench.csv") as meta_csv:
        csv_reader = csv.reader(meta_csv, delimiter=",")
        for row in csv_reader:
            params.append((row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])))
    return params


def accuracy_benchmarks():
    """Run the accuracy benchmarks for each FlowSOM implementation."""
    params = get_bench_params()
    with (open(f"{os.environ['VSC_DATA']}/output_flowsom/accuracy_numbsom.csv", "w") as f1,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/accuracy_xpysom.csv", "w") as f2,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/accuracy_original.csv", "w") as f3,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/accuracy_lr.csv", "w") as f4,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/accuracy_batch_som.csv", "w") as f5):
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer3 = csv.writer(f3)
        writer4 = csv.writer(f4)
        writer5 = csv.writer(f5)
        logging.info("Running accuracy benchmarks")
        for param in params:
            logging.info(f"Running accuracy benchmarks for {param[0]}")
            for i in range(10):
                logging.info(f"Running accuracy benchmarks for {param[0]} iteration {i}/10")
                path = f"../data/accuracy_benches/{param[0]}"
                logging.info(f"Running benchmarks for {path}")
                cols = list(range(param[2], param[3]))
                seed = np.random.randint(0, 100)
                writer1.writerow((param[0],
                                  *bench_file(path, fs.FlowSOM, dimensions=param[1],
                                              cols_to_use=cols, label_col=param[4], variant="numba", seed=seed)))
                logging.info(f"Running numbasom benchmarks for {param[0]} iteration {i}/10")
                writer2.writerow((param[0],
                                  *bench_file(path, fs.FlowSOM, dimensions=param[1],
                                              cols_to_use=cols, label_col=param[4], variant="xpysom", seed=seed,
                                              batch=True)))
                logging.info(f"Running xpysom benchmarks for {param[0]} iteration {i}/10")
                writer3.writerow(
                    (param[0], *bench_file(path, fs.FlowSOM, dimensions=param[1],
                                           cols_to_use=cols, label_col=param[4], variant="original", seed=seed)))
                logging.info(f"Running original benchmarks for {param[0]} iteration {i}/10")
                writer4.writerow(
                    (param[0], *bench_file(path, fs.FlowSOM, dimensions=param[1],
                                           cols_to_use=cols, label_col=param[4], variant="lr", seed=seed)))
                logging.info(f"Running learnig rate benchmarks for {param[0]} iteration {i}/10")
                writer5.writerow(
                    (param[0], *bench_file(path, fs.FlowSOM, dimensions=param[1],
                                           cols_to_use=cols, label_col=param[4], variant="batch_som", seed=seed,
                                           batch=True)))
                logging.info(f"Running my batch benchmarks for {param[0]} iteration {i}/10")


def speed_benchmarks():
    logging.info("Running performance benchmarks")
    files = os.listdir(f"{os.environ['VSC_DATA']}/performance_benches")
    logging.info(f"Found {len(files)} files")
    logging.info("Aggregating flowframes")
    cell_count = 0
    for file in files:
        f = FlowData(f"{os.environ['VSC_DATA']}/performance_benches/{file}")
        cell_count += f.event_count

    frame = aggregate_flowframes(files, cell_count)

    with (open(f"{os.environ['VSC_DATA']}/output_flowsom/performance_numbsom.csv", "w") as f1,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/performance_xpysom.csv", "w") as f2,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/performance_original.csv", "w") as f3,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/performance_lr.csv", "w") as f4,
          open(f"{os.environ['VSC_DATA']}/output_flowsom/performance_batch_som.csv", "w") as f5):
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer3 = csv.writer(f3)
        writer4 = csv.writer(f4)
        writer5 = csv.writer(f5)
        for i in range(10):
            logging.info(f"Running performance benchmarks iteration {i}/10")
            seed = np.random.randint(0, 100)
            writer1.writerow(bench_file(frame, fs.FlowSOM, dimensions=40, variant="numba", seed=seed))
            logging.info(f"Finished numbasom benchmarks iteration {i}/10")

            writer2.writerow(bench_file(frame, fs.FlowSOM, dimensions=40, variant="xpysom", seed=seed, batch=True))
            logging.info(f"Finished xpysom benchmarks iteration {i}/10")

            writer3.writerow(bench_file(frame, fs.FlowSOM, dimensions=40, variant="original", seed=seed))
            logging.info(f"Finished original benchmarks iteration {i}/10")

            writer4.writerow(bench_file(frame, fs.FlowSOM, dimensions=40, variant="lr", seed=seed))
            logging.info(f"Finished lr benchmarks iteration {i}/10")

            writer5.writerow(bench_file(frame, fs.FlowSOM, dimensions=40, variant="batch_som", seed=seed, batch=True))
            logging.info(f"Finished my batch benchmarks iteration {i}/10")


def main():
    logging.basicConfig(level=logging.INFO,filename='app.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

    accuracy_benchmarks()
    speed_benchmarks()


if __name__ == "__main__":
    main()
