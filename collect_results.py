from glob import glob
import argparse

def main(opt):
    result_dir = opt.result_dir
    metric_files = glob(f"{result_dir}/**/metrics.txt", recursive=True)
    with open(f"{result_dir}/total_metrics.txt", "w") as fout:
        metrics = {"base":[], "ours": []}
        for file_p in sorted(metric_files):
            with open(file_p, "r") as fin:
                metric = fin.read()
                method = "ours" if "ours/" in file_p else "base"
                metric_elements = metric.split("\n")
                metric_elements[0] = metric_elements[0] + "-" + method
                metric_elements[1:] = map(lambda x: f"{round(float(x), 3):.3f}", metric_elements[1:])
                metrics[method].append(metric_elements)
        baseline_output = zip(*metrics["base"])
        ours_output = zip(*metrics["ours"])
        for line in baseline_output:
            fout.write("\t".join(line))
            fout.write("\n")
        fout.write("\n")
        for line in ours_output:
            fout.write("\t".join(line))
            fout.write("\n")
        fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        help="dir to result images",
    )

    args = parser.parse_args()

    main(args)