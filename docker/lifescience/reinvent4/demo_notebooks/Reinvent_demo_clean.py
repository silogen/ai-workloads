# This script is based on the file notebooks/Reinvent_demo.py:
# https://github.com/MolecularAI/REINVENT4/blob/main/notebooks/Reinvent_demo.py

import os
import shutil
import subprocess

import pandas as pd


def setup_work_directory(wd):
    shutil.rmtree(wd, ignore_errors=True)
    os.mkdir(wd)
    os.chdir(wd)


def write_config_file(filename, config_data):
    with open(filename, "w") as tf:
        tf.write(config_data)


def run_reinvent(config_filename, log_filename="stage1.log"):
    try:
        result = subprocess.run(
            ["reinvent", "-l", log_filename, config_filename],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running REINVENT: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def analyze_results(wd):
    df = pd.read_csv("stage1_1.csv")
    print(df.head())
    return df


def calculate_sample_efficiency(df):
    total_smilies = len(df)
    total_invalid_smilies = len(df[df["SMILES_state"] == 0])
    total_batch_duplicate_smilies = len(df[df["SMILES_state"] == 2])
    total_duplicate_smilies = len(df[df.duplicated(subset=["SMILES"])])

    print(
        f"Total number of SMILES generated: {total_smilies}\n"
        f"Total number of invalid SMILES: {total_invalid_smilies}\n"
        f"Total number of batch duplicate SMILES: {total_batch_duplicate_smilies}\n"
        f"Total number of duplicate SMILES: {total_duplicate_smilies}"
    )


if __name__ == "__main__":
    wd = "R4_notebooks_output"
    setup_work_directory(wd)

    prior_filename = "../priors/reinvent.prior"
    agent_filename = prior_filename

    global_parameters = """
    run_type = "staged_learning"
    device = "cuda:0"
    tb_logdir = "tb_stage1"
    json_out_config = "_stage1.json"
    """
    parameters = f"""
    [parameters]

    prior_file = "{prior_filename}"
    agent_file = "{agent_filename}"
    summary_csv_prefix = "stage1"

    batch_size = 100

    use_checkpoint = false
    """

    learning_strategy = """
    [learning_strategy]

    type = "dap"
    sigma = 128
    rate = 0.0001
    """

    stages = """
    [[stage]]

    max_score = 1.0
    max_steps = 300

    chkpt_file = 'stage1.chkpt'

    [stage.scoring]
    type = "geometric_mean"

    [[stage.scoring.component]]
    [stage.scoring.component.custom_alerts]

    [[stage.scoring.component.custom_alerts.endpoint]]
    name = "Alerts"

    params.smarts = [
        "[*;r8]",
        "[*;r9]",
        "[*;r10]",
        "[*;r11]",
        "[*;r12]",
        "[*;r13]",
        "[*;r14]",
        "[*;r15]",
        "[*;r16]",
        "[*;r17]",
        "[#8][#8]",
        "[#6;+]",
        "[#16][#16]",
        "[#7;!n][S;!$(S(=O)=O)]",
        "[#7;!n][#7;!n]",
        "C#C",
        "C(=[O,S])[O,S]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
        "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
        "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
        "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
    ]

    [[stage.scoring.component]]
    [stage.scoring.component.QED]

    [[stage.scoring.component.QED.endpoint]]
    name = "QED"
    weight = 0.6


    [[stage.scoring.component]]
    [stage.scoring.component.NumAtomStereoCenters]

    [[stage.scoring.component.NumAtomStereoCenters.endpoint]]
    name = "Stereo"
    weight = 0.4

    transform.type = "left_step"
    transform.low = 0
    """

    config = global_parameters + parameters + learning_strategy + stages

    toml_config_filename = "stage1.toml"
    write_config_file(toml_config_filename, config)

    run_reinvent(toml_config_filename)
    df = analyze_results(wd)
    calculate_sample_efficiency(df)
