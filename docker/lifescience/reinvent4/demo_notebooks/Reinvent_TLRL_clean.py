# This script is based on the file notebooks/Reinvent_TLRL.py:
# https://github.com/MolecularAI/REINVENT4/blob/main/notebooks/Reinvent_TLRL.py

import os
import re
import shutil
import subprocess

import pandas as pd


def main():
    wd = "R4_TLRL_output"

    # Delete existing working directory and create a new one
    if not os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
        os.mkdir(wd)
    os.chdir(wd)

    # Write config file
    prior_filename = "../priors/reinvent.prior"
    agent_filename = prior_filename
    stage1_checkpoint = "stage1.chkpt"
    stage1_parameters = f"""
    run_type = "staged_learning"
    device = "cuda:0"
    tb_logdir = "tb_stage1"
    json_out_config = "_stage1.json"

    [parameters]
    prior_file = "{prior_filename}"
    agent_file = "{agent_filename}"
    summary_csv_prefix = "stage1"
    batch_size = 100
    use_checkpoint = false
    sample_strategy = "beamsearch" #Additional interesting param?

    [learning_strategy]
    type = "dap"
    sigma = 128
    rate = 0.0001

    [[stage]]
    max_score = 1.0
    max_steps = 5
    chkpt_file = "{stage1_checkpoint}"
    [stage.scoring]
    type = "geometric_mean"
    [[stage.scoring.component]]
    [stage.scoring.component.custom_alerts]
    [[stage.scoring.component.custom_alerts.endpoint]]
    name = "Alerts"
    params.smarts = [ "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]", "[*;r13]", "[*;r14]", "[*;r15]", "[*;r16]", "[*;r17]", "[#8][#8]", "[#6;+]", "[#16][#16]", "[#7;!n][S;!$(S(=O)=O)]", "[#7;!n][#7;!n]", "C#C", "C(=[O,S])[O,S]", "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]", "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]", "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]", "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]", "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]", "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]" ]
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

    stage1_config_filename = "stage1.toml"
    with open(stage1_config_filename, "w") as tf:
        tf.write(stage1_parameters)

    # Stage 1 Reinforcement Learning
    shutil.rmtree("tb_stage1_0", ignore_errors=True)

    # Run the stage1 process using subprocess
    print("Starting Stage 1 Reinforcement Learning...")
    stage1_result = subprocess.run(f"reinvent {stage1_config_filename} 2>&1 | tee stage1.log", shell=True, text=True)
    if stage1_result.returncode == 0:
        print("Stage 1 completed.")
    else:
        raise RuntimeError(f"Stage 1 execution failed with exit code: {stage1_result.returncode}")

    # Transfer Learning to focus the model
    # Prepare the data
    bdb = pd.read_csv("../notebooks/data/tnks2.csv")
    clean = bdb[~bdb["exp (nM)"].str.match("[<>]")]
    clean = clean.astype({"exp (nM)": "float"})

    good = clean[clean["exp (nM)"] < 1000]
    good = good[good["exp_method"] != "EC50"]
    good = good[good["exp_method"] != "Kd"]
    good = good.rename(columns={"exp (nM)": "IC50"})
    good = good.drop(columns=["exp_method"])

    # Write the good binders to a SMILES file
    TL_train_filename = "tnks2_train.smi"
    TL_validation_filename = "tnks2_validation.smi"
    data = good.sample(frac=1)
    n_head = int(0.8 * len(data))  # 80% of the data for training
    n_tail = len(good) - n_head
    print(f"number of molecules for: training={n_head}, validation={n_tail}")

    train, validation = data.head(n_head), data.tail(n_tail)
    train.to_csv(TL_train_filename, sep="\t", index=False, header=False)
    validation.to_csv(TL_validation_filename, sep="\t", index=False, header=False)

    # TL setup
    TL_parameters = f"""
    run_type = "transfer_learning"
    device = "cuda:0"
    tb_logdir = "tb_TL"

    [parameters]
    num_epochs = 1
    save_every_n_epochs = 1
    batch_size = 100
    sample_batch_size = 2000
    input_model_file = "{stage1_checkpoint}"
    output_model_file = "TL_reinvent.model"
    smiles_file = "{TL_train_filename}"
    validation_smiles_file = "{TL_validation_filename}"
    standardize_smiles = true
    randomize_smiles = true
    randomize_all_smiles = false
    internal_diversity = true
    """

    TL_config_filename = "transfer_learning.toml"
    with open(TL_config_filename, "w") as tf:
        tf.write(TL_parameters)

    # Start Transfer Learning
    shutil.rmtree("tb_TL", ignore_errors=True)

    # Run the transfer learning process using subprocess
    print("Starting Transfer Learning...")
    transfer_result = subprocess.run(
        f"reinvent {TL_config_filename} 2>&1 | tee transfer_learning.log", shell=True, text=True
    )
    if transfer_result.returncode == 0:
        print("Transfer learning completed.")
    else:
        raise RuntimeError(f"Transfer learning execution failed with exit code: {transfer_result.returncode}")

    # Choose the model from transfer learning
    TL_model_filename = "TL_reinvent.model.1.chkpt"

    stage2_parameters = re.sub("stage1", "stage2", stage1_parameters)
    stage2_parameters = re.sub("agent_file.*\n", f"agent_file = '{TL_model_filename}'\n", stage2_parameters)
    stage2_parameters = re.sub("max_steps.*\n", "max_steps = 5\n", stage2_parameters)

    # Stage 2 RL
    # Predictive model (ChemProp)
    chemprop_path = "../chemprop/"
    pred_model_parameters = f"""
    [[stage.scoring.component]]
    [stage.scoring.component.ChemProp]
    [[stage.scoring.component.ChemProp.endpoint]]
    name = "ChemProp"
    weight = 0.6
    params.checkpoint_dir = "{chemprop_path}"
    params.rdkit_2d_normalized = true
    params.target_column = "DG"
    params.features = "rdkit_2d_normalized"
    transform.type = "reverse_sigmoid"
    transform.high = 0.0
    transform.low = -50.0
    transform.k = 0.4
    """

    # Combine parameters and write to file
    full_stage2_parameters = stage2_parameters + pred_model_parameters
    df_parameters = """
    [diversity_filter]
    type = "IdenticalMurckoScaffold"
    bucket_size = 10
    minscore = 0.7
    """
    inception_parameters = """
    [inception]
    smiles_file = ""  # no seed SMILES
    memory_size = 50
    sample_size = 10
    """

    full_stage2_parameters += df_parameters + inception_parameters
    stage2_config_filename = "stage2.toml"
    with open(stage2_config_filename, "w") as tf:
        tf.write(full_stage2_parameters)

    # Run stage2 using subprocess
    print("Starting Stage 2 Reinforcement Learning...")
    stage2_result = subprocess.run(f"reinvent {stage2_config_filename} 2>&1 | tee stage2.log", shell=True, text=True)
    if stage2_result.returncode == 0:
        print("Stage 2 completed.")
    else:
        raise RuntimeError(f"Stage 2 execution failed with exit code: {stage2_result.returncode}")

    # Inspect results with TensorBoard
    # Run TensorBoard separately after REINVENT finishes
    # subprocess.run(["tensorboard", "--bind_all", "--logdir", f"{wd}/tb_stage2_0"])

    # Process the results for good binders
    # csv_file = os.path.join(wd, "stage2_1.csv")
    csv_file = "stage2_1.csv"
    df = pd.read_csv(csv_file)
    good_QED = df["QED"] > 0.8
    good_dG = df["ChemProp (raw)"] < -25.0  # kcal/mol
    good_binders = df[good_QED & good_dG]
    print(len(good_binders))

    # Duplicate removal
    good_binders = good_binders.drop_duplicates(subset=["SMILES"])
    print(len(good_binders))

    # Displaying good binders
    # grid = create_mol_grid(good_binders)
    # display(grid)


if __name__ == "__main__":
    main()
