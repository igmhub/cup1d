import os, sys, time, subprocess, textwrap
import numpy as np
from itertools import product

# our own modules
from lace.archive import gadget_archive, nyx_archive
from lace.cosmo import camb_cosmo
from cup1d.data import data_gadget, data_nyx
from cup1d.scripts.sam_sim import sam_sim, path_sampler


class Args:
    def __init__(self):
        self.version = "v1"
        self.training_set = "Pedersen21"
        self.emulator_label = "Pedersen21"
        self.mock_sim_label = "mpg_central"
        self.igm_sim_label = "mpg_central"
        self.cosmo_sim_label = "mpg_central"
        self.drop_sim = True
        self.add_hires = False
        self.use_polyfit = True
        self.cov_label = "Chabanier2019"
        self.emu_cov_factor = 0
        self.n_igm = 2
        self.prior_Gauss_rms = None
        self.test = False
        self.parallel = True
        self.verbose = True
        self.add_noise = False
        self.seed_noise = 0
        self.par2save = [
            "training_set",
            "emulator_label",
            "mock_sim_label",
            "igm_sim_label",
            "cosmo_sim_label",
            "drop_sim",
            "add_hires",
            "use_polyfit",
            "cov_label",
            "emu_cov_factor",
            "n_igm",
            "prior_Gauss_rms",
        ]

    def save(self):
        out = {}
        for par in self.par2save:
            out[par] = getattr(self, par)
        return out


def generate_batch_script(
    slurm_script_path,
    python_script_path,
    out_path,
    seed,
    args,
    n_tasks=32,
    n_nodes=1,
    qos="debug",
):
    # SLURM script content
    slurm_script_content = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --qos={qos}
        #SBATCH --account=desi
        #SBATCH --nodes={n_nodes}
        #SBATCH --ntasks-per-node={n_tasks}
        #SBATCH --constraint=cpu
        #SBATCH --output={out_path}output{seed}.log
        #SBATCH --error={out_path}error{seed}.log

        mpiexec -n {n_tasks} python {python_script_path}\
        --training_set {args.training_set}\
        --emulator_label {args.emulator_label}\
        --mock_sim_label {args.mock_sim_label}\
        --igm_sim_label {args.igm_sim_label}\
        --cosmo_sim_label {args.cosmo_sim_label}\
        --n_igm {args.n_igm}\
        --cov_label {args.cov_label}\
        --emu_cov_factor {args.emu_cov_factor}\
    """
    )
    if args.drop_sim:
        slurm_script_content += " --drop_sim"
    if args.add_hires:
        slurm_script_content += " --add_hires"
    if args.use_polyfit:
        slurm_script_content += " --use_polyfit"
    if args.verbose:
        slurm_script_content += " --verbose"
    if args.parallel:
        slurm_script_content += " --parallel"
    if args.add_noise:
        slurm_script_content += " --add_noise"
        slurm_script_content += f" --seed_noise {args.seed_noise}"

    print(slurm_script_content)
    return slurm_script_content


def get_total_job_count():
    # Use squeue to get the count of total (running + pending) jobs for the current user
    command = "squeue -h -u $USER | wc -l"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def launch_batch_script(slurm_script):
    # Submit the SLURM job using subprocess
    subprocess.run(["sbatch", slurm_script])


def main():
    qos = "debug"

    python_script_path = (
        os.environ["CUP1D_PATH"] + "src/cup1d/scripts/sam_sim.py"
    )

    out_path = os.environ["CUP1D_PATH"] + "src/cup1d/scripts/nersc/runs/"

    list_training_set = ["Pedersen21", "Cabayol23", "Nyx23_Oct2023"]
    emulator_label = [
        "Pedersen21",
        "Cabayol23",
        "Cabayol23_extended",
        "Nyx_v0",
        "Nyx_v0_extended",
    ]

    # list of options to set
    version = "v1"
    training_set = "Cabayol23"
    emulator_label = "Cabayol23"
    add_hires = False
    # emulator_label = "Cabayol23_extended"
    # add_hires = True

    # l1O or not?
    # list_sim = archive.list_sim
    list_sim = ["mpg_central"]

    # use own IGM history or that of central simulation
    # arr_igm_own = [True, False]
    arr_igm_own = [True]

    # use own cosmo or that of central simulation
    # arr_cosmo_own = [True, False]
    arr_cosmo_own = [True]

    use_polyfit = True
    cov_label = "Chabanier2019"
    arr_drop_sim = [True]
    arr_n_igm = [0, 1]
    override = False

    # noise to mock
    add_noise = True
    arr_seed_noise = np.arange(10)

    # read archive from outside
    if training_set == "Pedersen21":
        # set_P1D = data_gadget.Gadget_P1D
        # get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        archive = gadget_archive.GadgetArchive(postproc=training_set)
        z_min = 2
        z_max = np.max(archive.list_sim_redshifts)
    elif training_set == "Cabayol23":
        # set_P1D = data_gadget.Gadget_P1D
        # get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        archive = gadget_archive.GadgetArchive(postproc=training_set)
        z_min = 2
        z_max = np.max(archive.list_sim_redshifts)
    elif training_set[:5] == "Nyx23":
        # set_P1D = data_nyx.Nyx_P1D
        # get_cosmo = camb_cosmo.get_Nyx_cosmology
        archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
        z_min = 2.2
        z_max = np.max(archive.list_sim_redshifts)
    else:
        raise ValueError("Training_set not implemented")

    sim_avoid = [
        "nyx_14",
        "nyx_15",
        "nyx_16",
        "nyx_17",
        "nyx_seed",
        "nyx_wdm",
    ]

    combined_loop = product(
        arr_igm_own,
        arr_cosmo_own,
        arr_drop_sim,
        arr_n_igm,
        list_sim,
        arr_seed_noise,
    )

    seed = 200
    for ind in combined_loop:
        igm_own, cosmo_own, drop_sim, n_igm, sim_label, seed_noise = ind
        if sim_label not in sim_avoid:
            print("")
            print("external loop")
            print("")

            args = Args()
            args.version = version
            args.training_set = training_set
            args.emulator_label = emulator_label
            args.z_max = z_max
            args.add_noise = add_noise
            args.seed_noise = seed_noise

            args.mock_sim_label = sim_label
            if igm_own:
                args.igm_sim_label = sim_label
            else:
                args.img_sim_label = sim_label[:3] + "_central"
            if cosmo_own:
                args.cosmo_sim_label = sim_label
            else:
                args.cosmo_sim_label = sim_label[:3] + "_central"

            args.add_hires = add_hires
            args.use_polyfit = use_polyfit
            args.cov_label = cov_label

            args.drop_sim = drop_sim
            args.n_igm = n_igm

            slurm_script_path = (
                os.environ["CUP1D_PATH"]
                + "src/cup1d/scripts/nersc/runs/slurm"
                + str(seed)
                + ".sub"
            )

            # check if job has been run
            if override:
                run = True
            else:
                fsummary = path_sampler(args) + "chain1/results.npy"
                if os.path.exists(fsummary):
                    run = False
                else:
                    run = True

            if run == True:
                if qos == "debug":
                    max_jobs = 5  # max 5 submitted jobs at a time
                    while get_total_job_count() >= max_jobs:
                        # Wait and check again after a delay if the limit is reached
                        print("Waiting for a job to finish")
                        time.sleep(30)

                    slurm_script_content = generate_batch_script(
                        slurm_script_path,
                        python_script_path,
                        out_path,
                        seed,
                        args,
                        qos=qos,
                    )
                    with open(slurm_script_path, "w") as slurm_script_file:
                        slurm_script_file.write(slurm_script_content)
                    launch_batch_script(slurm_script_path)
                elif qos == "regular":
                    slurm_script_content = generate_batch_script(
                        slurm_script_path,
                        python_script_path,
                        out_path,
                        seed,
                        args,
                        qos=qos,
                    )
                    with open(slurm_script_path, "w") as slurm_script_file:
                        slurm_script_file.write(slurm_script_content)
                    launch_batch_script(slurm_script_path)
                else:
                    raise ValueError("qos not implemented")

            seed += 1


if __name__ == "__main__":
    main()
