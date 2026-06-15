#!/bin/bash

#SBATCH --job-name=ShockDroplet
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 20:00:00

#cd /mmfs1/home/ttryon/flames/bin/

### OLD HYDRO ###
#mpirun /home/ttryon/flames/bin/hydro-2d-g++ /home/ttryon/flames/tests/FlowRiemannShockTube/input

#### HDYRO 2 ####

#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/input2
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleSigmaHigh
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleGammaHigh
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleNewKappa
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleOldKappa
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleOldKappaHighSigmaHighGamma
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleNewKappaHighSigmaHighGamma

#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/LargeBubbleGammaHighTammann

#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleGammaHighTammannSame
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/RayleighTaylorInstability
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRiemannUnitTests/input_AllaireAirWater

#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowTwoPhase/LargeBubbleGammaHighTammannVapor


# Couette Flow # 
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowCouette/input_hydro2
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowCouette/input_hydro2_2phase


# Shock Droplet Interaction #
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/RockULikeAHuricane
#srun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet
#srun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_5Ma_H20
#srun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/1mm_Droplet_5Ma_n-Pentane
srun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/input_He_Braconnier
