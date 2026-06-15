#!/bin/bash

#SBATCH --job-name=RPE-KM
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 10:00:00

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

#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowKellerMiksis/input_KellerMiksis_MildOscillation_NSCBC_Large

srun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowRayleighPlesset/Sch20_Oscillating_minmod_Neumann_Large

# Couette Flow # 
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowCouette/input_hydro2
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowCouette/input_hydro2_2phase


# Shock Droplet Interaction #
#mpirun /home/ttryon/flames/bin/hydro2-2d-g++ /home/ttryon/flames/tests/FlowShockDroplet/RockULikeAHuricane
