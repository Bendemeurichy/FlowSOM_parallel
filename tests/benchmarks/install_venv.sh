#!/bin/bash -l
#PBS -l walltime=1:00:00
#PBS -l mem=16gb

# In de comments hierboven vragen 1 uur aan wallclock time en 16GB RAM.

# Met -v kunnen we argumenten doorgeven die dan ingevuld worden in de variabelen
# in het script. Bijvoorbeeld:
#
# $ qsub install_venv.pbs -v cluster=donphan
#
# zorgt ervoor dat de variabele $cluster de waarde "donphan" bevat.

# We laden Python 3.11 in als module
module load SciPy-bundle/2023.11-gfbf-2023b

# Navigeren naar de map "demo" in de Data directory
PIP_DIR="$VSC_SCRATCH/site-packages" # directory to install packages
CACHE_DIR="$VSC_SCRATCH/.cache" # directory to use as cache

# Als er geen cluster-argument meegegeven is, geven we een fout terug
if [ -z $cluster ]; then
	echo "Provide a cluster argument using -v, e.g.: qsub install_venv -v cluster=donphan" 1>&2
	exit 1
fi

# De naam van de virtual environment die we gaan maken
# Deze hangt af van de gekozen cluster
VENV_NAME="venv_$cluster"

# Indien er al zo'n virtual environment bestaat, verwijder ze
if [ -d $VENV_NAME ];
then
	rm -r $VENV_NAME
fi

# Maak de virtual environment aan en installeer packages
# Je kan dit ook doen met een requirements.txt file:
# https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#using-requirements-files
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate
pip install --upgrade -e $VSC_HOME/project

