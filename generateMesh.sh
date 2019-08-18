#!/bin/bash

USAGE="
Usage: ./$(basename $0) [OPTIONS] SIZE SPACE FIXED HOOKE MASS
"

OPT_BINDINGS=false
OPT_PARTICLES=false

PARAM_SIZE=""
PARAM_SPACE=""
PARAM_FIXED=""
PARAM_HOOKE=""
PARAM_MASS=""

PARAM_INDEX=0
while [ $# -gt 0 ]
do
	if grep -q '^-' <<< $1
	then
		case $1 in
		-b|--bindings) OPT_BINDINGS=true; shift ;;
		-h|--help) echo "${USAGE}"; exit 0 ;;
		-p|--particles) OPT_PARTICLES=true; shift ;;
		esac
	else
		case ${PARAM_INDEX} in
		0) PARAM_SIZE=$1; shift ;;
		1) PARAM_SPACE=$1; shift ;;
		2) PARAM_FIXED=$1; shift ;;
		3) PARAM_HOOKE=$1; shift ;;
		4) PARAM_MASS=$1; shift ;;
		*) echo "Invalid parameter '$1'" >&2; exit 1 ;;
		esac
		((++PARAM_INDEX))
	fi
done

if [ -z ${PARAM_MASS} ]; then
	echo "Too few parameters" >&2; exit 1
fi

if [ ${OPT_BINDINGS} == false ] && [ ${OPT_PARTICLES} == false ]; then
	echo "Must specify -p, -b, or both" >&2; exit 1
fi

LAST_INDEX=$((PARAM_SIZE-1))

particles () {
	# Particles
	for XY in $( seq 0 ${LAST_INDEX} )
	do
		DIAGONAL=$(echo "sqrt(2.0000)*0.5000*${PARAM_SPACE}*${XY}" | bc)
		PARTICLE=""
		for Z in $( seq 0 ${LAST_INDEX} )
		do
			FIXED=0
			NEGATIVE="-"
			if [ $(( Z + XY )) -lt ${PARAM_FIXED} ] \
				|| [ $(( (LAST_INDEX - Z) + XY )) -lt ${PARAM_FIXED} ] \
				|| [ $(( (LAST_INDEX - XY) + Z )) -lt ${PARAM_FIXED} ] \
				|| [ $(( (LAST_INDEX - Z) + (LAST_INDEX - XY) )) -lt ${PARAM_FIXED} ]
			then
				FIXED=1
			fi
			if [ $Z -eq 0 ]
			then
				NEGATIVE=""
			fi

			PARTICLE="${PARTICLE}p ${FIXED} ${DIAGONAL} ${DIAGONAL} ${NEGATIVE}${Z} ${PARAM_MASS}\n"
		done
		echo -ne "${PARTICLE}"
	done
}

bindings () {
	# Bindings
	for XY in $( seq 0 ${LAST_INDEX} )
	do
		for Z in $( seq 0 ${LAST_INDEX} )
		do
			INDEX=$((XY*PARAM_SIZE + Z))
			BINDING=""
			# Structure
			[ $Z -ne 0 ] \
				&& BINDING="${BINDING}b ${INDEX} $((INDEX - 1)) ${PARAM_HOOKE}\n"
			[ $XY -ne 0 ] \
				&& BINDING="${BINDING}b ${INDEX} $((INDEX - PARAM_SIZE)) ${PARAM_HOOKE}\n"
			# Cross
			[ $Z -ne 0 ] && [ $XY -ne 0 ] \
				&& BINDING="${BINDING}b ${INDEX} $((INDEX - PARAM_SIZE - 1)) ${PARAM_HOOKE}\nb $((INDEX - 1)) $((INDEX - PARAM_SIZE)) ${PARAM_HOOKE}\n"
			[ -n "${BINDING}" ] && echo -ne "${BINDING}"
		done
	done
}

[ ${OPT_PARTICLES} == true ] && particles
[ ${OPT_BINDINGS} == true ] && bindings
