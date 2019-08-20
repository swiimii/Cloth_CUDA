#!/bin/bash

USAGE="
Usage:
  ./$(basename $0) SIZE SPACE FIXED MASS -b HOOKE_STRUCT HOOKE_CROSS 
  ./$(basename $0) SIZE SPACE FIXED MASS -s HOOKE_STRUCT -c HOOKE_CROSS 
  ./$(basename $0) SIZE SPACE FIXED MASS -s HOOKE_STRUCT
  ./$(basename $0) SIZE SPACE FIXED MASS -c HOOKE_CROSS 
  ./$(basename $0) -h

SIZE
  Grid dimension. Mesh will be SIZExSIZE.
FIXED
  How many particles to hold fixed at the corners.
MASS
  Mass of each particle.
HOOKE
  Hooke constant for the stiffness of bindings.

Options:
  -b|--bindings HOOKE_STRUCT HOOKE_CROSS
    Create all bindings. Equivalent to '-c HOOKE_CROSS -s HOOKE_STRUCT'
  -c|--cross HOOKE_CROSS
    Generate cross bindings with hooke constant HOOKE_CROSS
  -h|--help
    Show this message
  -s|--structure HOOKE_STRUCT
    Generate struct bindings with hooke constant HOOKE_STRUCT
"

OPT_CROSS=false
OPT_STRUCT=false

PARAM_SIZE=""
PARAM_SPACE=""
PARAM_FIXED=""
PARAM_HOOKE_STRUCT=""
PARAM_HOOKE_CROSS=""
PARAM_MASS=""

PARAM_INDEX=0
while [ $# -gt 0 ]
do
	if grep -q '^-' <<< $1
	then
		case $1 in
		-b|--bindings) OPT_CROSS=true; OPT_STRUCT=true; shift
			PARAM_HOOKE_STRUCT=$1; shift
			PARAM_HOOKE_CROSS=$1; shift ;;
		-c|--cross) OPT_CROSS=true; shift
			PARAM_HOOKE_CROSS=$1; shift ;;
		-h|--help) echo "${USAGE}"; exit 0 ;;
		-s|--structure) OPT_STRUCT=true; shift
			PARAM_HOOKE_STRUCT=$1; shift ;;
		*) echo "Invalid option '$1'" >&2; exit 1 ;;
		esac
	else
		case ${PARAM_INDEX} in
		0) PARAM_SIZE=$1; shift ;;
		1) PARAM_SPACE=$1; shift ;;
		2) PARAM_FIXED=$1; shift ;;
		3) PARAM_MASS=$1; shift ;;
		*) echo "Invalid parameter '$1'" >&2; exit 1 ;;
		esac
		((++PARAM_INDEX))
	fi
done

if [ -z ${PARAM_MASS} ]; then
	echo "Too few parameters" >&2; exit 1
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

			PARTICLE="${PARTICLE}p ${FIXED} ${DIAGONAL} ${DIAGONAL} ${NEGATIVE}$((Z * PARAM_SPACE)) ${PARAM_MASS}\n"
		done
		echo -ne "${PARTICLE}"
	done
}

bindingsCross () {
	# Bindings
	for XY in $( seq 0 ${LAST_INDEX} )
	do
		for Z in $( seq 0 ${LAST_INDEX} )
		do
			INDEX=$((XY*PARAM_SIZE + Z))
			BINDING=""
			# Cross
			[ $Z -ne 0 ] && [ $XY -ne 0 ] \
				&& BINDING="${BINDING}b ${INDEX} $((INDEX - PARAM_SIZE - 1)) ${PARAM_HOOKE_CROSS}\nb $((INDEX - 1)) $((INDEX - PARAM_SIZE)) ${PARAM_HOOKE_CROSS}\n"
			[ -n "${BINDING}" ] && echo -ne "${BINDING}"
		done
	done
}

bindingsStructure () {
	# Bindings
	for XY in $( seq 0 ${LAST_INDEX} )
	do
		for Z in $( seq 0 ${LAST_INDEX} )
		do
			INDEX=$((XY*PARAM_SIZE + Z))
			BINDING=""
			# Structure
			[ $Z -ne 0 ] \
				&& BINDING="${BINDING}b ${INDEX} $((INDEX - 1)) ${PARAM_HOOKE_STRUCT}\n"
			[ $XY -ne 0 ] \
				&& BINDING="${BINDING}b ${INDEX} $((INDEX - PARAM_SIZE)) ${PARAM_HOOKE_STRUCT}\n"
			[ -n "${BINDING}" ] && echo -ne "${BINDING}"
		done
	done
}
particles
[ ${OPT_CROSS} == true ] && bindingsCross
[ ${OPT_STRUCT} == true ] && bindingsStructure
exit 0
