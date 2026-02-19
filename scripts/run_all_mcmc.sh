#!/bin/bash

#############################################################
#                                                           #
#  Automated MCMC Dust Composition Fitting                  #
#  Runs all apertures for BHR71 and L1448MM1                #
#                                                           #
#############################################################

# Set the Python script path
PYTHON_SCRIPT="/home/vorsteja/Documents/JOYS/JDust/mcmc_fitting_jitter_log_cmd.py"

# Log file
LOG_DIR="/home/vorsteja/Documents/JOYS/JDust/logdir/"
mkdir -p $LOG_DIR
MASTER_LOG="$LOG_DIR/mcmc_fitting_$(date +%Y%m%d_%H%M%S).log"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MCMC Dust Composition Fitting Suite  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Start time: $(date)"
echo "Master log: $MASTER_LOG"
echo ""

# Initialize counters
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# BHR71 apertures
BHR71_APERTURES=("A1" "A2" "A3" "A4" "B1" "B2" "B3" "B4" "C1" "C2")
#For Lukasz' data:
#("b1" "o5" "b2" "b3" "b4" "cr1")
#'cl1' 'cl2' 'cl3' 'cr2' 'cr3' 'cr4'

# L1448MM1 apertures - these need to be verified from your aperture file
# You may need to adjust this list based on your actual aperture names
L1448MM1_APERTURES=("B1"  "A1" "C1" "C2" "C3" "A2" "A3" "B2" "B3" "BS" )
#
#############################################################
#  Function to run a single fit                            #
#############################################################

run_fit() {
    local SOURCE=$1
    local APERTURE=$2
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Run $TOTAL_RUNS: $SOURCE - $APERTURE${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "Start time: $(date)"
    
    # Create individual log file
    LOCAL_LOG="$LOG_DIR/${SOURCE}_${APERTURE}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the Python script
    python3 $PYTHON_SCRIPT $SOURCE $APERTURE 2>&1 | tee $LOCAL_LOG
    
    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ SUCCESS: $SOURCE - $APERTURE${NC}"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        echo "$(date): SUCCESS - $SOURCE - $APERTURE" >> $MASTER_LOG
    else
        echo -e "${RED}✗ FAILED: $SOURCE - $APERTURE${NC}"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "$(date): FAILED - $SOURCE - $APERTURE" >> $MASTER_LOG
    fi
    
    echo "End time: $(date)"
    echo "Individual log: $LOCAL_LOG"
}

#############################################################
#  Run L1448MM1                                             #
#############################################################

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Processing L1448MM1                   ${NC}"
echo -e "${BLUE}========================================${NC}"

for APERTURE in "${L1448MM1_APERTURES[@]}"; do
    run_fit "L1448MM1" "$APERTURE"
done

#############################################################
#  Run BHR71                                                #
#############################################################

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Processing BHR71                      ${NC}"
echo -e "${BLUE}========================================${NC}"

for APERTURE in "${BHR71_APERTURES[@]}"; do
    run_fit "BHR71" "$APERTURE"
done



#############################################################
#  Summary                                                  #
#############################################################

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  FINAL SUMMARY                         ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total runs:      $TOTAL_RUNS"
echo -e "${GREEN}Successful:      $SUCCESSFUL_RUNS${NC}"
echo -e "${RED}Failed:          $FAILED_RUNS${NC}"
echo ""
echo "End time: $(date)"
echo "Master log: $MASTER_LOG"
echo ""

if [ $FAILED_RUNS -eq 0 ]; then
    echo -e "${GREEN}All fits completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some fits failed. Check logs for details.${NC}"
    exit 1
fi
