#!/bin/bash

RESULTS="sweep.csv"
GEMM_CU="gemm.cu"

TH_VALUES=(4 8 16 32)
BW_VALUES=(64 128 256)
BH_VALUES=(64 128 256)
BK_VALUES=(8 16 32 64)
WW_VALUES=(32 64 128)
WH_VALUES=(32 64 128)
W_HOR_VALUES=(1 2 4)

update_params() {
    local TH=$1 BW=$2 BH=$3 BK=$4 WW=$5 WH=$6 w_hor=$7

    sed -i "s/^constexpr int TH = .*/constexpr int TH = $TH;/" $GEMM_CU
    sed -i "s/^constexpr int BW = .*/constexpr int BW = $BW;/" $GEMM_CU
    sed -i "s/^constexpr int BH = .*/constexpr int BH = $BH;/" $GEMM_CU
    sed -i "s/^constexpr int BK = .*/constexpr int BK = $BK;/" $GEMM_CU
    sed -i "s/^constexpr int WW = .*/constexpr int WW = $WW;/" $GEMM_CU
    sed -i "s/^constexpr int WH = .*/constexpr int WH = $WH;/" $GEMM_CU
    sed -i "s/^constexpr int w_hor = .*/constexpr int w_hor = $w_hor;/" $GEMM_CU

}

total_configs="$((${#TH_VALUES[@]} * ${#BW_VALUES[@]} * ${#BH_VALUES[@]} * ${#BK_VALUES[@]} * ${#WW_VALUES[@]} * ${#WH_VALUES[@]} * ${#W_HOR_VALUES[@]} ))"

config_num=0

echo "" > $RESULTS
echo "TH,BW,BH,BK,WW,WH,w_hor,CUBLAS_GFLOPS,KERNEL_GFLOPS,PERCENT" > $RESULTS

for TH in "${TH_VALUES[@]}"; do
for BW in "${BW_VALUES[@]}"; do
for BH in "${BH_VALUES[@]}"; do
for BK in "${BK_VALUES[@]}"; do
for WW in "${WW_VALUES[@]}"; do
for WH in "${WH_VALUES[@]}"; do
for w_hor in "${W_HOR_VALUES[@]}"; do

    ((config_num++))
    update_params $TH $BW $BH $BK $WW $WH $w_hor
    echo "[$config_num/$total_configs] Testing: TH=$TH BW=$BW BH=$BH BK=$BK WW=$WW WH=$WH w_hor=$w_hor"
                      
    if ! make; then
        echo "  Build failed. Skipping..."
        continue
    fi

    OUTPUT=$(timeout 30s make run 2>&1 | tee /dev/tty) || { echo "  Run failed or timed out"; continue; }


    CUBLAS_GFLOPS=$(echo "$OUTPUT" | grep "CUBLAS Performance" -A 3 | grep "GFLOPs:" | tail -1 | awk '{print $2}')                  
    KERNEL_GFLOPS=$(echo "$OUTPUT" | grep "Kernel Performance" -A 3 | grep "GFLOPs:" | tail -1 | awk '{print $2}')
                                
    if [[ -z "$CUBLAS_GFLOPS" ]] || [[ -z "$KERNEL_GFLOPS" ]]; then
        echo "  FAILED TO PARSE OUTPUT"
        continue
    fi
                                
    PERCENT=$(echo "scale=2; 100 * $KERNEL_GFLOPS / $CUBLAS_GFLOPS" | bc)              
    echo "  CUBLAS=$CUBLAS_GFLOPS GFLOPS, Kernel=$KERNEL_GFLOPS GFLOPS ($PERCENT%)"
    echo "$TH,$BW,$BH,$BK,$WW,$WH,$w_hor,$CUBLAS_GFLOPS,$KERNEL_GFLOPS,$PERCENT" >> $RESULTS

done
done
done
done
done
done
done
