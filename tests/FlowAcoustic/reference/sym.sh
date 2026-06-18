#!/bin/bash
# Keyed (by z) comparison of z-NSCBC vs z-truth near zlo (z~-0.4) and zhi (z~+0.4).
awk 'BEGIN{ while((getline l < "/tmp/pz_t.txt")>0){ if(l ~ /^#/) continue; n=split(l,a," "); if(n>=2){ k=sprintf("%.4f",a[1]+0); T[k]=a[2] } } }
     !/^#/ && NF>=2 { z=$1+0; k=sprintf("%.4f",z);
       if(z>-0.43 && z<-0.39) printf "zlo z=%+.4f  p_N=%.8f  p_T=%.8f  diff=%+.2e\n", z, $2, T[k], $2-T[k];
       if(z> 0.39 && z< 0.43) printf "zhi z=%+.4f  p_N=%.8f  p_T=%.8f  diff=%+.2e\n", z, $2, T[k], $2-T[k]; }' /tmp/pz_n.txt
