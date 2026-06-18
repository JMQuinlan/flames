#!/bin/bash
# Compare z-axis pressure profiles of two slice files (cols: z, pressure).
# Usage: bash zprofile_cmp.sh fileN fileW
N=${1:-/tmp/pz_n.txt}
W=${2:-/tmp/pz_w.txt}
paste <(grep -vE '^#|^$' "$N" | awk '{print $1, $2}') \
      <(grep -vE '^#|^$' "$W" | awk '{print $2}') \
  | awk '{
      dn=($2>1)?$2-1:1-$2; dw=($3>1)?$3-1:1-$3;
      if(dn>mn)mn=dn; if(dw>mw)mw=dw;
      df=($2>$3)?$2-$3:$3-$2; if(df>md){md=df; zd=$1}
    } END {
      printf "  NSCBC axis max|p-1| = %.5g\n", mn;
      printf "  Wall  axis max|p-1| = %.5g\n", mw;
      printf "  max|NSCBC - Wall|   = %.5g  at z = %.3f\n", md, zd;
    }'
echo "--- profile near z-hi face (last 8 rows: z  p_NSCBC  p_Wall) ---"
paste <(grep -vE '^#|^$' "$N" | awk '{print $1, $2}') \
      <(grep -vE '^#|^$' "$W" | awk '{print $2}') | tail -8
