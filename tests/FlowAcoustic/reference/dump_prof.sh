#!/bin/bash
# Show z-NSCBC vs z-truth pressure near both z-faces and the max-error location.
paste <(grep -vE '^#|^$' /tmp/pz_n.txt | awk '{print $1, $2}') \
      <(grep -vE '^#|^$' /tmp/pz_t.txt | awk '{print $2}') > /tmp/cmp.txt
echo "  z         p_NSCBC        p_truth        diff"
echo "--- near z = -0.5 (zlo) ---"; head -5 /tmp/cmp.txt | awk '{printf "%8.4f  %12.8f  %12.8f  %+.2e\n",$1,$2,$3,$2-$3}'
echo "--- near z =  0.0 (mid) ---"; sed -n '23,26p' /tmp/cmp.txt | awk '{printf "%8.4f  %12.8f  %12.8f  %+.2e\n",$1,$2,$3,$2-$3}'
echo "--- near z = +0.5 (zhi) ---"; tail -5 /tmp/cmp.txt | awk '{printf "%8.4f  %12.8f  %12.8f  %+.2e\n",$1,$2,$3,$2-$3}'
