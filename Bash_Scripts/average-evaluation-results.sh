# Usage example: sh ./Bash_Scripts/average-evaluation-results.sh Results/MNIST/Logs/*bits-2.test.logit git

files_to_average_from="$@" # e.g.: *_bits-2.test.log

# -- Prints -- #
echo "AVG scores are created from files =>"
for file in $files_to_average_from ;
do echo -e ' \t '$file ;
done ;

# -- Do average -- #
awk 'FNR == 1 { nfiles++; ncols = NF }
     { for (i = 1; i <= NF; i++) sum[FNR,i] += $i
       if (FNR > maxnr) maxnr = FNR
     }
     END {
         for (line = 1; line <= maxnr; line++)
         {
             for (col = 1; col <= ncols; col++)
                  printf "  %f", sum[line,col]/nfiles;
             printf "\n"
         }
     }' $files_to_average_from