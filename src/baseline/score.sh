#!/usr/bin/env bash


for t in 1 .5 .25 .1 .05
do
  th score.lua --mode ri --cuda\
      --modelDir "data/commondefs/models/m2"\
      --entryFile "data/commondefs/models/m2/test_samples"$t"_rank.txt"\
      --batchSize 40\
      --outputFile "data/commondefs/models/m2/score_test_samples"$t"_rank.txt"\
      --RICharCNN --RIHypernym\
      --dataType labeled
  th score.lua --mode ri --cuda\
      --modelDir "data/commondefs/models/m2"\
      --entryFile "data/commondefs/models/m2/valid_samples"$t"_rank.txt"\
      --batchSize 40\
      --outputFile "data/commondefs/models/m2/score_valid_samples"$t"_rank.txt"\
      --RICharCNN --RIHypernym\
      --dataType labeled
done

# set models "m2" "m3" "m4"
# 
# for m in $models
#   for t in $temperatures
#     th score.lua --mode ri --cuda --modelDir "data/commondefs/models/"$m --entryFile "data/commondefs/models/"$m"/test_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/"$m"/score_test_samples"$t"_rank.txt"
#     th score.lua --mode ri --cuda --modelDir "data/commondefs/models/"$m --entryFile "data/commondefs/models/"$m"/valid_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/"$m"/score_valid_samples"$t"_rank.txt"
#   end
# end
# 
# 
# for t in $temperatures
#   th score.lua --mode ri --RICharCNN --cudnnCNN --cuda --modelDir "data/commondefs/models/m5" --entryFile "data/commondefs/models/m5/test_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m5/score_test_samples"$t"_rank.txt"
#   th score.lua --mode ri --RICharCNN --cudnnCNN --cuda --modelDir "data/commondefs/models/m5" --entryFile "data/commondefs/models/m5/valid_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m5/score_valid_samples"$t"_rank.txt"
# end
# 
# 
# for t in $temperatures
#   th score.lua --mode ri --RICharCNN --RIHypernym --cudnnCNN --cuda --modelDir "data/commondefs/models/m6" --entryFile "data/commondefs/models/m6/test_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m6/score_test_samples"$t"_rank.txt"
#   th score.lua --mode ri --RICharCNN --RIHypernym --cudnnCNN --cuda --modelDir "data/commondefs/models/m6" --entryFile "data/commondefs/models/m6/valid_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m6/score_valid_samples"$t"_rank.txt"
# end
