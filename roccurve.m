function [di, fp, prec, threshold] = roccurve(output,  label, specified_di)
%
% Calculates detection rate, false positive rates, precision from the
% output whose values are not binary but continuous, and class labels.
% 
%

[outsorted, inds] = sort(output);
labelsorted = label(inds);

NPOS = sum(label==1);                    % Total number of positive samples
NNEG = sum(label==-1);                   % Total number of negative samples
RNPOS = 0;                                      % Running sum of the # of positive samples
RNNEG = 0;                                      % Running sum of the # of negative samples

di = zeros([length(outsorted), 1]);
fp = zeros([length(outsorted), 1]);
prec = zeros([length(outsorted), 1]);

for q=1:length(outsorted)

    di(q) = 1 - RNPOS / NPOS;  % detection, recall
    fp(q) = 1 - RNNEG / NNEG;  % false positive rate,
    prec(q) = (NPOS - RNPOS) / (NPOS - RNPOS + NNEG - RNNEG);  % precision

    % update SPOS and SNEG for the next iteration
    if labelsorted(q) == 1
        RNPOS = RNPOS + 1;
    else
        RNNEG = RNNEG + 1;
    end
end

% -- get FP images at an operating point that is specified by a particular tpr, e.g., 0.7
[diff_val, ind] = min(abs(di-specified_di));
threshold = outsorted(ind);
