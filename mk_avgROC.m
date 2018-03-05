%
% script for compute an average ROC curve from (fpr, tpr) data from 
% several runs of classifiers such as deep net ...
%
num_runs = 20;
di_avg = zeros([num_runs  length([0:0.01:1])]);
prec_avg = zeros([num_runs  length([0:0.01:1])]);

for run=1:num_runs
    
    % CNN's result from color image only
    %raw_data = textread(sprintf('out_%d.txt', run-1));
    
    % CNN's result from color + color segmentation in HSV color space, the parameters of segmentation varied from ... 
    % raw image
    raw_data = textread(sprintf('./facial/cnn_output_2_%d.txt', run-1));
    % multi-part
    %raw_data = textread(sprintf('out_headshoulder_new_%d.txt', run-1));

   

    output_pos = raw_data(:,1); gt = raw_data(:,2);
    
    gt(gt <= 0) = -1; % ground truth 1 for pos. & -1 for negs.

    [di, fi, prec, threshold] = roccurve([output_pos, 1-output_pos], gt, 0.90);

    % ROC curve
    [fi_u, idx] = unique(fi, 'first');
    di_u = di(idx);
    di_avg(run, :) = interp1(fi_u, di_u, [0:0.01:1]);

    % precision-recall curve
    [di_u, idx] = unique(di, 'first');
    prec_u = prec(idx);
    prec_avg(run, :) = interp1(di_u, prec_u, [0:0.01:1]);
    
end

roc_curve = mean(di_avg);
disp(['AUC is ', num2str(sum(roc_curve(1:100)*0.01)) ]);
figure(1); grid on; hold on; plot([0:0.01:1], mean(di_avg), 'b+'); xlabel('false positive rate'); ylabel('true positive rate'); axis([0 1 0 1]);
figure(2); grid on; hold on; plot([0:0.01:1], mean(prec_avg), 'b+'); xlabel('true positive rate'); ylabel('precision'); axis([0 1 0 1]);

