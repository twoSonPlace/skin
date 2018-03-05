directory = './180109_dermapro_raw/';
target_directory = './datasets_backup/training/';
RECT = [1 250 447 447];
for k=0:9
    
    subdir = sprintf('Grade%d', k);
    fulldir = [directory, subdir];
    
    d = dir([fulldir, '/*.bmp']);
    
    for i=1:length(d)
        disp(d(i).name);
        im = imread([[fulldir, '/'], d(i).name]);
        %imshow(im);
        im_cropped = imcrop(im, RECT);
        
        new_dir = [[target_directory, subdir] , '/'];
        im_cropped_resized = imresize(im_cropped, 0.5);
        imwrite(im_cropped_resized, [new_dir, sprintf('G%d_', k), d(i).name]);
        %imshow(im_cropped_resized);
        
    end
end
