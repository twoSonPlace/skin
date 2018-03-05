fn = './file_and_coordinate2.txt';
%directory = './180109_dermapro_raw/';        % the directory where images are stored
%target_directory = './datasets_backup2/training/';

directory = './180214_dermapro_filtered/';        % the directory where images are stored
target_directory = './datasets_backup2/training/';

% fp = fopen(fn, 'wt');
% %save the point that mouse click has been made along with the file names
% % 
% for k=3:1:4
%     
%     subdir = sprintf('Grade%d', k);
%     fulldir = [directory, subdir];
%     
%     d = dir([fulldir, '/*.bmp']);
% 
%     for i=1:length(d)
% 
%         im = imread([fulldir, '/', d(i).name]);
% 
%         imshow(im);
%         [x, y] = ginput(1);
%         disp(['(x, y) = ', num2str(x), ' ', num2str(y)]); 
% 
%         % write the file name and its (x,y) coordinates at which mouse click is made
%         [height, width, ch] = size(im);
%         fprintf(fp, '%s\t%d\t%d\t%d\t%d\n', [fulldir, '/', d(i).name], height, width, int16(x), int16(y));
%     end
% end
% 
% fclose(fp);


%crop the ROI based on the mouse click point


CROP_SIZE_w = 500;   % the size (height and width) of the cropped area
CROP_SIZE_h = 400;

[names, height, width, xs, ys] = textread(fn, '%s%d%d%d%d', 'delimiter', '\t');

for i=1:length(xs)
    
    [path, fname, ext] = fileparts(names{i});
    im = imread(names{i});
    %figure(1); imshow(im);
    
    % left-sided or rightsided?
    if xs(i) < width(i) / 2  % left
        RECT = [xs(i), max(0, ys(i) - CROP_SIZE_h /2),  CROP_SIZE_w-1, CROP_SIZE_h-1]; % [xmin, ymin, width, height]  
        im_cropped = imcrop(im, RECT);
    else
        RECT = [max(0, xs(i)-CROP_SIZE_w), max(0, ys(i) - CROP_SIZE_h /2),  CROP_SIZE_w-1, CROP_SIZE_h-1];
        im_cropped = imcrop(im, RECT);
        im_cropped = flipdim(im_cropped, 2);
    end
    
    %figure(2); imshow(im_cropped);
    
    str = strsplit(path, '/');
    subdir = str{3};
    new_dir = [[target_directory, subdir] , '/'];
    im_cropped_resized = imresize(im_cropped, 0.5);
    imwrite(im_cropped_resized, [new_dir, subdir, fname, '.bmp']);    
   
        
    %pause;
    
    end
    