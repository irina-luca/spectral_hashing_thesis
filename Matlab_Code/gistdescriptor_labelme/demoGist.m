fileID = fopen('LabelMeGist.data','w');
% fmt = '%5d %5d %5d %5d\n';
% fprintf('%d',round(a));



imagefiles = dir('./*.jpg');      

nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
    try
        currentfilename = imagefiles(ii).name;
    %     disp(currentfilename)
        % Load image
        img1 = imread(currentfilename);

        % Parameters:
        clear param
        param.imageSize = [256 256]; % it works also with non-square images
        param.orientationsPerScale = [8 8 8 8];
        param.numberBlocks = 4;
        param.fc_prefilt = 4;

        % Computing gist requires 1) prefilter image, 2) filter image and collect
        % output energies
        [gist1, param] = LMgist(img1, '', param);

        gist1_str = strjoin(string(gist1),' ')
        filename_str = string(currentfilename)
        gist_joined = strjoin(horzcat(horzcat(ii, filename_str + ' '),gist1_str), ' ')
    %     disp(gist_joined)
        fprintf(fileID,'%s\n',gist_joined);

    catch
      continue
    end
%     disp("test")
%     disp(isstring(string(gist1)))
%     disp(isstring(string(imagefiles(ii).name)))
%     gist1_str = string(gist1)
%     imagename_str = string(imagefiles(ii).name)
%     data_line_array = [imagename_str gist1_str]
%     data_line_str = strjoin(data_line_array, ' ')

%     disp()

%     disp(data_line_str)
%     dlmwrite('LabelMeGist.data', gist_joined,'-append','delimiter',' ') %
%     only appends reals array to file, not strings array
%     dlmwrite('LabelMeGist.data', gist1, ' ')
%     dlmwrite('LabelMeGist.data',gist1,'-append')
%     fprintf(fileID, '\n');
   %    currentimage = imread(currentfilename);
%    images{ii} = currentimage;
end


fclose(fileID);



% % EXAMPLE 1
% % Load image
% img1 = imread('demo1.jpg');
% 
% % Parameters:
% clear param
% param.imageSize = [256 256]; % it works also with non-square images
% param.orientationsPerScale = [8 8 8 8];
% param.numberBlocks = 4;
% param.fc_prefilt = 4;
% 
% % Computing gist requires 1) prefilter image, 2) filter image and collect
% % output energies
% [gist1, param] = LMgist(img1, '', param);
% disp("descriptor is =>")
% disp(gist1)
% disp(param)






% % Visualization
% figure
% subplot(121)
% imshow(img1)
% title('Input image')
% subplot(122)
% showGist(gist1, param)
% title('Descriptor')


% % EXAMPLE 2
% % Load image (this image is not square)
% img2 = imread('demo2.jpg');
% 
% % Parameters:
% clear param 
% %param.imageSize. If we do not specify the image size, the function LMgist
% %   will use the current image size. If we specify a size, the function will
% %   resize and crop the input to match the specified size. This is better when
% %   trying to compute image similarities.
% param.orientationsPerScale = [8 8 8 8];
% param.numberBlocks = 4;
% param.fc_prefilt = 4;
% 
% % Computing gist requires 1) prefilter image, 2) filter image and collect
% % output energies
% [gist2, param] = LMgist(img2, '', param);
% 
% % Visualization
% figure
% subplot(121)
% imshow(img2)
% title('Input image')
% subplot(122)
% showGist(gist2, param)
% title('Descriptor')



