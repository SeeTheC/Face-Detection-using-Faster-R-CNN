% downsample the image and then save the image
% NOTE: maxWidth NOT USED
function [downsampleTable,noOfDataPoint] = downsampeDataSet(datasetPath,maxDim,savepath,saveLogPath,basepath)
    %% INIT        
    imgDataset=load(datasetPath);
    imgDataset=imgDataset.dataset;
    noOfImage=height(imgDataset);
    maxheight=maxDim(1);maxWidth=maxDim(2);
    downsampleTable = cell2table(cell(0,2));        
    fprintf('Total image to Process in this Dataset:%d\n',noOfImage);
    
    % Config
    minArea=16*16;    
    %% Downsampling
    downsampleImgInfo=strcat(saveLogPath,'/downsample_info.txt');
    dropImgInfo=strcat(saveLogPath,'/drop_info.txt');
    
    fid = fopen(downsampleImgInfo, 'w+');
    dropfid = fopen(dropImgInfo, 'w+');
    fprintf(fid,'filename#\torg_h\torg_w\tdownscale_by\tdwn_h\tdwn_w\n');
    fprintf(dropfid,'filename#\torg_h\torg_w\tdownscale_by\tdwn_h\tdwn_w\n');
    dropCount=1;
    for i=1:noOfImage
        row=imgDataset(i,:);
        filename=row.filename{1};
        box=row.box{1};
        img=imread(strcat(basepath,'/',filename));
        h=size(img,1);w=size(img,2);
        
        if ( h > maxheight && (h/maxheight)>=1.5)          
            downsample=(h/maxheight);
            %fprintf('Downsample: %f h:%d w:%d \t filename:%s\n',downsample,h,w,filename);
            rimg=imresize(img,1/downsample);            
            noOfBox = size(box,1);
            newBox=[];
            for j=1:noOfBox
                % x1, y1, w, h,
                b=box(j,:);
                newB=double(round(b./downsample));
                area=newB(3)*newB(4);
                % If min area of the face should be greater than minArea
                if area > minArea
                    newBox=[newBox;newB];                          
                end
            end            
        else
            downsample=1;
            rimg=img;
            newBox=box;
        end        
        nh=size(rimg,1);nw=size(rimg,2);        
        if size(newBox,1) >0
            path=strcat(savepath,'/',filename);
            folder=split(path,'/');folder= join(folder(1:end-1),'/');folder=folder{1};
            if ~(exist(folder))
                mkdir(folder);
            end
            imwrite(rimg,path);
            downsampleTable=[downsampleTable;cell2table({filename,{newBox}})]; 
            fprintf(fid,'%s\n%d\t%d\t%f\t%d\t%d\n',filename,h,w,downsample,nh,nw);
        else            
            fprintf('**%d)DROPPING IMAGE %s:\n',dropCount,filename);
            fprintf(dropfid,'%s\n%d\t%d\t%f\t%d\t%d\n',filename,h,w,downsample,nh,nw);
            dropCount=dropCount+1;
        end
        
    end
    %% Finishing
    fclose(fid);
    fclose(dropfid);
    noOfDataPoint=height(downsampleTable);
    downsampleTable.Properties.VariableNames={'filename','box'};
    fprintf('\n-->No of datapoint downsample:%d\n',noOfDataPoint);
end

