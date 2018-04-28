function [finaltbl] = parseFddbDataset(relativePath)
    numOfFiles=10;
    C={};
    numOfImages=1;
%     minheight=inf;
%     minwidth=inf;
%     maxheight=-inf;
%     maxwidth=-inf;
    for n=1:numOfFiles
        if n<10
            datasetFileName=[relativePath 'FDDB-fold-0' num2str(n) '-ellipseList.txt'];
        else
            datasetFileName=[relativePath 'FDDB-fold-' num2str(n) '-ellipseList.txt'];
        end
        fid = fopen(datasetFileName);
        while ~feof(fid)
            A = fgetl(fid);
            A=[A '.jpg'];
%             B=imread(['../data/' A '.jpg']);
%             [H,W]=size(B);
%             if(minheight>H)
%                 minheight=H;
%             end
%             if(minwidth>W)
%                 minwidth=W;
%             end
%             if(maxheight<H)
%                 maxheight=H;
%             end
%             if(maxwidth<W)
%                 maxwidth=W;
%             end
            C(numOfImages,1)=cellstr(A);
            A = fgetl(fid);
            numOfBox=str2num(A);
            paramArray=zeros(numOfBox,4);
            for i=1:numOfBox
                A=fgetl(fid);
                param=split(A);
                height=2*str2double(cell2mat(param(1)));
                width=2*str2double(cell2mat(param(2)));
                left_x=str2double(cell2mat(param(4)))-str2double(cell2mat(param(2)));
                top_y=str2double(cell2mat(param(5)))-str2double(cell2mat(param(1)));
                paramArray(i,:)=[left_x top_y width height];
%                 if(maxheight<height)
%                     maxheight=height;
%                 end
%                 if(minheight>height)
%                     minheight1=minheight;
%                     minheight=height;
%                 end
%                 for j=1:6
%                     param()
%                     paramArray(i,j)=str2double(param(j));
%                 end
            end
            C(numOfImages,2)=num2cell(paramArray,[1,2]);
            numOfImages=numOfImages+1;
        end
        finaltbl=cell2table(C);
        fclose(fid);
    end
    finaltbl.Properties.VariableNames={'filename','box'};
    save(strcat(relativePath,'/parseFDDB_dataset.mat'),'finaltbl');
    % Read one of the images.
%     imgNo=3;
%     I = imread('../data/FDDB/2002/07/19/big/img_423.jpg');
%     I = insertShape(I, 'Rectangle', finaltbl.box{imgNo});
%     figure
%     imshow(I);
end