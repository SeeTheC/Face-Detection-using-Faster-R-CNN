function [avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,minDim,testData,saveModelPath)
    fprintf('Evaluating for Test set...\n');
    resultsStruct = struct([]);
    savedTestImgPath=strcat(saveModelPath,'/test_img');
    mkdir(savedTestImgPath);
    negImgInfo=strcat(savedTestImgPath,'/neg_img_info.txt');
    negfid = fopen(negImgInfo, 'w+');   
    wrongTesImg=[];
    for i = 1:height(testData)      
        filename=testData.filename{i};
        %Temp: filename=strcat(filename,'.jpg');
        [status,bboxes]=isTestImgValidGroundTruth(testData.box{i});
        testData.box{i}=bboxes;
        if ~status            
            wrongTesImg=[wrongTesImg,i];            
        end
        I = imread(filename);   
        if(size(I,1)<minDim(1) || size(I,2)<minDim(2))
            fprintf('Error:Dim of Image is less than required [%d,%d] dim:[%d,%d]\n',minDim(1),minDim(2),size(I,1),size(I,2))
            continue;
        end        
        [bboxes, scores, labels] = detect(detector, I);        
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;        
        % save image after creating bounding box
       saveImage(I,bboxes,scores,filename,savedTestImgPath)
        if size(bboxes,1) ==0
            sfn=split(filename,'/');
            fprintf('** NO FACE FOUND:%s/%s\n',sfn{end-1},sfn{end});
            fprintf(negfid,'NO FACE FOUND:%s/%s\n',sfn{end-1},sfn{end});
            
        end
    end  
    fclose(negfid);
    
    % Convert the results into a table.
    result = struct2table(resultsStruct);
    % Removing Wrong Image
    testData(wrongTesImg,:)=[];
    result(wrongTesImg,:)=[];
    
    disp(wrongTesImg);
    
    % Saving
    savepath=strcat(saveModelPath,'/','test_result.mat');
    save(savepath,'result');
    fprintf('Completed..\n');
   
    fprintf('Evaluate the object detector using Average Precision metric...\n');
    expectedResults = testData(:, 2:end);
    [avgPrecision, recall, precision] = evaluateDetectionPrecision(result, expectedResults);    
    tbl=table(precision,recall);
    savepath=strcat(saveModelPath,'/','precision_recall.mat');    
    save(savepath,'tbl');
    fprintf('Completed\n');    
    fprintf('**Avg Precision of Dectector:%f\n',avgPrecision);
    tblPrecsionRecall=tbl;    
end

% Drawing box and Saving Image    
function saveImage(img,bboxes,scores,filename,savedTestImgPath)
        sfn=split(filename,'/');
        folder=strcat(savedTestImgPath,'/',sfn{end-2},'/',sfn{end-1});  
        if ~(exist(folder))
            mkdir(folder);
        end
        if size(bboxes,1)>0
            boxImg = insertObjectAnnotation(img,'rectangle',bboxes,scores);        
        else            
            boxImg=img;
        end
        imwrite(boxImg,strcat(folder,'/',sfn{end}));        
end

function [status,bboxes]=isTestImgValidGroundTruth(bboxes)
    n=size(bboxes,1);
    status=true;
    wrongBox=[];
    for i=1:n
        if(bboxes(i,1) < 1 || bboxes(i,2) < 1 || bboxes(i,3) < 1 || bboxes(i,4) < 1)
            wrongBox=[wrongBox,i];            
        end
    end
    if numel(wrongBox)>0
        bboxes(wrongBox,:)=[];
    end
    if size(bboxes,1)<1
        status = false;
    end
end

