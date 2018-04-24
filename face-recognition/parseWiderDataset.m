% Creating the full path dataset. 
function [finaltbl] = parseWiderDataset(datasetFileName,relativepath)    
    dataset=load(datasetFileName);
    tbl=table(dataset.event_list,dataset.file_list,dataset.face_bbx_list);
    filename=[];box={};    
    count=1;
    finaltbl = cell2table(cell(0,2));    
    for i=1:height(tbl)
        row=tbl(i,:);
        path=strcat(relativepath,'/',row.Var1);
        for j=1:numel(row.Var2{1})
            tpath=strcat(path,'/',row.Var2{1}{j},'.jpg');            
            finaltbl=[finaltbl;cell2table({char(tpath),{double(row.Var3{1}{j})}})];            
            count=count +1;
        end
    end   
    finaltbl.Properties.VariableNames={'filename','box'};
end