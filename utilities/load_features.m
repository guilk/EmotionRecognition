function [feature] = load_features(folder_path,file,option)
if exist(folder_path,'dir')
    if strcmp(option, 'hog')
        folder_path = [folder_path '/'];
        [feature,~,~] = Read_HOG_files({file},folder_path);
    end
end
end