function num_label = get_label(str_label)
if strcmp(str_label, 'Neutral') || strcmp(str_label,'NE')
    num_label = 0;
elseif strcmp(str_label,'Angry') || strcmp(str_label,'AN') 
    num_label = 1;
elseif strcmp(str_label,'Disgust') || strcmp(str_label,'DI') 
    num_label = 2;
elseif strcmp(str_label, 'Fear') || strcmp(str_label,'FE') 
    num_label = 3;
elseif strcmp(str_label, 'Happy') || strcmp(str_label,'HA') 
    num_label = 4;
elseif strcmp(str_label, 'Sad') || strcmp(str_label,'SA') 
    num_label = 5;
elseif strcmp(str_label, 'Surprise') || strcmp(str_label,'SU') 
    num_label = 6;
end
        
end