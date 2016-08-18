function [C, order] = plot_confusion_matrix(groundtruth, predictions)
src_class = {'Neutral', 'Angry','Disgust','Fear','Happy','Sad','Surprise'};
[C, order] = confusionmat(groundtruth, predictions);
tick_label = src_class(order+1);
clabel = unique(groundtruth);
num_class = length(clabel);
mat = C ./ repmat(sum(C,2),1,num_class) * 100;

imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:num_class);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:num_class,...                         %# Change the axes tick marks
        'XTickLabel',tick_label,...  %#   and tick labels
        'YTick',1:num_class,...
        'YTickLabel',tick_label,...
        'TickLength',[0 0]);
xlabel('Groundtruth');
ylabel('Prediction');

end