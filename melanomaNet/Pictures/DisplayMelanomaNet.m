%% Display Images from Classification
% Run code from 'Phone Camera Pictures' folder
t = imageDatastore(cd);
test = augmentedImageDatastore([227 227],t);
[y,p] = classify(MelanomaNet,test);
idx = randperm(numel(t.Files),9);
figure
for i = 1:9
    subplot(3,3,i)
    I = readimage(t,idx(i));
    imshow(I)
    label = y(idx(i));
    title(string(label) + ", " + num2str(100*max(p(idx(i),:)),3) + "%");
end
suptitle('MelanomaNet Classification')


