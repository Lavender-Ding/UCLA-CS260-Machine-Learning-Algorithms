clc;

load('boundary.mat');

x=0:0.01:1;
y=0:0.01:1;
[X,Y]=meshgrid(x,y);
X=reshape(X,1,[])';
Y=reshape(Y,1,[])';

train_data=features;
train_label=labels;

new_data=[X,Y];

j=0;
for k=[1,5,15,20]
    j=j+1;
    t_size=size(train_data,1);
    n_size=size(new_data,1);

    PredictLabel=zeros(n_size,1);

    for i=1:n_size
        tmp=repmat(new_data(i,:),t_size,1);
        dismap=(tmp-train_data).^2;
        distance=sum(dismap,2);
        [dis,num]=sort(distance);
        K_near=num(1:k,1);
        K_near_labels=train_label(K_near,1);
        labelCount=tabulate(K_near_labels);
        [maxVal,maxLabel]=max(labelCount(:,2));
        maxLabel=labelCount(labelCount(:,2)==maxVal,1);
        randomLabel=maxLabel(randperm(length(maxLabel)));
        PredictLabel(i,1)=randomLabel(1,1);
    %     newPredictLabel(i,1)=labelCount(maxLabel,1);
    end
    
    figure;
    %ax=subplot(2,2,j);
    scatter(X(PredictLabel==1,1),Y(PredictLabel==1,1),[],'k','x');
    hold on;
    scatter(X(PredictLabel==-1,1),Y(PredictLabel==-1,1),[],'y','o');
    %hold on;
    %title(sprintf('k=%d',k));
end


