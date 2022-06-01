clc 
clear all
%load data set
S = load('data_33rpz_cv05.mat');
%create training sets
training_set_20 = S.trn_20.images;
training_set_class_20 = S.trn_20.labels;

training_set_200 = S.trn_200.images;
training_set_class_200 = S.trn_200.labels;

training_set_2000 = S.trn_2000.images;
training_set_class_2000 = S.trn_2000.labels;
%create test sets
test_mat = S.tst.images;
test_class = S.tst.labels;

%create data sets w.t.r. given the second step (x axis=left side-right side,y axis=top side-bottom side)          
dataset_20 = myFunction(training_set_20,training_set_class_20);
dataset_200 = myFunction(training_set_200,training_set_class_200);
dataset_2000 = myFunction(training_set_2000,training_set_class_2000);

%find accuracy
accr_20=find_accr(dataset_20,test_mat,test_class);
accr_200=find_accr(dataset_200,test_mat,test_class);
accr_2000=find_accr(dataset_2000,test_mat,test_class);

%bar plot for the results
%Q1- X1 and X2 features are almost same but X1 is better than X2
%Q2-In this set no. Because result is almost same for every training set
%size. If there are big difference with the features than use.
%Q3-Training size is increase the accuracy also increase. So the system
%will be reliable.

X = categorical({'All Feature','X1','X2'});
X = reordercats(X,{'All Feature','X1','X2'});
comp_arr = [accr_20(1) accr_20(2) accr_20(3);accr_200(1) accr_200(2) accr_200(3);accr_2000(1) accr_2000(2) accr_2000(3)] ;
b = bar(X,comp_arr);

%Function for the create dataset array with second question
function create_x1_x2 = myFunction(x,training_set_class)
    x1 = [];
    x2 = [];
    w = [];
    for i=1: length(x)
         arr = x(:,:,i);
         summation_left = 0;
         summation_right = 0;
         summation_bottom = 0;
         summation_top = 0;
         for j=1: (length(arr)/2)
             summation_left = summation_left + sum(arr(:,j));
         end
         for z= length(arr)/2+1: length(arr)
             summation_right = summation_right + sum(arr(:,z));
         end
          x1(i) = summation_left-summation_right;    
         for j=1: size(arr,2)/2
             summation_top = summation_top + sum(arr(j,:));
         end
         for z= size(arr,2)/2+1: size(arr,2)
             summation_bottom = summation_bottom + sum(arr(z,:));
         end
         x2(i) = summation_top - summation_bottom;
         w(i) = training_set_class(i);
    end
    A1 = {x1,x2,w};
    %concrate three different matrix and create a data 
    data = cat(1,A1{:});
    dataset =transpose(data);
    create_x1_x2 = dataset;
end
%function that find which class the test sample
function calc_accr = myFunction1(test_sample,select,w1_class_mean,w2_class_mean,cov_w1,cov_w2)
    %In order to find which class the data in the given test sample belongs to, the discriminant function in the given document is used. This function takes advantage of the columns in the test data and calculates a result of which class each result may belong to. The test data belongs to the class that is larger than these results.
    accr = 0;
    for j=1:length(test_sample) 
        if select == "all"
            g_1 = -0.5*([test_sample(j,1);test_sample(j,2)]-w1_class_mean')'*inv(cov_w1)*([test_sample(j,1);test_sample(j,2)]-w1_class_mean')+5/2*log(2*pi)-0.5*log(det(cov_w1));
            g_2 = -0.5*([test_sample(j,1);test_sample(j,2)]-w2_class_mean')'*inv(cov_w2)*([test_sample(j,1);test_sample(j,2)]-w2_class_mean')+5/2*log(2*pi)-0.5*log(det(cov_w2));
            w = test_sample(j,3);
            if(g_1>=g_2 && w == 1)
                x=[test_sample(j,:),"class is 1"];
                accr = accr + 1;
            end
            if(g_1>=g_2 && w == 2)
                x=[test_sample(j,:),"class is 1"];
                disp(x)
            end
            if(g_2>g_1 && w == 2)
                x=[test_sample(j,:),"class is 2"];
                accr = accr +1;
            end
            if(g_2>g_1 && w == 1)
                x=[test_sample(j,:),"class is 2"];
                disp(x)
            end
        end
        if select == "x1"
            g_1 = -0.5*([test_sample(j,1)]-w1_class_mean')'*inv(cov_w1)*([test_sample(j,1)]-w1_class_mean')+5/2*log(2*pi)-0.5*log(det(cov_w1));
            g_2 = -0.5*([test_sample(j,1)]-w2_class_mean')'*inv(cov_w2)*([test_sample(j,1)]-w2_class_mean')+5/2*log(2*pi)-0.5*log(det(cov_w2));
            w = test_samples(j,3);
            if(g_1>=g_2 && w == 1)
                x=[test_sample(j,:),"class is 1"];
                accr = accr + 1;
            end
            if(g_1>=g_2 && w == 2)
                x=[test_sample(j,:),"class is 1"];
                disp(x)
            end
            if(g_2>g_1 && w==2)
                x=[test_sample(j,:),"class is 2"];
                accr = acrr +1;
            end
            if(g_2>g_1 && w==1)
                x=[test_sample(j,:),"class is 2"];
                disp(x)
            end
        end
        if select == "x1"
            g_1 = -0.5*([test_sample(j,1)]-w1_class_mean')'*inv(cov_w1)*([test_sample(j,1)]-w1_class_mean')+5/2*log(2*pi)-0.5*log(det(cov_w1));
            g_2 = -0.5*([test_sample(j,1)]-w2_class_mean')'*inv(cov_w2)*([test_sample(j,1)]-w2_class_mean')+5/2*log(2*pi)-0.5*log(det(cov_w2));
            w = test_samples(j,3);
            if(g_1>=g_2 && w == 1)
                x=[test_sample(j,:),"class is 1"];
                accr = accr + 1;
            end
            if(g_1>=g_2 && w == 2)
                x=[test_sample(j,:),"class is 1"];
                disp(x)
            end
            if(g_2>g_1 && w==2)
                x=[test_sample(j,:),"class is 2"];
                accr = acrr +1;
            end
            if(g_2>g_1 && w==1)
                x=[test_sample(j,:),"class is 2"];
                disp(x)
            end
        end
    end
    calc_accr = accr;
end
%function that finds mean 
function mean = myFunction3(dataset)
    x1w1_mean = 0;% x1 attr mean for normal distribution class 1
    x2w1_mean = 0;% x2 attr mean for normal distribution class 1

    x1w2_mean = 0;% x1 attr mean for normal distribution class 2
    x2w2_mean = 0;% x2 attr mean for normal distribution class 2

    %It will be used to show the numbers that the w1,w2 and w3 classes have.
    num_w1=0;
    num_w2=0;

    %We split the big data set according to the given 3 classes.
    w1_class = [];
    w2_class = [];
    w1_class_mean =[];
    w2_class_mean =[];

    for i=1: length(dataset)
        if(dataset(i,3)== 1.0)
            x1w1_mean = dataset(i,1)+x1w1_mean;
            x2w1_mean = dataset(i,2)+x2w1_mean;
            num_w1=num_w1+1;
            w1_class(num_w1,1) = dataset(i,1);
            w1_class(num_w1,2) = dataset(i,2);
        end
        if(dataset(i,3) == 2.0)
            x1w2_mean = dataset(i,1)+x1w2_mean;
            x2w2_mean = dataset(i,2)+x2w2_mean; 
            num_w2=num_w2+1;
            w2_class(num_w2,1) = dataset(i,1);
            w2_class(num_w2,2) = dataset(i,2);
        end
    end

    %The mean was calculated for different classes and different descriptive columns.
    %We defined these averages as a matrix in order to be able to use them comfortably in the next process.
    x1w1_mean = x1w1_mean/num_w1;
    x2w1_mean = x2w1_mean/num_w1;

    x1w2_mean = x1w2_mean/num_w2;
    x2w2_mean = x2w2_mean/num_w2;

    w1_class_mean = [x1w1_mean,x2w1_mean]
    w2_class_mean = [x1w2_mean,x2w2_mean]

    all_class_mean = containers.Map({1,2,3,4}, {w1_class_mean, w2_class_mean, w1_class,w2_class});
    mean = values(all_class_mean);
end
%function that find covariance for the dataset
function cov =myFunction4(w1_class,w2_class)
    num_w1=length(w1_class);
    num_w2=length(w2_class);

    %This time, a covariance matrix was created for each class separately.
    one_matrix_w1=ones(num_w1);
    w1 = w1_class -(one_matrix_w1*w1_class*1/num_w1);
    cov_w1 = (transpose(w1) * w1);
    cov_w1 = cov_w1/(length(w1_class)-1)

    one_matrix_w2=ones(num_w2);
    w2 = w2_class -(one_matrix_w2*w2_class*1/num_w2);
    cov_w2 = (transpose(w2) * w2);
    cov_w2 = cov_w2/(length(w2_class)-1)

    cov = values(containers.Map({1,2}, {cov_w1, cov_w2}));
end
%function that find accuracy for all x1,x2 and both x1,x2
function accr_arr=find_accr(dataset,test_mat,test_class)
    arr=[];
    mean_arr = myFunction3(dataset);
    w1_class_mean = cell2mat(mean_arr(1));
    w2_class_mean = cell2mat(mean_arr(2));

    w1_class = cell2mat(mean_arr(3));
    w2_class= cell2mat(mean_arr(4));

    cov_arr = myFunction4(w1_class,w2_class);
    cov_w1=cell2mat(cov_arr(1));
    cov_w2=cell2mat(cov_arr(2));

    test_sample = myFunction(test_mat,test_class);
    all_accr = myFunction1(test_sample,"all",w1_class_mean,w2_class_mean,cov_w1,cov_w2)
    x1_accr = myFunction1(test_sample,"all",w1_class_mean(1),w2_class_mean(1),cov_w1(1,1),cov_w2(1,1))
    x2_accr = myFunction1(test_sample,"all",w1_class_mean(2),w2_class_mean(2),cov_w1(2,2),cov_w2(2,2))

    arr(1)=all_accr;
    arr(2)=x1_accr;
    arr(3)=x2_accr;

    accr_arr=arr;
end